# Detail Layer M1 (Python Prototype) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a DetailNet that adds stable, FlashVSR-like texture on top of Apple AI's 4K output for 720p sources, and decide go/no-go on the bench gates.

**Architecture:**
- **Swift side.** The existing CLI gains dump options that record exactly what the app will later feed the network: decoded source frames, Apple's 4K base, and VideoToolbox optical flow.
- **Python side (`detailnet/`).**
  - Samples training sequences from the shot maps and makes FlashVSR teacher targets plus real-1080p ground-truth targets.
  - Trains a recurrent, flow-warped, cut-reset residual network on MPS.
  - Scores it with the existing `bench/`, plus a pulse metric and a gate checker.

**Tech Stack:** Swift 6 + VideoToolbox (macOS 26+); Python 3.13 venv at `third_party/venv`, torch 2.14 (MPS), numpy, opencv, lpips, DISTS_pytorch, pytest; ffmpeg 9.

**Spec:** `docs/superpowers/specs/2026-10-13-detail-layer-sr-design.md`. This plan covers milestone M1 only.

## Global Constraints

**Build and environment**
- Swift: build and test only with `--scratch-path .build-dev` (`swift build -c release --scratch-path .build-dev`, `swift test --scratch-path .build-dev`).
  - Never write into `.build/` while `scripts/batch-upscale.sh` is running.
  - The dev binary is `.build-dev/release/metalfx-upscale`.
- Python: `source bench/env.sh` first. Tests live in `detailnet/tests/` and run with `python -m pytest detailnet/tests -q`.

**Protecting the user's batch**
- Never touch `output/4K/`, `output/BATCH-STATUS.md`, or the batch processes.
- GPU-heavy jobs refuse to start while `pgrep -f scripts/batch-upscale.sh` finds the batch, unless passed `--force`. The GPU-heavy jobs are: teacher generation, training, and bench runs of DetailNet/Apple.
- Every long GPU job aborts cleanly (exit 3, resumable) if swap grows by more than 8 GB from its start (`sysctl vm.swapusage`).

**Data and outputs**
- Held out from all training data: `Series 1 Carving the Landscape [DO2104H003S00]` and `Series 1 Ep 1 [DO2303H001S00]`.
- Training episodes:
  - 720p (teacher examples): `Arteries and Veins [DO2104H001S00]`, `Following the Rain [DO2104H002S00]`.
  - 1080p (ground-truth examples): `Ep 2 [DO2303H002S00]`, `Ep 3 [DO2303H003S00]`.
- Paths:
  - training data: `output/detail-data/`
  - runs: `output/detail-runs/<name>/`
  - disk budget about 150 GB

**Spec values** (use these exactly)

| What | Value |
|---|---|
| Scale | s = 3 |
| Dump crop | 320×180 source px + 32 px margin on each side (dumped area 384×244) |
| Training sub-crop | 160×90 source px, inside the central 320×180 |
| Sequence length | 12 frames (10% are 24); plus up to 8 burn-in frames from the same shot, with no loss on them |
| Teacher | FlashVSR v1.1 tiny via `bench/flashvsr_mps.py`: sparse_ratio 2.0, local_range 11, color_fix on |
| Ground-truth input | 1080p area-downscaled to 640×360, then x264 at 0.1 bits/pixel |
| Data volume | ≥ 8,000 kept teacher frames (700 sequences) and 400 ground-truth sequences |
| Network | trunk 12 residual blocks × 64 channels; hidden memory 32 channels with tanh |
| Training | AdamW, lr 2e-4, weight decay 1e-4, cosine decay, grad clip 1.0, 100k iterations, batch 4 × 12 frames |
| Loss weights (initial) | perceptual 1.0, L1 (ground truth only) 1.0, texture lock 2.0, ping-pong 0.5 |

**Gates** (from the spec), all vs Apple AI on bench clips:
- Detail ≥ 1.5×.
- Jitter ≤ 1.1× overall, and in each motion bucket.
- Pulse: max over k of jitter(t mod 4 = k) ≤ 1.1 × mean.
- DetailNet's own cut leak ≤ Apple's cut_leak.
- LPIPS on the new ×3 ground-truth clips ≤ Apple's.

**Version control:** the project is not a git repository. "Commit" steps apply only if the user approves `git init` at execution start; otherwise skip them.

## Review Focus

1. **Colour/decoder mismatch** between the Swift-decoded source and Apple's base. The base, area-downscaled ×3, must match the source dump within 3 levels per channel mean, with channel-bias spread < 1 level. Test owner: Task 2.
2. **Frame misalignment** between Swift `--start` seeks and the ffmpeg-based shot-map frame indices. Source dump frame i must equal the ffmpeg-decoded episode frame `start+i` (mean |Δ| < 2 levels). Test owner: Task 5.
3. **Short shots, cuts and dissolves inside sampled windows.** Within-shot windows never cross a cut; straddle windows contain exactly one cut; burn-in never crosses a cut. Test owner: Task 4.
4. **Frame 0 / after a cut there is no previous frame.** The flow is marked invalid, and the dataset and bench runner reset the state there; an invalid flow must never be warped through. Test owners: Tasks 3 and 8.
5. **Texture lock with an all-occluded mask** (water, dissolves) must return 0, not NaN. Test owner: Task 7.

---

### Task 1: Dump file format (Swift writer + Python reader)

**Files:**
- Create: `Sources/UpscaleCore/FrameDump.swift`
- Create: `detailnet/__init__.py`, `detailnet/dumps.py`
- Test: `Tests/UpscaleCoreTests/FrameDumpTests.swift`, `detailnet/tests/test_dumps.py`

**Interfaces:**
- **Produces the format** (both languages):
  - Directory `DIR/` holds one file per frame, `DIR/%06d.f16`: raw little-endian IEEE float16, planar CHW, no header.
  - `DIR/meta.json` = `{"kind": "source"|"base"|"flow"|"teacher"|"target", "channels", "width", "height", "layout": "CHW", "dtype": "float16", "frames", "pts": [seconds], "source_size": [w,h], "scale": int, "crop": [x,y,w,h] (source px, before scaling), "valid": [bool per frame]}`.
  - Flow dumps add `"flow_channels": ["back_x","back_y","fwd_x","fwd_y"]` and `"flow_units": "source_px"`.
  - `meta.json` is rewritten by `finish()`.
- **Produces (Swift):** `public final class FrameDumpWriter`
  - `init(directory: URL, kind: String, channels: Int, width: Int, height: Int, sourceSize: (Int, Int), scale: Int, crop: CropRect) throws`
  - `func write(_ chw: [Float16], pts: Double, valid: Bool = true) throws`
  - `func finish() throws`
  - `public struct CropRect: Equatable { x, y, width, height: Int }` with `static func parse(_ text: String) throws -> CropRect` for `"X,Y,W,H"`.
- **Produces (Python):**
  - `read_meta(d: Path) -> dict`
  - `read_frame(d: Path, i: int) -> np.ndarray` (float16, C×H×W)
  - `read_flow(d: Path, i: int, size: tuple[int,int] | None = None) -> np.ndarray` (float32, 4×H×W). Values are in pixels **of the returned grid**: bilinear resize, with x scaled by `size[0]/width` and y by `size[1]/height`.
  - `class DumpWriter(d, kind, channels, width, height, source_size, scale, crop)` with `.write(chw: np.ndarray, pts: float, valid=True)` and `.finish()`.

- [ ] **Step 1: Write failing tests.**
  - Swift `FrameDumpTests`:
    - `writesPlanarHalfFloats`: two 3×2×4 frames → each file is 48 bytes; the bytes decode back to the written values; meta `frames == 2`, `pts == [0, 0.04]`.
    - `cropParses`: `"10,20,320,180"` → `CropRect(10,20,320,180)`; `"1,2,3"` and `"-1,0,10,10"` throw.
  - Python `test_dumps.py`:
    - `test_roundtrip`: DumpWriter → read_frame gives identical float16.
    - `test_read_flow_rescales_units`: a 4×4×8 flow dump that is constant `(+4, 0, -4, 0)` → `read_flow(size=(8,4))[0]` == 4; `read_flow(size=(24,12))[0]` == 12 and `[2]` == -12.
- [ ] **Step 2: Run them.** `swift test --scratch-path .build-dev --filter FrameDumpTests` and `python -m pytest detailnet/tests/test_dumps.py -q`. Expected: FAIL, not defined. First run `uv pip install pytest` in the venv.
- [ ] **Step 3: Implement** `FrameDump.swift` and `detailnet/dumps.py` to the interfaces above. Use `JSONSerialization` with sorted keys and `numpy.fromfile` / `tofile` with `<f2`.
- [ ] **Step 4: Run both test commands.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: frame dump format for training data`.

### Task 2: Split Apple SR; dump source and base frames

**Files:**
- Modify: `Sources/UpscaleCore/AppleMLEngine.swift`; `Sources/UpscaleCore/VideoUpscaler.swift` (Options + engine construction); `Sources/metalfx-upscale/main.swift` (flags + usage)
- Create: `Sources/UpscaleCore/DumpSet.swift`
- Test: `Tests/UpscaleCoreTests/DumpOptionsTests.swift`; verification script `detailnet/tests/check_dump_colors.py`

**Interfaces:**
- **Consumes:** `FrameDumpWriter`, `CropRect` (Task 1).
- **Produces:**
  - `@available(macOS 26.0, *) final class AppleSuperResolution`
    - `init(sourceWidth: Int, sourceHeight: Int) throws`
    - `var sourceAttributes: [String: Any]`
    - `func upscale(_ source: CVPixelBuffer, at time: CMTime) throws -> CVPixelBuffer`: returns the 4× RGBA half buffer, holds the previous-frame state, and uses `.random` submission for the first frame.
  - `AppleMLEngine` uses `AppleSuperResolution`. Its behaviour and the bytes it writes are unchanged.
  - `public struct DumpOptions { var source: URL?; var base: URL?; var flow: URL?; var crop: CropRect? }`, plus `VideoUpscaler.Options.dumps: DumpOptions?`.
  - `func baseScale(sourceWidth:sourceHeight:outputWidth:outputHeight:) throws -> Int`: throws unless both ratios are the same integer.
  - CLI: `--dump-source DIR --dump-base DIR --dump-flow DIR --dump-crop X,Y,W,H`.
    - Dumps require `--engine ml` and an integer scale.
    - The default crop is the full frame; the base crop is `crop × s`.
- **Pins:**
  - The base is the 4× buffer resized to the target with `kVTDownsamplingMode_Average` into a target-size `kCVPixelFormatType_64RGBAHalf` buffer, then cropped. That makes it exactly what M2's `DetailEngine` feeds the net.
  - The source dump is the decoded buffer (`64RGBAHalf`), cropped. Throw for any other format.
  - RGB only (alpha dropped).

- [ ] **Step 1: Write failing tests** (`DumpOptionsTests`):
  - `scaleIsInteger`: 1280×720→3840×2160 gives 3; 1920×1080→3840×2160 gives 2; 640×360→1920×1080 gives 3; 640×480→2880×2160 throws.
  - `dumpsRequireMLEngine`: Options with `.metalfx` and dumps → `VideoUpscaler.validate(_:)` throws.
  - `cropMustFitSource`: crop 1000,600,320,180 on 1280×720 throws.
- [ ] **Step 2: Run** `swift test --scratch-path .build-dev --filter DumpOptionsTests`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Add `static func validate(_ options: Options, sourceWidth:, sourceHeight:, outputWidth:, outputHeight:) throws` on `VideoUpscaler`; `upscale` calls it.
  - Add the `AppleSuperResolution` split and `DumpSet`: it owns the writers, records at each processed frame, and calls `finish()` when the job ends, including on error or cancel.
  - Add the CLI flags.
- [ ] **Step 4: Run all Swift tests.** `swift test --scratch-path .build-dev`. Expected: PASS, and `TargetSizeTests` is unchanged.
- [ ] **Step 5: Check that the `ml` output is unchanged.**
  - Run `.build-dev/release/metalfx-upscale` and the batch's `.build/release/metalfx-upscale` on the same 2 s excerpt (`--start 600 --duration 2 --engine ml --bitrate 50`). The second command only *runs* the existing binary; it does not rebuild it.
  - Compare decoded frames with ffmpeg. Expected: mean |Δ| < 0.5 levels; the encoder isn't bit-exact across runs.
- [ ] **Step 6: Colour check (Review Focus 1).**
  - Run with `--dump-source output/detail-scratch/s --dump-base output/detail-scratch/b --dump-crop 480,270,320,180` on 2 s of *Arteries and Veins*.
  - Write `check_dump_colors.py SRC BASE`: it area-downscales the base ×3 and compares it with the source per channel.
  - Expected: per-channel mean |Δ| ≤ 3 levels, and the (max − min) of the per-channel mean signed Δ is < 1 level.
- [ ] **Step 7: Commit** `feat: split Apple SR and dump source/base frames`.

### Task 3: VT optical flow dumps

**Files:**
- Create: `Sources/UpscaleCore/OpticalFlow.swift`
- Modify: `Sources/UpscaleCore/DumpSet.swift`
- Test: `Tests/UpscaleCoreTests/OpticalFlowTests.swift`

**Interfaces:**
- **Consumes:** `DumpSet`, `FrameDumpWriter`.
- **Produces:** `@available(macOS 26.0, *) final class OpticalFlowEstimator`
  - `init(width: Int, height: Int) throws`: uses `VTOpticalFlowConfiguration(frameWidth:frameHeight:qualityPrioritization: .quality, revision: .defaultRevision)`.
  - `func flow(previous: CVPixelBuffer, current: CVPixelBuffer, sequential: Bool) throws -> (backward: [Float16], forward: [Float16], width: Int, height: Int)`
    - Planes are 2×H×W in **source pixels**: backward = displacement from t to t−1, forward = from t−1 to t.
    - Inputs are converted with a `VTPixelTransferSession` to `configuration.frameSupportedPixelFormats.first` when needed.
- **Flow dump:** 4 channels `[back_x, back_y, fwd_x, fwd_y]`, at VT's native flow resolution, cropped to `crop` scaled to that resolution. Frame 0 of a run is all zeros with `valid[0] = false`.

- [ ] **Step 1: Write the failing test.** `OpticalFlowTests.detectsShift`:
  - Skip when `!VTOpticalFlowConfiguration.isSupported`.
  - Build two 256×144 frames from blurred random noise, the second shifted +4 px in x.
  - Expect median backward x ∈ [−4.5, −3.5], median forward x ∈ [3.5, 4.5], and |median y| < 0.5 for both.
- [ ] **Step 2: Run** `swift test --scratch-path .build-dev --filter OpticalFlowTests`. Expected: FAIL.
- [ ] **Step 3: Implement.**
  - Map VT's `backwardFlow` / `forwardFlow` buffers to the pinned meaning. If the test shows they're the other way round, swap them here, not later.
  - Convert units to source pixels using buffer size vs source size.
  - Wire `--dump-flow` into `DumpSet`.
- [ ] **Step 4: Run all Swift tests.** Expected: PASS.
- [ ] **Step 5: Smoke run.** 2 s with all three dumps and a crop. Expected: three dump dirs with equal `frames`; flow `valid == [false, true, …]`.
- [ ] **Step 6: Commit** `feat: dump VideoToolbox optical flow`.

### Task 4: Sequence sampler

**Files:**
- Create: `detailnet/sampling.py`
- Test: `detailnet/tests/test_sampling.py`

**Interfaces:**
- **Consumes:** `output/sr-bench/shots/<episode>.frames.csv`. Columns: frame, cut, content, pan, local, texture, luma.
- **Produces:**
  - `@dataclass SeqSpec(episode: str, kind: str, start: int, frames: int, burn_in: int, bucket: str, cuts: list[int], crop: tuple[int,int,int,int])`
    - `kind` ∈ {teacher, gt}.
    - `bucket` ∈ {static, slow, fast, local, straddle, dissolve}.
    - `cuts` are relative to `start − burn_in`.
    - `crop` is in the sequence's source pixels: 720p frames for teacher, 640×360 for gt.
  - `sample(rows_by_episode: dict[str, list[dict]], kind: str, n: int, frame_size: tuple[int,int], seed: int) -> list[SeqSpec]`
- **Rules:**
  - **Length:** 10% of sequences are 24 frames, the rest 12.
  - **Straddle (15%):** exactly one cut, at relative position 3…frames−3.
  - **Dissolve (5%):** at least 3 consecutive non-cut frames with content > 15.
  - **The rest:** within one shot, balanced across 4 buckets by window medians:
    - static: pan < 0.3 and local < 0.5
    - fast: pan ≥ 2
    - local: local ≥ 2 and pan < 2
    - slow: everything else
  - **Burn-in:** up to 8 preceding frames from the same shot.
  - **Rejections:** mean luma < 20.
  - **Crop:** uniform, with crop + 32 px margin fully inside the frame.

- [ ] **Step 1: Write failing tests** on a synthetic 3-shot table:
  - `test_buckets_balanced`: within ±1 per bucket.
  - `test_within_shot_never_crosses_cut`.
  - `test_straddle_has_one_cut`.
  - `test_burn_in_stays_in_shot`.
  - `test_short_shots_skipped`: shots < 12 frames yield no within-shot windows.
  - `test_crop_with_margin_inside` for (1280,720) and (640,360).
  - `test_deterministic_seed`.
- [ ] **Step 2: Run** `python -m pytest detailnet/tests/test_sampling.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `sample`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: training sequence sampler`.

### Task 5: Data builder (dumps per sequence)

**Files:**
- Create: `detailnet/build_data.py`
- Test: `detailnet/tests/test_build_data.py`

**Interfaces:**
- **Consumes:** `sample`/`SeqSpec` (Task 4); the dump CLI flags (Tasks 2–3); `bench/make_clips.py`'s x264 recipe (copy its constants, don't import its CLI).
- **Produces:** `python -m detailnet.build_data --out output/detail-data --teacher-seqs 700 --gt-seqs 400 --seed 0 [--limit N]`
  - Writes `sequences.json`, and per sequence `output/detail-data/<kind>/<id:05d>/{spec.json, source/, base/, flow/}`, plus `target/` for gt.
  - Resumable: it skips a sequence whose `flow/meta.json` exists.
- **Teacher sequences:** Swift on the episode `.mp4` with `--start (start−burn_in)/fps`, `--duration (burn_in+frames)/fps`, `--dump-crop` = crop expanded by the 32 px margin.
- **Ground-truth sequences:**
  1. ffmpeg extracts the 1080p frames (`-ss`, `-frames:v`) to ProRes 4444.
  2. Swift `--engine ml --dump-source target/` with crop = (crop × 3) on the 1080p clip. The 360p crop's ×3 region is 960×540 at 1080p.
  3. ffmpeg `scale=640:360:flags=area` then x264 `-b:v` = 640·360·fps·0.1 bits/s, maxrate 2×.
  4. Swift `--engine ml` on that 360p clip, with source/base/flow dumps and the margin-expanded crop.
- **Pure helpers:**
  - `def dump_crop(crop, margin=32) -> tuple`
  - `def gt_bitrate(fps: float) -> int`
  - `def swift_cmd(binary, input, out_dir, start, duration, crop, dumps: set[str]) -> list[str]`

- [ ] **Step 1: Write failing tests.**
  - `test_dump_crop_adds_margin`.
  - `test_gt_bitrate`: `gt_bitrate(25) == 576000`.
  - `test_swift_cmd_flags`: exact argv, including `--engine ml` and `--dump-crop 288,148,384,244` for crop (320,180,320,180).
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Alignment check (Review Focus 2).**
  - `--limit 3` smoke run; check the counts and shapes in each meta.
  - Then for one teacher sequence, compare `source/` frame i with the ffmpeg-decoded episode frame `start−burn_in+i` (bt709, same crop).
  - Expected: mean |Δ| < 2 levels for all i, and a larger Δ for a ±1 frame offset.
- [ ] **Step 6: Full run** (CPU/Media Engine-heavy). It may run while the batch runs, because it only uses the ml engine on short excerpts. It stops early if the swap guard trips. Expected: 700 + 400 sequence dirs.
- [ ] **Step 7: Commit** `feat: build DetailNet training data`.

### Task 6: Teacher targets

**Files:**
- Create: `detailnet/teacher.py`
- Test: `detailnet/tests/test_teacher.py`

**Interfaces:**
- **Consumes:**
  - `read_frame` / `DumpWriter` (Task 1).
  - `FlashVSR().run_shot(frames: list[np.ndarray uint8 HxWx3], emit)` from `bench/models.py`, which emits ×4 tensors (3,H,W) in [0,1]. Import it with `sys.path.insert(0, <repo>/bench)` then `from models import FlashVSR`, the same way the bench scripts import each other.
  - Teacher sequence dirs (Task 5).
- **Produces:** `python -m detailnet.teacher --data output/detail-data [--limit N] [--force]` writes `teacher/` (kind "teacher", 960×540, all burn-in+frames frames).
  - Order: round-robin over buckets.
  - Resumable; refuses to run while the batch runs unless `--force`; swap guard.
  - `def to_teacher_grid(hr4: np.ndarray, margin: int = 32) -> np.ndarray`: (3, 4H, 4W) → area resize ×0.75 → crop `3·margin` on every side → (3, 540, 960) for a 384×244 input.
  - `def make_teacher(seq_dir: Path, upscale: Callable[[list[np.ndarray]], list[np.ndarray]]) -> None`: the upscaler is injected so tests can use a fake.

- [ ] **Step 1: Write failing tests.**
  - `test_grid_shape`: input (3,976,1536) → (3,540,960).
  - `test_grid_alignment`: an LR marker at (x=32+10, y=32+20), upscaled with a fake nearest-×4 upscaler, lands at teacher pixels x∈[30,33), y∈[60,63).
  - `test_make_teacher_with_fake_upscaler`: writes a dump whose frames == nearest×3 of the central source crop (≤ 1/255).
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Real run on 2 sequences** after the batch finishes. Record the s/frame. Look at one teacher frame next to the base: the texture should match FlashVSR's look in `output/sr-bench/compare/`.
- [ ] **Step 6: Full run.** Overnight; resumable; ≥ 700 sequences. Training (Task 9) may start once ≥ 400 are done.
- [ ] **Step 7: Commit** `feat: FlashVSR teacher targets`.

### Task 7: DetailNet model and losses

**Files:**
- Create: `detailnet/model.py`, `detailnet/losses.py`
- Test: `detailnet/tests/test_model.py`, `detailnet/tests/test_losses.py`

**Interfaces (model.py):**
- `space_to_depth(x, s)`, `depth_to_space(x, s)`.
- `upsample_flow(flow_px, s) -> Tensor`: bilinear ×s, values × s.
- `warp(x, flow_px) -> Tensor`: `grid_sample`, bilinear, border padding, align_corners=True; flow (N,2,H,W) is the backward displacement in x's pixels.
- `texture_only(r, s) -> Tensor`: r − bicubic_up(area_down(r, s), s).
- `tracking_maps(x_t, x_prev, flow4) -> Tensor` (N,2,h,w):
  - luma |x_t − warp(x_prev, back)|;
  - fb inconsistency |back + warp(fwd, back)|.
- `occlusion_mask(maps) -> Tensor` (bool): luma error < 0.05 and fb < 0.5 + 0.01·(|b|² + |f|²).
- `class DetailNet(nn.Module)`:
  - `__init__(scale=3, channels=64, blocks=12, hidden=32)`
  - `forward(x_t, x_prev, base, flow4, prev_detail, hidden) -> tuple[Tensor, Tensor]`
    - Input stack at source resolution: x_t (3), s2d(base) (3s²), s2d(warp(prev_detail, up(back))) (3s²), warp(hidden, back) (32), tracking_maps (2) → conv3×3 → ResBlocks → heads.
    - Detail = `texture_only(d2s(head_tex), s)`; hidden = `tanh(head_hid)`.
  - `zero_state(n, h, w, device) -> tuple`
- `compose(base, detail, strength) -> Tensor`: clamp to [0,1].

**Interfaces (losses.py):**
- `class Perceptual`: `__call__(pred, target)` = DISTS + LPIPS-VGG, inputs in [0,1], (N,3,H,W).
- `texture_lock(d_t, d_prev, back_flow_hr, mask_hr) -> Tensor`: mean over mask of |d_t − warp(d_prev)|; returns 0 when mask.sum() == 0.
- `ping_pong(fwd: Tensor, bwd: Tensor) -> Tensor`: mean |fwd − bwd|.

- [ ] **Step 1: Write failing tests.**
  - `test_shapes_s3_s2`: forward returns detail at s× and hidden at (32,h,w).
  - `test_param_count`: between 0.8 M and 2.5 M at defaults.
  - `test_texture_only_kills_smooth`: a smooth ramp retains < 5% energy.
  - `test_texture_only_keeps_hf`: an HR checkerboard retains > 80%.
  - `test_texture_only_zero_mean`: |mean| < 1e-3.
  - `test_compose_strength_zero_is_base`: exact.
  - `test_warp_inverts_shift`: an image shifted +2 px, warped with back flow −2, matches the interior to < 1e-5.
  - `test_hidden_bounded`: 50 recurrent steps on random input stay within |h| ≤ 1.
  - `test_texture_lock_zero_when_consistent`.
  - `test_texture_lock_empty_mask_is_zero_not_nan` (Review Focus 5).
  - `test_perceptual_identity_near_zero` (< 1e-3) and positive for noise.
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: DetailNet model and losses`.

### Task 8: Training dataset

**Files:**
- Create: `detailnet/data.py`
- Test: `detailnet/tests/test_data.py`

**Interfaces:**
- **Consumes:** the dumps (Task 1) and sequence dirs (Tasks 5–6).
- **Produces:** `class SequenceDataset(torch.utils.data.Dataset)`
  - `__init__(root, kind, crop=(160, 90), ping_pong_prob=0.2, reset_prob=0.15, seed=0)`; `kind` ∈ {teacher, gt}.
  - Item dict:
    - `x` (T,3,h,w), `base` (T,3,3h,3w), `flow` (T,4,h,w) in source px of the crop grid, `target` (T,3,3h,3w)
    - `reset` (T,) bool: True at frame 0, at spec cuts, wherever `valid` is false, and at a random frame with probability reset_prob
    - `loss_mask` (T,) bool: False for burn-in
  - The sub-crop lies inside the central 320×180: dumped-coordinate x0 ∈ [32, 192], y0 ∈ [32, 122]. The target offset is (x0 − 32)·3.
  - **Ping-pong:** append the reversed frames (excluding the last), swapping back↔fwd flow channels.

- [ ] **Step 1: Write failing tests** on synthetic dumps where base = nearest×3 of the source and target = base:
  - `test_crop_alignment`: area_down(base) == x exactly.
  - `test_reset_at_invalid_flow_and_cuts` (Review Focus 4).
  - `test_burn_in_unmasked`: loss_mask False for burn-in frames.
  - `test_ping_pong_swaps_flow`: the reversed half's flow channels 0–1 equal the original's 2–3.
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: DetailNet training dataset`.

### Task 9: Training loop

**Files:**
- Create: `detailnet/train.py`
- Test: `detailnet/tests/test_train.py`

**Interfaces:**
- **Consumes:** `DetailNet`, `compose`, `occlusion_mask`, `tracking_maps`, `upsample_flow` (Task 7); the losses (Task 7); `SequenceDataset` (Task 8).
- **Produces:**
  - `@dataclass TrainConfig`: data, out, iters=100_000, batch=4, lr=2e-4, weight_decay=1e-4, w_perc=1.0, w_l1=1.0, w_lock=2.0, w_pp=0.5, ckpt_every=5000, seed=0, channels=64, blocks=12.
  - `train(cfg) -> Path` returns the last checkpoint.
  - `load_checkpoint(path) -> tuple[DetailNet, dict]`
  - CLI: `python -m detailnet.train --data output/detail-data --out output/detail-runs/r1 [--iters N] [--resume]`
- **Pins:**
  - Batches alternate teacher and gt.
  - Truncated BPTT over the full sequence; the state is reset to zeros where `reset` is true.
  - Losses only at `loss_mask` frames:
    - perceptual vs the target (both kinds);
    - L1 vs the target (gt only);
    - texture lock using `occlusion_mask` upsampled nearest ×3;
    - ping-pong between each frame's output and its reversed twin.
  - The output is `compose(base, detail, 1.0)`.
  - Checkpoints `ckpt_{iter:06d}.pt` and `latest.pt` contain `{"model", "opt", "iter", "cfg"}`.
  - Log lines go to `train.jsonl`.
  - Batch guard, swap guard, and `--force`, as in Global Constraints.

- [ ] **Step 1: Write failing tests.**
  - `test_overfits_one_sample`: channels 16, blocks 2, 40 steps on one synthetic sample; total loss falls ≥ 30%.
  - `test_checkpoint_roundtrip`: identical outputs after a reload.
  - `test_resume_continues_iter`.
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat: DetailNet training loop`.

### Task 10: Bench integration (×3 ground-truth clips, dumps, DetailNet runner, pulse, gates)

**Files:**
- Modify: `bench/make_clips.py` (add kind `gt3`); `bench/metrics.py` (add `jitter_phase`, `pulse`); `bench/summarize.py` (carry `pulse`; include `g3` in the ground-truth means as `gt3_lpips`)
- Create: `bench/clips-g3.json` (5 `g3-*` clips, same source, starts and frames as the `g-*` clips; kept apart so `make_clips` doesn't re-extract the existing clips)
- Create: `detailnet/bench_dumps.py`, `detailnet/run_bench.py`, `bench/gates.py`
- Test: `detailnet/tests/test_run_bench.py`, `detailnet/tests/test_gates.py`

**Interfaces:**
- **`gt3` clips:** lr = 640×360 (area + x264 at 0.1 bits/pixel), target 1920×1080, gt.mkv as for `gt`.
- **`bench_dumps.py CLIP_DIR...`** writes `CLIP_DIR/dumps/{source,base,flow}/` per shot:
  - Convert to ProRes 4444 per shot, as `bench/run_model.run_apple` does.
  - Run the dev Swift binary with all three dumps.
  - Crop: for 4K clips, `run_model.crop_plan(clip)`'s LR box (640×360 + 32 margin); for `gt3`, the full frame.
  - Real clips: the first 100 frames only.
  - Shot starts are recorded in meta `valid`.
- **`run_bench.py --ckpt P --name N [--strength 1.0] [--ignore-cuts] CLIP_DIR...`** writes `out/N.mkv` and `out/N.json` in `run_model`'s stats format:
  - `model, label, frames, seconds, s_per_frame, center_crop, out_size, frame_pixels, clip, ignore_cuts`
  - 4K clips: `center_crop: true`, `out_size: [1920,1080]`, margin 96 cut off.
  - The state resets at clip cuts unless `--ignore-cuts`.
  - The cut-leak gate uses DetailNet's own `N-nocut`, over the same base.
- **`metrics.py`:** `jitter_phase` = mean jitter of pair t grouped by t mod 4; `pulse = max/mean`.
- **`gates.py MODEL [--summary output/sr-bench/summary.json]`:** prints PASS/FAIL per gate (Global Constraints) against `apple`; exit 0 only if all pass.
  - Scope: real clips are the `c-*` clips; `e-*` (1080p) are excluded until M3; ground-truth clips are `g3-*`.

- [ ] **Step 1: Write failing tests.**
  - `test_strength_zero_equals_base`: run_bench on a synthetic clip dir with fake dumps and an untrained net at strength 0 → output frames == the base crop (≤ 1/255).
  - `test_resets_at_cuts`: with cuts=[3], the frame-3 output equals the output of a fresh run started at frame 3.
  - `test_pulse_detects_period4`: a synthetic jitter series with every 4th pair ×2 gives pulse > 1.5; a flat series gives pulse < 1.05.
  - `test_gates_logic`: a synthetic summary at exactly the thresholds → PASS; detail 1.49 → FAIL.
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Bench prep** (after the batch finishes):
  - `python bench/make_clips.py bench/clips-g3.json output/sr-bench/clips`;
  - `bench/run_all.sh apple` (Apple on `g3-*`; finished clips are skipped);
  - `python -m detailnet.bench_dumps output/sr-bench/clips/c-* output/sr-bench/clips/g3-*`;
  - re-score Apple with the pulse metric: `rm output/sr-bench/clips/*/metrics/apple.json && bench/score_all.sh 2`.
  - Expected: Apple's `pulse` is about 1.0–1.1 on every `c-*` clip.
- [ ] **Step 6: Commit** `feat: bench integration for DetailNet`.

### Task 11: M1 run, tuning, go/no-go, deliverables

**Files:**
- Create: `output/detail-runs/<name>/gates.txt` (generated)

- [ ] **Step 1: Train** `r1`: `python -m detailnet.train --data output/detail-data --out output/detail-runs/r1` (about 6–10 h). At each 20k-iteration checkpoint, run `run_bench` + `score_all` + `gates.py` on `c-slow-gorge` only, as a quick read.
- [ ] **Step 2: Full bench** for the final checkpoint:
  - `run_bench` on all `c-*` and `g3-*` clips, plus `--ignore-cuts` on `c-cut-nest`;
  - `bench/score_all.sh 2`, `python bench/summarize.py output/sr-bench/summary.json`;
  - `python bench/gates.py detail-r1 | tee output/detail-runs/r1/gates.txt`.
- [ ] **Step 3: Tuning rule (at most 2 rounds).** Resume from latest for 30k iterations as `r2`, `r3`:
  - jitter or pulse fails → double `w_lock`;
  - detail fails while jitter passes → halve `w_lock`;
  - both fail → stop.
- [ ] **Step 4: Comparison videos.** `python bench/compare.py CLIP output/sr-bench/compare/detail detail-rN apple flashvsr…` for `c-slow-gorge`, `c-static-foliage`, `c-fast-waterfall`, `c-cut-nest`, `g3-waterfall` (with `gt`). Send them to the user with light copies under 30 MB.
- [ ] **Step 5: Go/no-go report to the user.**
  - Contents: the gate table, the videos, and the recommendation: proceed to M2 (app), or stop / reconsider option B.
  - Update the memory file `own-sr-model-investigation.md` with the outcome.

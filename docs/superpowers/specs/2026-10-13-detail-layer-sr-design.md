# Detail layer on Apple AI — design

Date: 2026-10-13 · Status: approved in conversation, awaiting spec review

## Goal

Upscale this project's footage (720p/1080p broadcast H.264 documentaries) to 4K with **visibly more real-looking detail than Apple AI (`VTSuperResolutionScaler`) and no added motion artifacts**, running on this M5 Max (64 GB) inside the Upscale app, at **≤ 10 h for a 58-minute 720p episode** (≤ 0.4 s/frame).

Decisions taken with the user:
- All training and data generation on this Mac (no cloud, no other GPU).
- Speed budget: overnight, ≤ 0.4 s/frame for 720p → 4K.
- Runs in the Upscale app via Core ML, as a new engine.
- Approach A: a learned detail layer on top of Apple AI. Option B (our own full upscaler) is a later upgrade, so the code must let a learned base replace Apple's.

Non-goals: real time; HDR; sources above 1080p; still images; iOS.

## Evidence this design rests on (bench, 2026-10-13)

From `output/sr-bench/` (12 clips from *Carving the Landscape* and *Ep 1*, see `bench/README.md`):

- **Apple AI is the best practical upscaler, but soft.** Its jitter is about the same as the real footage's. Every model fast enough to ship either smooths or adds texture that shimmers.
- **FlashVSR v1.1 is the only model that looks like native 4K.**
  - About 1.8× Apple's detail, and the closest perceptually to the real 1080p (LPIPS 0.317 vs Apple's 0.405).
  - Its instability is a pulse every 4 frames, where its decoder starts a new group of frames. Within a group its jitter is about 1.07× Apple's.
  - It runs at about 14–40 s/frame on this Mac, so it can be a teacher but can't ship.
- **Both source types are soft for their resolution.** Their finest octave holds about 15% of a natural image's energy, so 720p → 4K detail has to be *synthesized* and kept locked to the scene.
- **Episodes cut and move a lot.**
  - About 800 cuts per episode; median shot 3.1–3.7 s.
  - About 20% of frames are static, 20% fast pan, and 20% strong local motion.
  - Recurrent models smear across cuts unless they are reset (RealBasicVSR: cut leak 1.8 vs Apple's 0.7).

## Architecture

### Per-frame data flow (720p → 4K; s = 3)

```
source x_t (1280x720) ──┬─► Apple SR (VT, 4x) ─► resize ─► base B_t (3840x2160, RGB half)
                        ├─► VT optical flow (x_t, x_{t-1}) ─► flow F_t→t-1 (+ backward if provided)
                        └──────────────────────────────┐
cut list (pre-pass) ─► reset? ─► zero D̃ and h        │
                                                       ▼
   DetailNet( x_t, x_{t-1}, B_t, F, D_{t-1}, h_{t-1} )  →  D_t (4K texture), h_t
                                                       │
   output O_t = B_t + strength · D_t  ─► 10-bit 4:2:0 ─► HEVC (existing encoder)
```

### DetailNet

- **Inputs.**
  - Source frames x_t and x_{t-1}.
  - Base B_t, rearranged space-to-depth by s to source resolution (27 channels at 720p).
  - Flow F (2 channels at source resolution).
  - Previous texture D_{t-1} (4K).
  - Recurrent memory h_{t-1}: 32 channels at source resolution.
- **Inside the model.**
  - Upsample the flow and warp D_{t-1} (at 4K) and h_{t-1} (at source resolution) to frame t with `grid_sample`.
  - Rearrange the warped D̃ space-to-depth like the base.
  - Compute a *tracking-failed* map: |x_t − warp(x_{t-1})| in luma, plus a forward-backward inconsistency map if VT returns both flow directions.
  - Trunk: about 12 residual blocks × 64 channels, all at source resolution, ~1–2 M parameters. The trunk's final size is set by the speed gate.
  - Two heads:
    - a 27-channel texture head, rearranged depth-to-space to the 4K raw texture R;
    - a 32-channel memory head, giving h_t.
- **Texture-only constraint.** D = R − U(A(R)), where A is area downsampling by s and U is bicubic upsampling back. D therefore holds no content representable at source resolution, so it cannot change brightness, colour or shapes.
- **Cut reset.** At frames in the cut list, D̃ = 0 and h = 0. Training includes random resets so the model warms up within 1–2 frames.
- **Strength.** O = B + strength · D, with strength in 0…1. Strength 0 reproduces Apple AI exactly.
- **1080p variant (M3).** Same trunk at 1080p with s = 2 (12-channel rearrangement); about 2.3× the compute.

## Training (Python, PyTorch on MPS)

### Data

- **Training episodes:** *Arteries and Veins* and *Following the Rain* (720p); *Ep 2* and *Ep 3* (1080p).
- **Held out for evaluation only:** *Carving the Landscape* and *Ep 1*.
- **Samples:** stretches of 12 frames drawn from the shot maps.
  - Balanced over the static, slow, fast-pan and local-motion buckets.
  - Some sequences are reset at random frames; some contain dissolves.
  - 10% are 24 frames long.
- **Teacher examples (720p).**
  - FlashVSR v1.1 tiny (`bench/flashvsr_mps.py`, sparse_ratio 2.0, local_range 11, color_fix on) upscales 320×180 source crops, each with a 32 px margin.
  - Its output is cut to the 960×540 region at 4K scale.
  - The Apple base and VT flow come from the Swift tool's dump options, over the same frames.
  - Target: ≥ 8,000 frames (~670 sequences).
- **Ground-truth examples (×3).**
  - 1080p frames from *Ep 2* and *Ep 3* are area-downscaled to 640×360 and x264-encoded at 0.1 bits/pixel. That 360p clip is the input; the real 1080p is the target.
  - The Apple base and flow come from the 360p clip.
- **Training crops:** 160×90 at source resolution (480×270 at target resolution), batch 4 × 12 frames.

### Losses

- **Teacher examples:** perceptual distance (DISTS, plus LPIPS-VGG) between O_t and the FlashVSR frame. No pixel L1, because the teacher's texture is one sample of many plausible ones.
- **Ground-truth examples:** L1 + LPIPS against the real frame.
- **Texture lock (all examples).**
  - mean(mask · |D_t − warp(D_{t-1})|), with the mask from the tracking-failed / forward-backward maps.
  - This is the bench's jitter numerator applied to the texture, and it is also what removes FlashVSR's 4-frame pulse.
- **Ping-pong:** on forward-then-reversed sequences, |O_t(forward) − O_t(backward)| prevents drift (TecoGAN).
- **No adversarial (GAN) loss initially.** It is added only if the M1 output looks too soft.
- **Optimiser:** AdamW, about 100k iterations. Every checkpoint is scored on the bench.

## App integration (Swift)

- **`Engine.detail`.** CLI `--engine detail --detail-strength 0..1` (default 1.0). In the app, a toggle plus a strength slider.
- **Split of `AppleMLEngine`.** It becomes `AppleSuperResolution`, which holds the VT processor and previous-frame state and outputs the 4× half-float RGBA, plus the existing resize/convert step. The `ml` engine behaves exactly as today.
- **`DetailEngine: VideoFrameEngine`** composes, per frame:
  - Apple SR, then a resize to a 4K half-float RGB base;
  - `VTOpticalFlow` at source resolution;
  - the `DetailNet` Core ML model (mlprogram, fp16, compute units chosen by measurement), with D and h kept between frames;
  - a Metal kernel computing B + strength · D;
  - `VTPixelTransferSession` to 10-bit 4:2:0.
- **`CutDetector` pre-pass.**
  - Decodes at 320×180 and runs the adaptive content detector, the same rules as PySceneDetect's AdaptiveDetector used for the shot maps.
  - The resulting cut list is handed to `DetailEngine`, adding about 1–2 min per episode.
- **Failure handling.** If the model can't load or fails mid-video, the job stops with an error. It never silently falls back to plain Apple AI.
- **Training-data tools** (CLI options):
  - `--dump-base DIR`: 16-bit Apple base frames;
  - `--dump-flow DIR`: raw VT flow, float16, source resolution.
- **Build safety.** Development builds use `swift build --scratch-path .build-dev`. `.build/release/metalfx-upscale` is not rebuilt while the user's batch is running.

## Evaluation gates

All gates are measured on the 12-clip bench, which contains held-out footage only.

The bench's existing ground-truth clips test ×4 (270p → 1080p). M1 therefore adds ×3 versions of the same five *Ep 1* stretches: a 360p input made the same way as the training data, scored against the real 1080p.

| Gate | Target |
|---|---|
| Detail (4K clips) | ≥ 1.5× Apple AI |
| Jitter (4K clips) | ≤ 1.1× Apple AI overall, and in each motion bucket (static, slow, fast) |
| 4-frame pulse | For every phase k of t mod 4, jitter(k) ≤ 1.1 × mean jitter |
| Cut leak | ≤ Apple AI |
| Ground-truth clips (×3 versions) | LPIPS ≤ Apple AI's |
| Speed, app | ≤ 0.4 s/frame for 720p → 4K with the GPU otherwise idle |
| User's eye | Side-by-side videos of the bench clips and one full held-out episode |

## Milestones

1. **M1, Python prototype.** Swift dump tools, data builder, teacher generation, `DetailNet` training with a bench `detail` runner, then comparison videos.
   - **Go/no-go:** detail ≥ 1.5× Apple at jitter ≤ 1.1×.
   - **If no:** stop or reconsider option B before any app work.
2. **M2, app.** Core ML export, the `AppleSuperResolution` split, `DetailEngine`, `CutDetector`, CLI flags and UI, then the speed gate and one full held-out episode in 4K.
3. **M3, 1080p.** ×2 variant, same gates.

Each milestone gets its own implementation plan, written once the previous milestone's gate has passed.

## Risks

| Risk | Mitigation |
|---|---|
| Not enough teacher data on a Mac-only budget | Small crops; ground-truth examples carry structure; the net only learns texture |
| FlashVSR's 4-frame pulse leaks into the net | Texture-lock loss; the pulse gate |
| Flow fails on water, occlusion or dissolves | Tracking-failed input; dissolves in the training data |
| Synthesized texture looks fake | Strength knob; adversarial loss if too soft; option B if Apple's base is the ceiling |
| `grid_sample` is slow or unsupported in Core ML | Do the warp in a Metal kernel instead |
| GPU contention with the user's batch | Teacher generation and timing start only once the batch is done |

## Constraints

- Never rebuild `.build/release/metalfx-upscale` or touch `output/4K/` or `output/BATCH-STATUS.md` while the batch runs.
- Keep swap growth small; the batch shares the GPU and RAM.
- Third-party code and weights stay in `third_party/`; bench and training outputs go in `output/`.

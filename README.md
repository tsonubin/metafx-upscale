# metalfx-upscale

Upscale videos (and images) to 4K on a Mac with Apple's own frameworks, plus a benchmark and research code for doing better than Apple.

It began as a MetalFX proof of concept. It grew into:
- a CLI;
- a minimal SwiftUI app;
- a resumable batch runner;
- a super-resolution benchmark;
- an experimental learned "detail layer" (DetailNet).

Everything here was built and measured on a MacBook Pro with an M5 Max (64 GB), macOS 27, and Xcode 27.

## Contents

- [Findings](#findings) (including [does the 4K look better than the original?](#does-the-4k-output-look-better-than-the-original))
- [Quick start](#quick-start)
- [The Upscale app](#the-upscale-app)
- [The CLI](#the-cli)
- [Batch upscaling](#batch-upscaling)
- [How it works](#how-it-works)
- [Quality settings](#quality-settings)
- [Models compared](#models-compared)
- [DetailNet (research)](#detailnet-research)
- [Benchmark](#benchmark)
- [Limits and known issues](#limits-and-known-issues)
- [Project layout](#project-layout)
- [Building and testing](#building-and-testing)
- [License](#license)

## Findings

**Apple AI is the best upscaler for this footage that can actually finish an episode on this Mac.** Apple AI here means VideoToolbox's `VTSuperResolutionScaler`, Apple's on-device ML video super-resolution.
- **Detail:** it rebuilds plausible texture in rock and foliage.
- **Stability:** it stays about as steady as the real footage.
- **Speed:** about 0.35 s per frame for 720p → 4K, roughly 8.5 hours per hour of video.

The six episodes in this project were upscaled with it. See [Batch upscaling](#batch-upscaling).

| Approach | Verdict |
|---|---|
| **MetalFX spatial** (`MTLFXSpatialScaler`) | Fast (about 125 fps) but only sharpens edges. Played full-screen it's barely distinguishable from the original, because the player already upscales. MetalFX's temporal scalers need depth and motion vectors from a game engine, so they can't be used on video files. |
| **Apple AI** (`VTSuperResolutionScaler`) | **Used for all final outputs.** Real texture with no visible artifacts. Soft compared with true 4K. |
| **SeedVR2 7B** (one-step diffusion; PyTorch MPS and MLX ports) | By far the most invented detail, but 35–70 s per 4K frame, which comes to weeks per episode. Run frame by frame (the only MLX mode that fits in 64 GB), its invented texture shimmers: +23% frame-to-frame change on rock, against +3% for Apple. Rejected for AI-looking artifacts on motion. All of it has been removed from this machine. |
| **FlashVSR v1.1, Real-ESRGAN, RealBasicVSR, RealViformer** | Benchmarked (see [Models compared](#models-compared)). Each one is either too slow, shimmers, or smooths. FlashVSR looks the most like native 4K, at 14–40 s per frame, so it serves as a training teacher. |
| **DetailNet r2** (ours, a detail layer on top of Apple AI) | **Not a visible improvement over Apple AI.** On the metrics it is just as steady, cleaner at cuts, and closer to ground truth (LPIPS 0.252 vs 0.317). But its extra detail (1.13×) is too small to see, and it missed its own 1.5× goal. |

Side-by-side videos of DetailNet r2 against Apple AI are in `output/sr-bench/compare/detailnet-r2/`. There is one video per benchmark clip, plus a combined 4K video.

### Does the 4K output look better than the original?

**Only slightly.** Measured on the finished episodes (Sobel edge energy on 1:1 crops of the same frame), Apple AI's 4K has only 7–16% more fine detail than the original file enlarged to 4K with a plain resize, which is roughly what a TV or player does:

| Source | Frame | Original resized to 4K | Apple AI 4K | Gain |
|---|---|---|---|---|
| 1080p (Ep 1) | 10:05, trees | 54.6 | 58.3 | +7% |
| 1080p (Ep 1) | 25:00, centre crop | 21.9 | 25.4 | +16% |
| 720p (Carving the Landscape) | 2:00, cliffs | 42.9 | 46.5 | +8% |

- **What you can see:** at 1:1 pixels, leaves, branches and rock fractures are a little crisper. No upscaler can add detail the camera never recorded; Apple AI cleans up edges and rebuilds a little texture.
- **Why it often looks identical:** the gain is only visible on a 4K screen at full size, or zoomed in. Watched in a window, or on a MacBook's built-in display (3456×2234, below 4K), the 4K file is scaled down and looks like the original.
- **Cost:** for the 1080p episodes it's 17–19 GB per episode against 1.3 GB for the original, for a barely visible gain. 720p sources gain somewhat more.
- **Clearly better needs generative models:** only models that invent detail look clearly better (SeedVR2, FlashVSR). On this Mac they're far too slow and add AI-looking artifacts on motion (see [Models compared](#models-compared)).

## Quick start

```bash
swift build -c release                       # CLI → .build/release/metalfx-upscale
scripts/build-app.sh && open build/Upscale.app

# One video to 4K with Apple AI, at the settings used for the final episodes
.build/release/metalfx-upscale input.mp4 "input - 4K.mp4" --engine ml --bitrate 50
```

On first use of `--engine ml`, macOS may need to download Apple's model. Add `--download-model`, or use the app's **Download AI model** button. It's a 66 MB download, fetched by macOS from Apple.

## The Upscale app

A one-window SwiftUI app. It always uses Apple AI at its highest-quality settings: 4K, HEVC 10-bit, 50 Mbps, and the encoder set to quality over speed.

```bash
scripts/build-app.sh          # builds and wraps build/Upscale.app (ad-hoc signed, runs locally)
open build/Upscale.app
```

- **Adding videos:** drop videos or a whole folder on the window, or use **Add Videos…**. Each row shows the source size → 4K and the length.
- **Output folder:** pick it with the folder button. The default is `~/Movies/Upscaled`. Files are named `<name> - 4K.mp4`.
- **Start / Stop:** each running row shows progress, fps and time left, then **Show in Finder** when done. **Stop** discards the video in progress and puts it back in the queue.
- **Safe output:** files are written under a hidden `.partial` name and renamed only after their duration matches the source.
- **Other formats:** WebM and MKV (AV1, Opus) are first converted to a ProRes intermediate with ffmpeg, if it's installed.
- **While running:** the Mac doesn't idle-sleep, and quitting mid-job asks first.
- **Scripting:**
  ```bash
  build/Upscale.app/Contents/MacOS/Upscale --add FILES_OR_FOLDERS --output DIR --start
  ```

## The CLI

```bash
metalfx-upscale <input> <output> [options]
```

```bash
# Video → 4K with MetalFX (fast) or Apple AI (quality)
metalfx-upscale input.mp4 out.mp4
metalfx-upscale input.mp4 out.mp4 --engine ml --bitrate 50

# Other sizes: presets keep the aspect ratio and follow the source's orientation
metalfx-upscale input.mp4 out.mp4 --size 1440p
metalfx-upscale input.mov out.mov --size 3840x1600
metalfx-upscale photo.jpg photo-2x.png --size 2x       # images use MetalFX

# Try a 30-second excerpt first
metalfx-upscale input.mp4 test.mp4 --engine ml --start 600 --duration 30
```

| Option | Meaning |
|---|---|
| `--size` | `4k` (default), `8k`, `1440p`, `1080p`, `WIDTHxHEIGHT`, or a factor such as `2x` / `1.5x` |
| `--engine` | `metalfx` (default): fast, sharpens edges. `ml`: Apple AI, video only, 10-bit output |
| `--download-model` | With `--engine ml`: let macOS download Apple's model if it's missing |
| `--bitrate` | Video bitrate in Mbps. The default is about 0.07 bits per pixel per frame, which is 14.5 Mbps for 4K at 25 fps |
| `--start`, `--duration` | Upscale only an excerpt, in seconds |
| `--overwrite` | Replace an existing output file |
| `--dump-source`, `--dump-base`, `--dump-flow`, `--dump-crop` | Frame dumps for DetailNet training (see `metalfx-upscale --help`) |

**Output format:**
- **Video:** HEVC (`hvc1`, plays in QuickTime) in `.mp4` or `.mov`, with every audio track copied untouched. Color tags are carried over from the source; untagged video is treated as BT.709.
- **Images:** written in the format of the output extension (png, jpg, heic or tiff), keeping their color profile and orientation.

| Engine | 720p → 4K | 1080p → 4K | Output |
|---|---|---|---|
| `metalfx` | about 125 fps | (not measured) | 8-bit HEVC |
| `ml` | about 6–11 fps alone; 2–6 fps when running two at a time | about 3.5 fps alone; about 2 fps each when running two | 10-bit HEVC Main10 |

## Batch upscaling

```bash
scripts/batch-upscale.sh output/batch-jobs.txt   # one input path per line
```

- **Settings:** Apple AI, 4K, 10-bit HEVC at 50 Mbps, two videos at a time. The `PARALLEL`, `BITRATE` and `DEST` environment variables override these.
- **Safe output:** each output is written to `output/4K/.partial-<name>.mp4`. It's renamed to `output/4K/<name> - 4K.mp4` only after its duration (±0.5 s), frame count, 4K size and audio are checked against the source.
- **Failures:** a failed job retries once.
- **Resuming:** re-running the script skips outputs that pass the checks. A stopped job can't resume mid-file; it starts over.
- **Other formats:** WebM and MKV are converted to a ProRes intermediate first.
- **Logs:** progress goes to `output/BATCH-STATUS.md`, and per-job logs to `output/batch-logs/`.

**Result for this project:**
- **The episodes:** all six episodes in `~/Downloads` were finished on 2026-10-22, three 720p and three 1080p, 48–58 minutes each. Each output is about 17–22 GB, 3840×2160, HEVC Main10, averaging about 50 Mbps, and each was checked against its source.
- **Speed:** the 720p episodes took about 4–6 hours each; the 1080p episodes took about 8.3–9.6 hours each when running two at a time.

## How it works

**MetalFX path** (`FrameUpscaler`, also used for images):

```
AVAssetReader (hardware decode → BGRA, IOSurface)
  → CVMetalTextureCache (zero-copy texture)
  → MTLFXSpatialScaler (perceptual mode) → private texture → blit
  → AVAssetWriter pixel-buffer pool (zero-copy) → hardware HEVC encode
```

**Apple AI path** (`AppleSuperResolution` + `AppleMLEngine`):

```
AVAssetReader (hardware decode → half-float RGBA, the model's input format)
  → VTFrameProcessor + VTSuperResolutionScaler (exactly 4×, using the previous frame and output for temporal stability)
  → VTPixelTransferSession: resize to the target + RGB → 10-bit 4:2:0, tagged with the source's color
  → AVAssetWriter → hardware HEVC Main10 encode (audio copied)
```

- **Scaling:** the model only does 4× and takes at most 1920×1080. 720p becomes 5120×2880 and 1080p becomes 7680×4320, and the transfer session averages that down to 3840×2160. Targets beyond 4× (for example 640×480 → 2880×2160) are finished by the same resize.
- **Engine structure:** `AppleSuperResolution` is only the 4× model. `AppleMLEngine` adds the resize, plus optional frame dumps for DetailNet training.
- **Frame order:** frames must arrive in order, because the model keeps the previous source/output pair.
- **Memory:** about 3.4 GB per 720p job and about 7.4 GB per 1080p job.

## Quality settings

Everything VideoToolbox lets you choose is set to quality over speed:

| Stage | Setting |
|---|---|
| AI upscaler | `qualityPrioritization: .normal`. It's the only level VideoToolbox offers for super resolution; `.quality` exists only for frame-rate conversion, optical flow and motion blur |
| Resize | `kVTDownsamplingMode_Average` (the higher-quality filter); destination color tags match the encoder's |
| Encoder | `PrioritizeEncodingSpeedOverQuality = false`, `RealTime = false`, `MaximizePowerEfficiency = false`, spatial adaptive QP left on |
| Bitrate | 50 Mbps average, 10-bit 4:2:0 (HEVC Main10) |

## Models compared

Numbers come from `output/sr-bench/summary.json` (12 clips; see [Benchmark](#benchmark)), except where noted.
- **Detail** and **jitter** are ratios to Apple AI. Jitter is motion-compensated frame-to-frame change; lower is steadier.
- **LPIPS** is the perceptual distance to the real 1080p on the ×4 ground-truth clips; lower is better.
- **Speed** was measured while the 4K batch shared the GPU, so speeds are only comparable with each other.

| Model | Detail | Jitter | LPIPS | s per frame (720p → 4K) | Hours per hour of video |
|---|---|---|---|---|---|
| Bicubic resize | 0.91 | 0.17 | 0.478 | 0.09 | 2.2 |
| **Apple AI** | 1.00 | 1.00 | 0.396 | 0.35 | 8.5 |
| Real-ESRGAN general-x4v3 | 1.17 | 0.78 | 0.386 | 0.48 | 11.6 |
| Real-ESRGAN x4plus | 1.17 | 1.13 | 0.363 | 6.4 | 155 |
| RealBasicVSR | 1.28 | 1.22 | 0.406 | 2.7 | 66 |
| RealViformer | 1.11 | 1.40 | 0.375 | 2.5 | 61 |
| FlashVSR v1.1 tiny (MPS port) | about 1.8 | about 1.07 within its 4-frame groups, but pulses every 4 frames | about 0.32 (vs Apple 0.41 on its subset) | 14–40 | Too slow; used as a teacher |
| SeedVR2 7B, 8-bit (not on the bench) | Much higher (generative) | +23% flicker on rock (frame by frame, MLX) | not measured | 35 (PyTorch video mode) to 70 (MLX per frame) | Weeks |

**SeedVR2 notes** (2026-10-12, removed since):
- **PyTorch (numz ComfyUI-SeedVR2 CLI):** needs `PYTORCH_MPS_HIGH_WATERMARK_RATIO=1.0`, because otherwise memory hits 72–98 GB and swaps. It also needs tiled encode and decode, because otherwise MPS fails with "tensor dims larger than INT_MAX".
- **MLX (mlx-gen):** its video mode runs out of memory on 64 GB at any upscale; image mode works at 4K but has no temporal consistency.

## DetailNet (research)

Status: **milestone 1 (M1) is a no-go on detail.** The adversarial-loss round (r3) has been approved but not yet run.

**Goal:** visibly more real detail than Apple AI, with no new motion artifacts, at 10 hours or less per 58-minute 720p episode, running in the app via Core ML.

**Approach:**
- DetailNet is a small recurrent network.
- It adds a high-pass texture layer on top of Apple AI's 4K frame, so it can't change brightness, color or shapes.
- It warps its previous texture along VideoToolbox optical flow, so texture stays locked to the scene, and resets at scene cuts.
- It was trained on this Mac on 1,100 sequences from the project's episodes, with FlashVSR outputs as the teacher targets.

**Documents:**
- Design: `docs/superpowers/specs/2026-10-13-detail-layer-sr-design.md`
- M1 plan: `docs/superpowers/plans/2026-10-13-detail-layer-m1.md`
- Progress ledger: `.superpowers/sdd/2026-10-13-detail-layer-m1/progress.md`

**Gates** (`python bench/gates.py MODEL`) compare each run against `apple-base`, which is Apple AI's output before compression:

| Run | Detail (need ≥ 1.5) | Jitter (≤ 1.1) | Pulse (≤ 1.1) | Cut leak (≤ Apple's 0.52) | LPIPS ×3 ground truth (≤ Apple's 0.317) | Result |
|---|---|---|---|---|---|---|
| r0 | 1.03 | 1.05 | 1.01 | 0.29 | 0.287 | 7/8 |
| r1 | 1.11 | 0.98 | 1.01 | 0.13 | 0.256 | 7/8 |
| **r2** | **1.13** | 1.00 | 1.01 | 0.14 | **0.252** | 7/8 |

**By eye:** r2 is not distinguishable from Apple AI as an improvement. Its measured wins mostly come from smoothing away Apple's painterly artifacts, which LPIPS rewards. Its texture isn't locked well enough to amplify: scaling it 3× reaches 1.59× detail but 1.28× jitter.

**Proposed next steps:**
- an adversarial loss (r3);
- a stronger temporal texture lock;
- a fully learned upscaler (option B).

**Running it:** see `bench/README.md` ("DetailNet and the go / no-go gates"). In short:

```bash
source bench/env.sh
python -m detailnet.bench_dumps output/sr-bench/clips/c-* output/sr-bench/clips/g3-*
python -m detailnet.run_bench --base-only --name apple-base output/sr-bench/clips/c-* output/sr-bench/clips/g3-*
python -m detailnet.run_bench --ckpt output/detail-runs/r2/ckpt_014000.pt --name dn-r2 output/sr-bench/clips/c-* output/sr-bench/clips/g3-*
bench/score_all.sh 3 && python bench/gates.py dn-r2
python bench/compare.py output/sr-bench/clips/g3-lizard-grass output/sr-bench/compare/x gt apple-base dn-r2
```

## Benchmark

`bench/` compares video upscalers on this project's own footage, measuring detail, temporal stability, behavior at cuts, and speed. The details are in `bench/README.md`.

- **Clips (`bench/clips.json`, `bench/clips-g3.json`):**
  - real 720p and 1080p shots upscaled to 4K: static foliage, slow and fast pans over rock, moving water, a hard cut, and a dark scene;
  - ground-truth clips: real 1080p frames shrunk to 480×270 (×4) or 640×360 (×3), then upscaled back and scored against the original.
- **Metrics (`bench/metrics.py`):**
  - motion-compensated jitter, split into static, slow and fast motion;
  - pulse, a 4-frame periodic shimmer;
  - cut leak, how much state carries across a cut;
  - LPIPS, DISTS, PSNR and tLP on the ground-truth clips;
  - speed.
- **Comparisons (`bench/compare.py`):** labelled 1:1 side-by-side videos of the most detailed 960×540 window, plus time-slice images in which steady texture draws clean streaks.
- **Environment:** Python lives in `third_party/venv` (set up by `bench/env.sh`); third-party weights live in `third_party/`, and results in `output/sr-bench/`.

## Limits and known issues

- **Apple AI:**
  - Video only; input up to 1920×1080. The model only does 4×, and Apple's low-latency variant (`VTLowLatencySuperResolutionScaler`) accepts no scale factor at 1080p.
  - Renders bright highlights slightly darker, about 5/255 on a bright test frame.
  - The bench also found the encoded output about 2 levels darker than Apple's raw output. That may affect the batch files too; it's unexplained.
  - Its model-download callback can report `VTFrameProcessorAssetDownloadFailed` (-19743) even when the model installed fine, so the tool trusts the model status instead.
- **MetalFX spatial** can't invent detail. Its temporal scalers need a game renderer's depth and motion vectors.
- **SDR only:** HDR (PQ/HLG) input is rejected. Untagged video is treated as BT.709.
- **Upscale only:** a target smaller than the source is an error.
- **Images with transparency** are processed, but MetalFX isn't designed for alpha.
- **Jobs can't resume mid-file:** stopping one loses its progress. Freezing the process with `kill -STOP` (and resuming with `kill -CONT`) keeps it, but not across a reboot.
- **Speed:** Apple AI dominates the run time, and two jobs at once give only about 14% more total throughput at 1080p.

## Project layout

```
Sources/
  UpscaleCore/          shared library: engines, video/image pipelines, sizing, frame dumps, optical flow
    VideoUpscaler.swift      decode → engine → encode, audio pass-through, cancellation, inspect()
    AppleMLEngine.swift      AppleSuperResolution (4× model) + AppleMLEngine (resize, 10-bit) + model download
    FrameUpscaler.swift      MetalFX spatial scaler
    ImageUpscaler.swift      still images via MetalFX
    TargetSize.swift         4k/1440p/WxH/2x parsing and resolution
    DumpSet.swift, FrameDump.swift, OpticalFlow.swift   DetailNet training dumps
  metalfx-upscale/      CLI
  UpscaleApp/           SwiftUI app (queue, converter, UI)
Tests/UpscaleCoreTests/ Swift tests (sizing, dumps, optical flow)
scripts/
  batch-upscale.sh      resumable, verifying batch runner
  build-app.sh          builds build/Upscale.app
bench/                  super-resolution benchmark (Python)
detailnet/              DetailNet model, data, training, teacher, bench integration (+ tests)
docs/superpowers/       DetailNet design spec and M1 plan
output/
  4K/                   final upscaled episodes
  sr-bench/             benchmark clips, outputs, metrics, comparison videos
  detail-runs/          DetailNet training runs and gate results
```

## Building and testing

```bash
swift build -c release        # CLI and library
scripts/build-app.sh          # app
swift test                    # Swift tests
source bench/env.sh && python -m pytest detailnet/tests   # DetailNet and bench tests
```

Requires macOS 14+ on Apple silicon. The `ml` engine and the app need macOS 26+. ffmpeg (`brew install ffmpeg`) is needed for WebM/MKV input and for the benchmark.

## License

Apache-2.0 (see `LICENSE` and `NOTICE`).
- **DetailNet r2 weights:** published at https://huggingface.co/tsonubin/DetailNet under the same license.
- **Credits:** DetailNet was trained against FlashVSR v1.1 (Apache-2.0) outputs.
- **Footage:** the documentary footage used for training and benchmarking is copyrighted and isn't included.

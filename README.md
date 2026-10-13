# metalfx-upscale

Upscale images and videos on the Mac with Apple's SDKs, to 4K by default or any size you ask for. Two engines:

| Engine | API | Speed (720p → 4K, M5 Max) | Result |
|---|---|---|---|
| `metalfx` (default) | MetalFX `MTLFXSpatialScaler` | ~125 fps | Cleaner edges than a plain resize; no new detail |
| `ml` | VideoToolbox `VTSuperResolutionScaler` | ~11 fps (720p source), ~3.5 fps (1080p) | Reconstructs texture (rock, foliage); video only; 10-bit output |

## App

A minimal SwiftUI app for the Apple AI route at its highest quality (4K, HEVC 10-bit, 50 Mbps, encoder set to quality over speed):

```bash
scripts/build-app.sh          # → build/Upscale.app
open build/Upscale.app
```

Drop videos or a whole folder on the window (or **Add Videos…**), pick the output folder (default `~/Movies/Upscaled`), press **Start**. Each row shows source → 4K, then progress with fps and time left, then **Show in Finder**. WebM/MKV are converted with ffmpeg first if it's installed. Files are written under a hidden `.partial` name and renamed only after their duration checks out; **Stop** discards the in-progress video. The Mac stays awake while it runs, and quitting mid-job asks first. For scripting: `Upscale.app/Contents/MacOS/Upscale --add FILES_OR_FOLDERS --output DIR --start`.

## Build

```bash
swift build -c release
```

The binary is `.build/release/metalfx-upscale`. Requires macOS 14+ on Apple silicon; the `ml` engine needs macOS 26+ (built and tested with Xcode 27 on an M5 Max).

## Use

```bash
# Video → 4K (aspect ratio kept; portrait sources become 2160x3840)
metalfx-upscale input.mp4 output.mp4

# Other sizes: presets, exact dimensions, or a factor
metalfx-upscale input.mp4 out.mp4 --size 1440p
metalfx-upscale input.mov out.mov --size 3840x1600
metalfx-upscale photo.jpg photo-2x.png --size 2x

# Try a 30-second excerpt first
metalfx-upscale input.mp4 test.mp4 --start 600 --duration 30

# Apple's ML super-resolution (first run: let macOS download the model from Apple)
metalfx-upscale input.mp4 output-ml.mp4 --engine ml --download-model
```

| Option | Meaning |
|---|---|
| `--size` | `4k` (default), `8k`, `1440p`, `1080p`, `WIDTHxHEIGHT`, or a factor such as `2x` / `1.5x` |
| `--engine` | `metalfx` (default) or `ml` |
| `--download-model` | With `--engine ml`: download Apple's model if it isn't installed |
| `--bitrate` | Video bitrate in Mbps (default ≈ 0.07 bits/pixel/frame, i.e. 14.5 Mbps for 4K25) |
| `--start`, `--duration` | Upscale only an excerpt (seconds) |
| `--overwrite` | Replace an existing output file |

Video is written as HEVC (`hvc1`, plays in QuickTime) in `.mp4` or `.mov`; audio tracks are copied untouched. Images are written in the format of the output extension (png, jpg, heic, tiff).

## How it works

```
AVAssetReader (hardware decode → BGRA, IOSurface)
  → CVMetalTextureCache (zero-copy texture)
  → MTLFXSpatialScaler (perceptual mode) → private texture → blit
  → AVAssetWriter pixel-buffer pool (zero-copy) → hardware HEVC encode
```

Frames never leave GPU-visible memory. `FrameUpscaler` is the MetalFX core and is shared by the video and image paths.

The `ml` engine swaps the middle step: the decoder outputs half-float RGBA, `VTFrameProcessor` upscales exactly 4× using the previous frame for temporal stability, and `VTPixelTransferSession` resizes to the target (720p → 5120x2880 → 3840x2160; 1080p → 7680x4320 → 3840x2160) while converting to 10-bit 4:2:0, encoded as HEVC Main10. Targets beyond 4× (e.g. 640x480 → 2880x2160) are finished by that resize.

## Batch

```bash
scripts/batch-upscale.sh jobs.txt   # one input path per line
```

Runs the `ml` engine at 30 Mbps, two videos at a time (`PARALLEL`, `BITRATE` env vars override). Non-AVFoundation inputs (WebM, MKV, AV1, Opus) are first converted to a ProRes intermediate. Each output is written to a hidden `.partial` file and renamed to `output/<name> - 4K Apple AI.mp4` only after its duration, frame count, 4K size and audio are verified; failures retry once. Re-running skips verified outputs. Progress goes to `output/BATCH-STATUS.md`, logs to `output/batch-logs/`.

Measured on an M5 Max, 1280x720 → 3840x2160: about **125 fps end-to-end** for video (5× real time for 25 fps material); a single 4K still takes ~17 ms on the GPU including first-use setup.

## Limits

- **`ml` engine:** video only, input up to 1920x1080 (the model only does 4×; Apple's low-latency 2× variant doesn't accept 1080p). Apple's model renders bright highlights very slightly darker (≈5/255 on a bright test frame). Its model download completion handler can report `VTFrameProcessorAssetDownloadFailed` (-19743) even when the model installed fine; the tool trusts the model status instead.
- **MetalFX spatial is a sharpening upscaler, not an ML one.** It produces cleaner edges than a plain resize but cannot invent detail. MetalFX's temporal scalers need depth and motion vectors from a game renderer, so they don't apply to video files.
- 8-bit SDR pipeline: HDR (PQ/HLG) video is rejected. Untagged video is treated as BT.709.
- Upscale only: a target smaller than the source is an error.
- Images with transparency are processed, but MetalFX isn't designed for alpha; expect the best results on opaque images.

## Tests

```bash
swift test
```

Covers size parsing and resolution (presets, orientation, aspect ratio, even-dimension rounding for video).

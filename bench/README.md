# Super-resolution bench

Compares video upscalers on this project's own footage, for detail, temporal stability, behaviour at cuts, and speed on this Mac.

```bash
source bench/env.sh                                    # venv + MPS memory cap
python bench/analyze_shots.py EPISODE.mp4 output/sr-bench/shots   # cut + motion map
python bench/make_clips.py bench/clips.json output/sr-bench/clips # frame-exact clips
bench/run_all.sh bicubic apple realesrgan-general realesrgan realbasicvsr realviformer flashvsr
bench/score_all.sh 3                                   # metrics for every finished output
python bench/summarize.py output/sr-bench/summary.json # scoreboard
python bench/compare.py CLIP_DIR output/sr-bench/compare MODEL...  # side-by-side video + time slice
```

## Clips

`bench/clips.json` lists 12 clips, each chosen from the shot maps for one failure mode:

- **Real clips:** 720p and 1080p sources upscaled to 4K. They cover static foliage, slow and fast pans over rock, moving water, a hard cut, and a dark scene.
- **Ground-truth clips:** 1080p frames downscaled to 480x270 and re-encoded with H.264 at the broadcast bits/pixel, then upscaled 4× and compared with the real 1080p.

## Metrics (`metrics.py`)

- **jitter:** motion-compensated frame change minus the bicubic upscale's change, divided by the output's own gradient (see the docstring). Motion comes from optical flow on each output.
  - `validate_jitter.py` checks it against synthetic cases. The real 1080p and a sharpened copy of it score alike. Texture re-drawn every frame scores about 2× higher.
  - It is reported per motion bucket of the low-res source: static, slow, or fast.
- **cut_leak:** how much a run that ignores scene cuts differs from a run that resets at each cut, over the 25 frames after the cut.
- **On ground-truth clips:**
  - LPIPS and DISTS (perceptual distance to the real frame)
  - PSNR
  - tLP (temporal change that doesn't match the real footage)
- **Scored area:** 4K outputs are scored on their central 1920x1080 at 1:1 pixels. Real clips are scored on their first 100 frames.

## Models (`models.py`)

| key | what | notes |
|---|---|---|
| `apple` | VideoToolbox `VTSuperResolutionScaler` via this project's CLI | Runs one CLI call per shot. Output goes through HEVC at about 1.6 bits/pixel. |
| `bicubic` | Plain resize | Baseline floor |
| `realesrgan`, `realesrgan-general` | Real-ESRGAN x4plus / general-x4v3 | Per-frame GAN |
| `realbasicvsr` | RealBasicVSR | Bidirectional recurrent; runs on overlapping 24-frame windows |
| `realviformer` | RealViformer (ECCV'24) | Forward-only recurrent, channel attention |
| `flashvsr` | FlashVSR v1.1 tiny | One-step streaming diffusion. Ported to MPS in `flashvsr_mps.py` with pure-PyTorch block-sparse attention. |

Heavy models (`realesrgan`, `realbasicvsr`, `realviformer`, `flashvsr`) run on 4K clips with `--center-crop --frames 100`. This upscales only the low-res region behind the scored area, plus a 32 px margin. The scoreboard scales their speed back up to a full frame.

Third-party code and weights live in `third_party/`; results go in `output/sr-bench/`. Timings taken while the 4K batch runs share the GPU with it, so they are only comparable to each other.

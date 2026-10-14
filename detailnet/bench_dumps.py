"""Frame dumps of the bench clips, for running DetailNet on exactly the base Apple AI produces in the bench.

    python -m detailnet.bench_dumps output/sr-bench/clips/c-* output/sr-bench/clips/g3-* [--binary PATH]

Writes, per shot of each clip,

    CLIP_DIR/dumps/shot<k>/{source,base,flow}/

from the dev build of the Swift CLI (`--engine ml --dump-source/--dump-base/--dump-flow`). Each shot is converted to
ProRes 4444 first and run through the CLI on its own, the same as `bench/run_model.run_apple` does for the Apple
output (same ffmpeg arguments, bt709 tags included), so frame 0 of every dump is a shot start: the CLI marks it
`valid: false` in meta.json (it has no previous frame, so no flow), and `detailnet.run_bench` starts DetailNet's state
there. The Swift output video is a throwaway temp file.

What is dumped:
  4K clips (720p -> 3840x2160)   the LR box `run_model.crop_plan` gives for the scored centre 1920x1080: 704x424 at
                                 (288, 148) for 720p, so the base is 2112x1272 (the scored region plus 96 px of context
                                 on every side, which run_bench cuts off after DetailNet has seen it);
  gt3 clips (640x360 -> 1920x1080)  the full frame;
  real clips                     the first 100 frames only (`metrics.REAL_SCORE_FRAMES`, the span every model is scored
                                 on), split at the cuts inside them: c-cut-nest's cut at 75 gives shots [0, 75) and
                                 [75, 100).

A finished shot directory is kept; a shot is built in a hidden `.shot<k>.tmp` directory and renamed once all three dumps
check out, so an interrupted run leaves nothing half-written under a shot's name. Rerunning finishes the rest.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import NamedTuple

from detailnet.build_data import (FFMPEG_TIMEOUT_S, SWIFT_TIMEOUT_S, TMP_VIDEO, BuildError, run_checked,
                                  swift_cmd)
from detailnet.dumps import read_meta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import metrics  # noqa: E402
from clipio import shots_of  # noqa: E402
from run_model import CROP, crop_plan  # noqa: E402

DEFAULT_BINARY = ROOT / ".build-dev" / "release" / "metalfx-upscale"
DUMP_KINDS = ("source", "base", "flow")
CHANNELS = {"source": 3, "base": 3, "flow": 4}
BITRATE_MBPS = 5  # of the throwaway video


class DumpPlan(NamedTuple):
    frames: int  # frames dumped (the scored span of the clip)
    shots: list  # [(start, end)] in clip frames
    crop: tuple  # (x, y, w, h) LR box dumped, in source px
    scale: int  # base px per source px
    size: str  # --size handed to the CLI


def dump_plan(clip, crop_size=CROP):
    """What to dump for a clip.json dict (`crop_size` is the scored region of a 4K target; tests shrink it)."""
    n = clip["frames"]
    if clip["kind"] == "real":
        n = min(n, metrics.REAL_SCORE_FRAMES)
    lw, lh = clip["lr_size"]
    tw, th = clip["target"]
    scale = tw // lw
    if scale * lw != tw or scale * lh != th:
        raise ValueError(f"{clip['name']}: target {tw}x{th} is not a whole multiple of the {lw}x{lh} source")
    if tw > crop_size[0]:
        x, y, w, h, _, _ = crop_plan(clip, crop_size)
        crop = (x, y, w, h)
    else:
        crop = (0, 0, lw, lh)
    return DumpPlan(n, shots_of({**clip, "frames": n}), crop, scale, f"{tw}x{th}")


def prores_cmd(lr, start, end, dst):
    """ffmpeg: frames [start, end) of the clip's lr.mkv as ProRes 4444, tagged bt709. These are the arguments
    `run_model.run_apple` uses for the Apple output; keep the two in step."""
    return ["ffmpeg", "-v", "error", "-y", "-i", str(lr), "-vf",
            f"select='between(n,{start},{end - 1})',setpts=N/FRAME_RATE/TB",
            "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuv444p10le",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", str(dst)]


def check_shot(shot_dir, name, frames, plan):
    """Raises BuildError unless the three dumps in `shot_dir` hold `frames` frames of the planned geometry and flow
    frame 0 is marked invalid. `name` is how messages call the shot."""
    x, y, w, h = plan.crop
    want = {"source": (w, h), "base": (plan.scale * w, plan.scale * h), "flow": (w, h)}
    for kind in DUMP_KINDS:
        try:
            meta = read_meta(Path(shot_dir) / kind)
        except (OSError, ValueError) as error:
            raise BuildError(f"{name}: {kind}/meta.json is missing or unreadable ({error})")
        if meta["frames"] != frames:
            raise BuildError(f"{name}: {kind} has {meta['frames']} frames, expected {frames}")
        got = (meta["channels"], meta["width"], meta["height"])
        if got != (CHANNELS[kind], *want[kind]):
            raise BuildError(f"{name}: {kind} is {got} (channels, width, height), expected "
                             f"{(CHANNELS[kind], *want[kind])}")
        if list(meta["crop"]) != [x, y, w, h]:
            raise BuildError(f"{name}: {kind} was dumped with crop {meta['crop']}, expected {[x, y, w, h]}")
    if read_meta(Path(shot_dir) / "flow")["valid"][0]:
        raise BuildError(f"{name}: flow frame 0 should be marked invalid (a shot start has no previous frame)")


def shot_is_done(shot_dir, name, frames, plan):
    try:
        check_shot(shot_dir, name, frames, plan)
    except (BuildError, KeyError, IndexError):
        return False
    return True


def build_clip(clip_dir, binary=DEFAULT_BINARY, run=run_checked, log=print):
    """Dump every shot of one clip that is not finished yet; -> the shots [(start, end)]. BuildError on a failed tool
    or a dump that does not check out."""
    clip_dir = Path(clip_dir)
    clip = json.loads((clip_dir / "clip.json").read_text())
    plan = dump_plan(clip)
    dumps = clip_dir / "dumps"
    dumps.mkdir(exist_ok=True)
    for k, (a, b) in enumerate(plan.shots):
        name = f"{clip['name']} shot{k}"
        final, work = dumps / f"shot{k}", dumps / f".shot{k}.tmp"
        if shot_is_done(final, name, b - a, plan):
            log(f"{name}: frames {a}..{b - 1} already dumped")
            continue
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir()
        started = time.monotonic()
        try:
            prores = work / "tmp.mov"
            run(prores_cmd(clip_dir / "lr.mkv", a, b, prores), FFMPEG_TIMEOUT_S)
            run(swift_cmd(binary, prores, work, None, None, plan.crop, set(DUMP_KINDS), size=plan.size, margin=0,
                          bitrate=BITRATE_MBPS), SWIFT_TIMEOUT_S)
            check_shot(work, name, b - a, plan)
            (work / "tmp.mov").unlink(missing_ok=True)
            (work / TMP_VIDEO).unlink(missing_ok=True)
            shutil.rmtree(final, ignore_errors=True)
            work.rename(final)
        except BaseException:
            shutil.rmtree(work, ignore_errors=True)
            raise
        log(f"{name}: frames {a}..{b - 1} dumped in {time.monotonic() - started:.0f} s")
    return plan.shots


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("clip_dirs", nargs="+", type=Path)
    ap.add_argument("--binary", type=Path, default=DEFAULT_BINARY, help="the dev build of the Swift CLI")
    args = ap.parse_args(argv)
    if not args.binary.exists():
        sys.exit(f"{args.binary} not found: build it with `swift build -c release --scratch-path .build-dev`")
    failed = 0
    for clip_dir in args.clip_dirs:
        try:
            build_clip(clip_dir, args.binary)
        except BuildError as error:
            failed += 1
            print(f"FAILED {clip_dir.name}: {error}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

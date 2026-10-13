import Foundation
import UniformTypeIdentifiers
import UpscaleCore

let usage = """
    Upscale an image or video on the GPU with Apple MetalFX.

    usage: metalfx-upscale <input> <output> [options]

      --size <size>      4k (default), 8k, 1440p, 1080p, WIDTHxHEIGHT, or a factor like 2x.
                         Presets keep the aspect ratio and follow the source's orientation.
      --engine <name>    metalfx (default): MetalFX spatial scaler, fast, sharpens edges.
                         ml: Apple's ML super-resolution (video only, up to 1080p input;
                         4x by the model, any extra by resizing); slower but reconstructs
                         texture, written as 10-bit HEVC.
      --download-model   With --engine ml: let macOS download Apple's model if it's missing.
      --bitrate <mbps>   Video bitrate in Mbps (default: picked from resolution and frame rate).
      --start <sec>      Video only: start of an excerpt, in seconds.
      --duration <sec>   Video only: length of the excerpt, in seconds.
      --overwrite        Replace <output> if it already exists.

    Video is written as HEVC (.mp4 or .mov) with the audio copied untouched.
    Images are written in the format of the output extension (png, jpg, heic, tiff).
    """

struct Arguments {
    var input = URL(fileURLWithPath: "")
    var output = URL(fileURLWithPath: "")
    var size = TargetSize.uhd4K
    var engine = Engine.metalfx
    var downloadModel = false
    var bitrateMbps: Double?
    var start: Double?
    var duration: Double?
    var overwrite = false

    static func parse(_ args: [String]) throws -> Arguments {
        var result = Arguments()
        var positional: [String] = []
        var iterator = args.makeIterator()

        func value(for flag: String) throws -> String {
            guard let value = iterator.next() else { throw UpscaleError.io("\(flag) needs a value.") }
            return value
        }
        func number(for flag: String) throws -> Double {
            let text = try value(for: flag)
            guard let number = Double(text), number >= 0 else { throw UpscaleError.io("\(flag) expects a number, got '\(text)'.") }
            return number
        }

        while let arg = iterator.next() {
            switch arg {
            case "--size": result.size = try TargetSize.parse(try value(for: arg))
            case "--engine":
                let name = try value(for: arg)
                guard let engine = Engine(rawValue: name) else { throw UpscaleError.io("Unknown engine '\(name)'; use metalfx or ml.") }
                result.engine = engine
            case "--download-model": result.downloadModel = true
            case "--bitrate": result.bitrateMbps = try number(for: arg)
            case "--start": result.start = try number(for: arg)
            case "--duration": result.duration = try number(for: arg)
            case "--overwrite": result.overwrite = true
            case "-h", "--help": print(usage); exit(0)
            default:
                guard !arg.hasPrefix("--") else { throw UpscaleError.io("Unknown option \(arg).\n\n\(usage)") }
                positional.append(arg)
            }
        }
        guard positional.count == 2 else { throw UpscaleError.io(usage) }
        result.input = URL(fileURLWithPath: positional[0])
        result.output = URL(fileURLWithPath: positional[1])
        return result
    }
}

func formatTime(_ seconds: Double) -> String {
    let s = Int(seconds.rounded())
    return s >= 3600 ? String(format: "%d:%02d:%02d", s / 3600, s / 60 % 60, s % 60) : String(format: "%d:%02d", s / 60, s % 60)
}

func fileSize(_ url: URL) -> String {
    let bytes = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? Int64) ?? 0
    return ByteCountFormatter.string(fromByteCount: bytes, countStyle: .file)
}

do {
    let args = try Arguments.parse(Array(CommandLine.arguments.dropFirst()))
    let fm = FileManager.default
    guard fm.fileExists(atPath: args.input.path) else { throw UpscaleError.io("No such file: \(args.input.path)") }
    if fm.fileExists(atPath: args.output.path) {
        guard args.overwrite else { throw UpscaleError.io("\(args.output.path) already exists (pass --overwrite to replace it).") }
        try fm.removeItem(at: args.output)
    }

    if args.engine == .ml && args.downloadModel {
        guard #available(macOS 26.0, *) else { throw UpscaleError.scalerUnavailable("The ml engine needs macOS 26 or later.") }
        if !AppleMLModel.isReady {
            print("Downloading Apple's super-resolution model (macOS fetches it from Apple)…")
            try AppleMLModel.download { fraction in
                FileHandle.standardError.write(Data(String(format: "\r  %3.0f%%", fraction * 100).utf8))
            }
            FileHandle.standardError.write(Data("\n".utf8))
        }
    }

    let type = UTType(filenameExtension: args.input.pathExtension)
    if type?.conforms(to: .audiovisualContent) == true {
        var options = VideoUpscaler.Options()
        options.size = args.size
        options.engine = args.engine
        options.bitrate = args.bitrateMbps.map { Int($0 * 1_000_000) }
        options.start = args.start
        options.duration = args.duration

        var lastPrint = -1.0
        let summary = try await VideoUpscaler.upscale(input: args.input, output: args.output, options: options) { p in
            guard p.elapsed - lastPrint >= 0.5 || p.framesDone == p.framesTotal else { return }
            lastPrint = p.elapsed
            let fps = Double(p.framesDone) / max(p.elapsed, 0.001)
            let remaining = Double(max(p.framesTotal - p.framesDone, 0)) / max(fps, 0.001)
            let percent = min(100, 100 * Double(p.framesDone) / Double(p.framesTotal))
            let line = String(format: "\r  %5.1f%%  frame %d/%d  %.0f fps  ETA %@   ",
                              percent, p.framesDone, p.framesTotal, fps, formatTime(remaining))
            FileHandle.standardError.write(Data(line.utf8))
        }
        FileHandle.standardError.write(Data("\n".utf8))
        print("""
            \(summary.engine)
              \(summary.sourceWidth)x\(summary.sourceHeight) → \(summary.outputWidth)x\(summary.outputHeight), \
            \(summary.frames) frames in \(formatTime(summary.elapsed)) \
            (\(String(format: "%.0f", Double(summary.frames) / max(summary.elapsed, 0.001))) fps end-to-end)
              HEVC \(String(format: "%.1f", Double(summary.bitrate) / 1e6)) Mbps → \(args.output.path) (\(fileSize(args.output)))
            """)
    } else if type?.conforms(to: .image) == true {
        guard args.engine == .metalfx else { throw UpscaleError.unsupportedInput("The ml engine is video-only for now.") }
        let summary = try ImageUpscaler.upscale(input: args.input, output: args.output, size: args.size)
        print("""
            MetalFX spatial upscale on \(summary.device)
              \(summary.sourceWidth)x\(summary.sourceHeight) → \(summary.outputWidth)x\(summary.outputHeight) \
            in \(String(format: "%.1f", summary.gpuSeconds * 1000)) ms on the GPU
              → \(args.output.path) (\(fileSize(args.output)))
            """)
    } else {
        throw UpscaleError.unsupportedInput("Unrecognised file type: .\(args.input.pathExtension)")
    }
} catch {
    FileHandle.standardError.write(Data("error: \(error)\n".utf8))
    exit(1)
}

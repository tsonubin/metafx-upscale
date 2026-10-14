import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

/// Which frame dumps a video job writes alongside its output, for training the detail layer.
/// Each directory gets one float16 CHW file per processed frame plus a `meta.json` (see `FrameDumpWriter`).
public struct DumpOptions {
    /// The decoded source frames (RGB), cropped.
    public var source: URL?
    /// Apple's 4× frames resized to the output size (RGB), cropped to `crop` scaled by the integer scale.
    public var base: URL?
    /// VideoToolbox optical flow between consecutive source frames.
    public var flow: URL?
    /// The window to dump, in source pixels; `nil` dumps the whole frame.
    public var crop: CropRect?

    public init(source: URL? = nil, base: URL? = nil, flow: URL? = nil, crop: CropRect? = nil) {
        self.source = source
        self.base = base
        self.flow = flow
        self.crop = crop
    }
}

/// The integer factor between the source and output sizes; the base dump needs the same factor on both axes.
func baseScale(sourceWidth: Int, sourceHeight: Int, outputWidth: Int, outputHeight: Int) throws -> Int {
    guard sourceWidth > 0, sourceHeight > 0,
          outputWidth >= sourceWidth, outputHeight >= sourceHeight,
          outputWidth % sourceWidth == 0, outputHeight % sourceHeight == 0,
          outputWidth / sourceWidth == outputHeight / sourceHeight
    else {
        throw UpscaleError.unsupportedInput(
            "Frame dumps need an integer scale (e.g. 720p → 4k is 3x), but \(sourceWidth)x\(sourceHeight) → "
                + "\(outputWidth)x\(outputHeight) isn't one. Pick a --size that is a whole multiple of the source.")
    }
    return outputWidth / sourceWidth
}

/// The frame dumps of one video job. It owns the writers, records each processed frame, and writes every
/// `meta.json` in `finish()`, which the job calls however it ends so a partial dump is still readable.
///
/// Not thread-safe: `record` and `finish` must be called from one queue at a time, like `FrameDumpWriter`.
final class DumpSet {
    /// Integer output / source factor.
    let scale: Int
    /// The dumped window in source pixels.
    let crop: CropRect

    private let sourceWriter: FrameDumpWriter?
    private let baseWriter: FrameDumpWriter?
    private let baseResizer: BaseResizer?
    private let baseCrop: CropRect
    private var finished = false

    init(options: DumpOptions, sourceWidth: Int, sourceHeight: Int, outputWidth: Int, outputHeight: Int) throws {
        // Task 3 replaces this with the optical-flow writer.
        guard options.flow == nil else { throw UpscaleError.unsupportedInput("--dump-flow isn't available yet.") }
        let scale = try baseScale(
            sourceWidth: sourceWidth, sourceHeight: sourceHeight, outputWidth: outputWidth, outputHeight: outputHeight)
        let crop = options.crop ?? CropRect(x: 0, y: 0, width: sourceWidth, height: sourceHeight)
        guard crop.x >= 0, crop.y >= 0, crop.width > 0, crop.height > 0,
              crop.x + crop.width <= sourceWidth, crop.y + crop.height <= sourceHeight
        else {
            throw UpscaleError.unsupportedInput(
                "The crop \(crop.x),\(crop.y),\(crop.width),\(crop.height) doesn't fit inside the \(sourceWidth)x\(sourceHeight) source.")
        }
        let baseCrop = CropRect(x: crop.x * scale, y: crop.y * scale, width: crop.width * scale, height: crop.height * scale)

        self.scale = scale
        self.crop = crop
        self.baseCrop = baseCrop
        sourceWriter = try options.source.map {
            try FrameDumpWriter(
                directory: $0, kind: "source", channels: 3, width: crop.width, height: crop.height,
                sourceSize: (sourceWidth, sourceHeight), scale: 1, crop: crop)
        }
        if let directory = options.base {
            baseWriter = try FrameDumpWriter(
                directory: directory, kind: "base", channels: 3, width: baseCrop.width, height: baseCrop.height,
                sourceSize: (sourceWidth, sourceHeight), scale: scale, crop: crop)
            baseResizer = try BaseResizer(width: outputWidth, height: outputHeight)
        } else {
            baseWriter = nil
            baseResizer = nil
        }
    }

    /// Records one frame: the decoded `source` and, when a base dump is wanted, the model's 4× `upscaled` frame
    /// resized to the output size. Both are computed before either is written, so the dumps keep equal frame counts.
    func record(source: CVPixelBuffer, upscaled: CVPixelBuffer, at time: CMTime) throws {
        let sourcePlanes = try sourceWriter.map { _ in try Self.rgbPlanes(of: source, crop: crop) }
        var basePlanes: [Float16]?
        if let baseResizer {
            basePlanes = try Self.rgbPlanes(of: try baseResizer.resize(upscaled), crop: baseCrop)
        }
        if let sourceWriter, let sourcePlanes { try sourceWriter.write(sourcePlanes, pts: time.seconds) }
        if let baseWriter, let basePlanes { try baseWriter.write(basePlanes, pts: time.seconds) }
    }

    /// Writes every `meta.json` for the frames recorded so far. Safe to call more than once.
    func finish() throws {
        guard !finished else { return }
        finished = true
        var firstError: Error?
        for writer in [sourceWriter, baseWriter].compactMap({ $0 }) {
            do { try writer.finish() } catch { firstError = firstError ?? error }
        }
        if let firstError { throw firstError }
    }

    /// The `crop` window of a `64RGBAHalf` buffer as planar R, G, B half floats (alpha dropped).
    static func rgbPlanes(of buffer: CVPixelBuffer, crop: CropRect) throws -> [Float16] {
        let format = CVPixelBufferGetPixelFormatType(buffer)
        guard format == kCVPixelFormatType_64RGBAHalf else {
            throw UpscaleError.unsupportedInput(
                "Frame dumps need half-float RGBA frames, but the buffer is format \(fourCC(format)).")
        }
        let (width, height) = (CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer))
        guard crop.x >= 0, crop.y >= 0, crop.width > 0, crop.height > 0,
              crop.x + crop.width <= width, crop.y + crop.height <= height
        else {
            throw UpscaleError.unsupportedInput(
                "The crop \(crop.x),\(crop.y),\(crop.width),\(crop.height) doesn't fit inside a \(width)x\(height) frame.")
        }
        guard CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess,
              let base = CVPixelBufferGetBaseAddress(buffer)
        else { throw UpscaleError.io("Could not read the frame's pixels for the dump.") }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }

        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let plane = crop.width * crop.height
        var planes = [Float16](repeating: 0, count: 3 * plane)
        planes.withUnsafeMutableBufferPointer { out in
            for y in 0..<crop.height {
                let row = (base + (crop.y + y) * stride).assumingMemoryBound(to: Float16.self) + 4 * crop.x
                let offset = y * crop.width
                for x in 0..<crop.width {
                    out[offset + x] = row[4 * x]
                    out[plane + offset + x] = row[4 * x + 1]
                    out[2 * plane + offset + x] = row[4 * x + 2]
                }
            }
        }
        return planes
    }

    private static func fourCC(_ format: OSType) -> String {
        let bytes = [24, 16, 8, 0].map { UInt8((format >> $0) & 0xFF) }
        let text = String(decoding: bytes, as: UTF8.self)
        return bytes.allSatisfy { $0 >= 32 && $0 < 127 } ? "'\(text)'" : String(format)
    }
}

/// Resizes Apple's 4× frame to the output size as half-float RGBA, averaging when it shrinks: the same step the
/// encoder path uses, minus the YCbCr conversion. This is the base the detail network is fed.
private final class BaseResizer {
    private let pool: CVPixelBufferPool
    private let transfer: VTPixelTransferSession

    init(width: Int, height: Int) throws {
        var pool: CVPixelBufferPool?
        let attributes = PixelBuffers.attributes(format: kCVPixelFormatType_64RGBAHalf, width: width, height: height)
        CVPixelBufferPoolCreate(nil, nil, attributes as CFDictionary, &pool)
        var transfer: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &transfer)
        guard let pool, let transfer else { throw UpscaleError.scalerUnavailable("Could not set up buffers for the base dump.") }
        VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DownsamplingMode, value: kVTDownsamplingMode_Average)
        self.pool = pool
        self.transfer = transfer
    }

    deinit {
        VTPixelTransferSessionInvalidate(transfer)
    }

    func resize(_ upscaled: CVPixelBuffer) throws -> CVPixelBuffer {
        var resized: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &resized)
        guard let resized else { throw UpscaleError.io("Could not allocate a frame for the base dump.") }
        let status = VTPixelTransferSessionTransferImage(transfer, from: upscaled, to: resized)
        guard status == noErr else { throw UpscaleError.scalerUnavailable("Resizing the base dump frame failed (\(status)).") }
        return resized
    }
}

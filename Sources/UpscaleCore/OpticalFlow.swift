import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

/// Dense optical flow between consecutive video frames, from VideoToolbox's `VTOpticalFlowConfiguration`.
///
/// For the frames t−1 (`previous`) and t (`current`) it returns two fields, 2 planes (x, then y) of H×W half floats
/// each, in **source pixels**, with +x to the right and +y down:
/// - `backward`: from frame t to frame t−1 — at a pixel of frame t, where that content was in frame t−1. Sampling
///   t−1 at `p + backward(p)` warps it onto t.
/// - `forward`: from frame t−1 to frame t.
///
/// VideoToolbox names them the same way (`VTFrameProcessorOpticalFlow.forwardFlow` runs from `sourceFrame` to
/// `nextFrame`); `OpticalFlowTests` checks this against a known shift.
///
/// The flow is smaller than the frame: a quarter of its size up to 720p (320×180 for 1280×720), an eighth at 1080p and
/// 4K (240×135 for 1920×1080), and VideoToolbox reports it in pixels of that smaller grid. Values are multiplied by the
/// frame/grid size ratio, the same on both axes. That only holds when the grid covers the whole frame at that ratio,
/// which the initialiser checks: for sizes that don't get such a grid (854×480, 1280×544, 256×144, …) it throws
/// rather than hand back flow that is wrong or only valid on part of the frame. Not thread-safe.
@available(macOS 26.0, *)
final class OpticalFlowEstimator {
    let frameWidth: Int
    let frameHeight: Int
    /// Size of the flow grid `flow` returns.
    let flowWidth: Int
    let flowHeight: Int

    private let processor = VTFrameProcessor()
    /// The format VideoToolbox wants its frames in.
    private let inputFormat: OSType
    private let forwardBuffer: CVPixelBuffer
    private let backwardBuffer: CVPixelBuffer
    /// Source pixels per flow-grid pixel, for both axes.
    private let scale: Float
    private var converter: FrameConverter?
    /// Counts submissions so each frame gets its own, increasing, presentation time.
    private var submissions: Int64 = 0

    /// Throws when VideoToolbox can't do optical flow at this size, or when its flow grid doesn't cover the frame at
    /// a whole-number scale of 4 or 8 (the grid may be one pixel off the rounded-up frame/4 or frame/8). VideoToolbox
    /// picks the grid from a few preset sizes; when the frame doesn't fit one it pads the grid, and the flow covers
    /// only the top-left part of it, with values still in units of (frame width / grid width). 256×144 (an 80×60
    /// grid), 854×480 (240×135) and 1280×544 (320×180) are such sizes. 320×180, 640×360, 960×540, 1280×720,
    /// 1920×1080 and 3840×2160 are covered.
        init(width: Int, height: Int) throws {
        guard VTOpticalFlowConfiguration.isSupported else {
            throw UpscaleError.scalerUnavailable("VideoToolbox optical flow isn't supported on this Mac (needs macOS 26+).")
        }
        guard let configuration = VTOpticalFlowConfiguration(
            frameWidth: width, frameHeight: height, qualityPrioritization: .quality,
            revision: VTOpticalFlowConfiguration.defaultRevision)
        else {
            throw UpscaleError.unsupportedInput(
                "VideoToolbox optical flow can't handle \(width)x\(height) frames (up to 8192x4320).")
        }
        guard let inputFormat = configuration.supportedPixelFormats.first else {
            throw UpscaleError.scalerUnavailable("VideoToolbox optical flow reported no supported pixel formats.")
        }

        let attributes = configuration.destinationPixelBufferAttributes
        guard let flowWidth = Self.integer(attributes[kCVPixelBufferWidthKey as String]),
              let flowHeight = Self.integer(attributes[kCVPixelBufferHeightKey as String]),
              let flowFormat = Self.integer(attributes[kCVPixelBufferPixelFormatTypeKey as String]).map({ OSType(truncatingIfNeeded: $0) }),
              flowWidth > 0, flowHeight > 0
        else {
            throw UpscaleError.scalerUnavailable("VideoToolbox optical flow didn't describe its output buffers.")
        }
        guard FlowGrid.supportedFormats.contains(flowFormat) else {
            throw UpscaleError.unsupportedInput(
                "VideoToolbox optical flow produces \(PixelBuffers.fourCC(flowFormat)) buffers, which aren't supported.")
        }
        func ceilDiv(_ value: Int, _ divisor: Int) -> Int { (value + divisor - 1) / divisor }
        let covered = [4, 8].contains { divisor in
            abs(flowWidth - ceilDiv(width, divisor)) <= 1 && abs(flowHeight - ceilDiv(height, divisor)) <= 1
        }
        guard covered else {
            throw UpscaleError.unsupportedInput(
                "VideoToolbox returns a \(flowWidth)x\(flowHeight) optical flow grid for \(width)x\(height) frames, "
                    + "which isn't a quarter or an eighth of the frame, so the flow covers only part of the frame and "
                    + "can't be turned into source pixels. Use a source such as 1280x720 or 1920x1080.")
        }

        var forward: CVPixelBuffer?
        var backward: CVPixelBuffer?
        CVPixelBufferCreate(nil, flowWidth, flowHeight, flowFormat, attributes as CFDictionary, &forward)
        CVPixelBufferCreate(nil, flowWidth, flowHeight, flowFormat, attributes as CFDictionary, &backward)
        guard let forward, let backward else {
            throw UpscaleError.io("Could not allocate the optical flow buffers.")
        }
        try processor.startSession(configuration: configuration)

        frameWidth = width
        frameHeight = height
        self.flowWidth = flowWidth
        self.flowHeight = flowHeight
        self.inputFormat = inputFormat
        forwardBuffer = forward
        backwardBuffer = backward
        scale = Float(width) / Float(flowWidth)
    }

    deinit {
        processor.endSession()
    }

    /// The flow between two consecutive frames. `sequential` says `previous` is the frame that came before `current`
    /// in the last call too (frames in presentation order, none skipped); pass false for the first pair and after any
    /// jump. Frames are converted to the format VideoToolbox wants when they aren't in it.
    ///
    /// `valid` is false when VideoToolbox returned a value that is not a finite half float (it does on flat frames,
    /// where there is nothing to track). Those values are written as 0, and the pair should be treated as untracked.
    func flow(
        previous: CVPixelBuffer, current: CVPixelBuffer, sequential: Bool
    ) throws -> (backward: [Float16], forward: [Float16], width: Int, height: Int, valid: Bool) {
        for frame in [previous, current] {
            guard CVPixelBufferGetWidth(frame) == frameWidth, CVPixelBufferGetHeight(frame) == frameHeight else {
                throw UpscaleError.io(
                    "Optical flow frame is \(CVPixelBufferGetWidth(frame))x\(CVPixelBufferGetHeight(frame)), expected \(frameWidth)x\(frameHeight).")
            }
        }
        let previousInput = try input(previous)
        let currentInput = try input(current)

        let time = submissions
        submissions += 1
        guard let previousFrame = VTFrameProcessorFrame(buffer: previousInput, presentationTimeStamp: CMTime(value: time, timescale: 1)),
              let currentFrame = VTFrameProcessorFrame(buffer: currentInput, presentationTimeStamp: CMTime(value: time + 1, timescale: 1)),
              let destination = VTFrameProcessorOpticalFlow(forwardFlow: forwardBuffer, backwardFlow: backwardBuffer),
              let parameters = VTOpticalFlowParameters(
                  sourceFrame: previousFrame, nextFrame: currentFrame,
                  submissionMode: sequential ? .sequential : .random, destinationOpticalFlow: destination)
        else {
            throw UpscaleError.scalerUnavailable("Could not prepare frames for optical flow.")
        }

        let done = DispatchSemaphore(value: 0)
        var failure: Error?
        processor.process(parameters: parameters, completionHandler: { _, error in
            failure = error
            done.signal()
        })
        done.wait()
        if let failure {
            throw UpscaleError.scalerUnavailable("Optical flow failed: \(failure.localizedDescription)")
        }

        let backward = try FlowGrid.planes(of: backwardBuffer, scale: scale)
        let forward = try FlowGrid.planes(of: forwardBuffer, scale: scale)
        return (
            backward: backward.planes, forward: forward.planes, width: flowWidth, height: flowHeight,
            valid: backward.finite && forward.finite)
    }

    /// `frame` as VideoToolbox wants it: itself when it already is, otherwise a converted copy.
    private func input(_ frame: CVPixelBuffer) throws -> CVPixelBuffer {
        if CVPixelBufferGetPixelFormatType(frame) == inputFormat, CVPixelBufferGetIOSurface(frame) != nil {
            return frame
        }
        if converter == nil {
            converter = try FrameConverter(format: inputFormat, width: frameWidth, height: frameHeight)
        }
        return try converter!.convert(frame)
    }

    private static func integer(_ value: Any?) -> Int? {
        (value as? NSNumber)?.intValue ?? (value as? [NSNumber])?.first?.intValue
    }
}

/// Copies frames into IOSurface-backed buffers of one pixel format.
private final class FrameConverter {
    private let pool: CVPixelBufferPool
    private let transfer: VTPixelTransferSession

    init(format: OSType, width: Int, height: Int) throws {
        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, PixelBuffers.attributes(format: format, width: width, height: height) as CFDictionary, &pool)
        var transfer: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &transfer)
        guard let pool, let transfer else {
            throw UpscaleError.scalerUnavailable("Could not set up buffers for optical flow.")
        }
        self.pool = pool
        self.transfer = transfer
    }

    deinit {
        VTPixelTransferSessionInvalidate(transfer)
    }

    func convert(_ frame: CVPixelBuffer) throws -> CVPixelBuffer {
        var converted: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &converted)
        guard let converted else { throw UpscaleError.io("Could not allocate a frame for optical flow.") }
        let status = VTPixelTransferSessionTransferImage(transfer, from: frame, to: converted)
        guard status == noErr else {
            throw UpscaleError.scalerUnavailable("Converting a frame for optical flow failed (\(status)).")
        }
        return converted
    }
}

/// Reading and resampling flow fields. Fields are 2×H×W half floats: all x displacements, then all y.
enum FlowGrid {
    /// The two-channel buffer formats `planes` can read.
    static let supportedFormats: Set<OSType> = [kCVPixelFormatType_TwoComponent16Half, kCVPixelFormatType_TwoComponent32Float]

    /// The planes of a two-component flow buffer with both multiplied by `scale`. `finite` is false when a value was
    /// not a finite half float (after scaling); such values come out as 0. On flat frames (all black, say) there is
    /// nothing to track and VideoToolbox returns noise, sometimes huge or NaN, which must not reach a training set.
    static func planes(of buffer: CVPixelBuffer, scale: Float) throws -> (planes: [Float16], finite: Bool) {
        let format = CVPixelBufferGetPixelFormatType(buffer)
        guard supportedFormats.contains(format) else {
            throw UpscaleError.unsupportedInput(
                "Optical flow buffers of format \(PixelBuffers.fourCC(format)) aren't supported; expected two-component half or float.")
        }
        guard CVPixelBufferLockBaseAddress(buffer, .readOnly) == kCVReturnSuccess else {
            throw UpscaleError.io("Could not read the optical flow buffer.")
        }
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let base = CVPixelBufferGetBaseAddress(buffer) else {
            throw UpscaleError.io("Could not read the optical flow buffer.")
        }

        let (width, height) = (CVPixelBufferGetWidth(buffer), CVPixelBufferGetHeight(buffer))
        let stride = CVPixelBufferGetBytesPerRow(buffer)
        let plane = width * height
        var planes = [Float16](repeating: 0, count: 2 * plane)
        var finite = true
        planes.withUnsafeMutableBufferPointer { out in
            for y in 0..<height {
                let row = base + y * stride
                let half = row.assumingMemoryBound(to: Float16.self)
                let float = row.assumingMemoryBound(to: Float.self)
                for x in 0..<width {
                    let (u, v): (Float, Float) =
                        format == kCVPixelFormatType_TwoComponent16Half
                        ? (Float(half[2 * x]), Float(half[2 * x + 1])) : (float[2 * x], float[2 * x + 1])
                    if let scaled = finiteHalf(u * scale) { out[y * width + x] = scaled } else { finite = false }
                    if let scaled = finiteHalf(v * scale) { out[plane + y * width + x] = scaled } else { finite = false }
                }
            }
        }
        return (planes, finite)
    }

    /// `value` as a half float, or nil when it is NaN, infinite or too large for one.
    private static func finiteHalf(_ value: Float) -> Float16? {
        abs(value) <= Float(Float16.greatestFiniteMagnitude) ? Float16(value) : nil // false for NaN too
    }

    /// Samples a flow field whose pixels are `scale` source pixels wide and tall (the same on both axes: grid pixel
    /// j covers source pixels j·scale..<(j+1)·scale, from the frame's top-left corner) at the centres of the pixels of
    /// `crop` (in source pixels), bilinearly with the edges clamped. This gives a 2×crop.height×crop.width field. The
    /// field's values are already in source pixels, so they are kept as they are. With scale 1 this is a plain crop.
    static func resample(
        _ field: [Float16], width: Int, height: Int, scale: Float, crop: CropRect
    ) -> [Float16] {
        /// For each of `count` output pixels starting at `origin`: the two field indices to blend and the weight of the second.
        func taps(origin: Int, count: Int, fieldSize: Int) -> [(low: Int, high: Int, weight: Float)] {
            (0..<count).map { index in
                let position = (Float(origin + index) + 0.5) / scale - 0.5
                let clamped = min(max(position, 0), Float(fieldSize - 1))
                let low = min(Int(clamped), fieldSize - 1)
                return (low, min(low + 1, fieldSize - 1), clamped - Float(low))
            }
        }
        let columns = taps(origin: crop.x, count: crop.width, fieldSize: width)
        let rows = taps(origin: crop.y, count: crop.height, fieldSize: height)

        let (fieldPlane, outPlane) = (width * height, crop.width * crop.height)
        var out = [Float16](repeating: 0, count: 2 * outPlane)
        field.withUnsafeBufferPointer { field in
            for channel in 0..<2 {
                for (y, row) in rows.enumerated() {
                    let (top, bottom) = (channel * fieldPlane + row.low * width, channel * fieldPlane + row.high * width)
                    for (x, column) in columns.enumerated() {
                        let upper = Float(field[top + column.low]) * (1 - column.weight) + Float(field[top + column.high]) * column.weight
                        let lower = Float(field[bottom + column.low]) * (1 - column.weight) + Float(field[bottom + column.high]) * column.weight
                        out[channel * outPlane + y * crop.width + x] = Float16(upper * (1 - row.weight) + lower * row.weight)
                    }
                }
            }
        }
        return out
    }
}

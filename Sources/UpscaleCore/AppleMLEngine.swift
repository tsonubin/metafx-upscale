import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

/// Apple's ML video super-resolution model (VideoToolbox `VTSuperResolutionScaler`) and nothing else: it turns a
/// decoded half-float RGBA frame into a 4× half-float RGBA frame. It keeps the previous source/output pair for
/// temporal stability, so frames must arrive in order. `AppleMLEngine` resizes and converts its output for the
/// encoder; the detail-layer dumps and the future `DetailEngine` use the 4× frame directly.
@available(macOS 26.0, *)
final class AppleSuperResolution {
    static let scaleFactor = 4

    /// Pixel buffer attributes the decoder should produce (the model's input: half-float RGBA).
    let sourceAttributes: [String: Any]

    private let processor = VTFrameProcessor()
    private let pool: CVPixelBufferPool
    private var previousSource: VTFrameProcessorFrame?
    private var previousOutput: VTFrameProcessorFrame?

    init(sourceWidth: Int, sourceHeight: Int) throws {
        let configuration = try AppleMLModel.configuration(width: sourceWidth, height: sourceHeight)
        guard configuration.configurationModelStatus == .ready else {
            throw UpscaleError.scalerUnavailable(
                "Apple's super-resolution model isn't installed. Run once with --download-model (macOS fetches it from Apple).")
        }
        try processor.startSession(configuration: configuration)

        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, configuration.destinationPixelBufferAttributes as CFDictionary, &pool)
        guard let pool else {
            processor.endSession()
            throw UpscaleError.scalerUnavailable("Could not set up buffers for ML super-resolution.")
        }

        self.pool = pool
        self.sourceAttributes = configuration.sourcePixelBufferAttributes
    }

    deinit {
        processor.endSession()
    }

    /// Upscales `source` 4×. The returned buffer stays valid at least until the next call, because it is kept as the
    /// previous output.
    func upscale(_ source: CVPixelBuffer, at time: CMTime) throws -> CVPixelBuffer {
        var upscaled: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &upscaled)
        guard let upscaled,
              let sourceFrame = VTFrameProcessorFrame(buffer: source, presentationTimeStamp: time),
              let outputFrame = VTFrameProcessorFrame(buffer: upscaled, presentationTimeStamp: time),
              let parameters = VTSuperResolutionScalerParameters(
                  sourceFrame: sourceFrame, previousFrame: previousSource, previousOutputFrame: previousOutput,
                  opticalFlow: nil, submissionMode: previousSource == nil ? .random : .sequential,
                  destinationFrame: outputFrame)
        else {
            throw UpscaleError.scalerUnavailable("Could not prepare a frame for ML super-resolution.")
        }

        let done = DispatchSemaphore(value: 0)
        var failure: Error?
        processor.process(parameters: parameters, completionHandler: { _, error in
            failure = error
            done.signal()
        })
        done.wait()
        if let failure {
            throw UpscaleError.scalerUnavailable("ML super-resolution failed at \(time.seconds)s: \(failure.localizedDescription)")
        }

        previousSource = sourceFrame
        previousOutput = outputFrame
        return upscaled
    }
}

/// The `ml` video engine: `AppleSuperResolution` (4×), then a resize to the requested size (e.g. 720p → 5120x2880 →
/// 3840x2160; beyond 4× the resize finishes the job). The resize writes 10-bit 4:2:0 so the model's half-float
/// precision isn't lost to 8-bit banding.
@available(macOS 26.0, *)
final class AppleMLEngine: VideoFrameEngine {
    let label = "Apple ML super-resolution (VideoToolbox, 4× then resized)"
    let decodeAttributes: [String: Any]
    let outputPixelFormat = PixelBuffers.tenBitYUV

    private let superResolution: AppleSuperResolution
    private let transfer: VTPixelTransferSession
    private let dumps: DumpSet?

    /// `color` holds the AVVideoColor*Key tags the encoder is given, so the RGB → YCbCr conversion matches them.
    /// `dumps`, when given, records each decoded frame and its 4× result alongside the encode.
    init(
        sourceWidth: Int, sourceHeight: Int, outputWidth: Int, outputHeight: Int, color: [String: String],
        dumps: DumpSet? = nil
    ) throws {
        let superResolution = try AppleSuperResolution(sourceWidth: sourceWidth, sourceHeight: sourceHeight)

        var transfer: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &transfer)
        guard let transfer else {
            throw UpscaleError.scalerUnavailable("Could not set up buffers for ML super-resolution.")
        }
        VTSessionSetProperty(transfer, key: kVTPixelTransferPropertyKey_DownsamplingMode, value: kVTDownsamplingMode_Average)
        for (key, tag) in [
            (kVTPixelTransferPropertyKey_DestinationColorPrimaries, AVVideoColorPrimariesKey),
            (kVTPixelTransferPropertyKey_DestinationTransferFunction, AVVideoTransferFunctionKey),
            (kVTPixelTransferPropertyKey_DestinationYCbCrMatrix, AVVideoYCbCrMatrixKey),
        ] {
            if let value = color[tag] { VTSessionSetProperty(transfer, key: key, value: value as CFString) }
        }

        self.superResolution = superResolution
        self.transfer = transfer
        self.dumps = dumps
        self.decodeAttributes = superResolution.sourceAttributes
    }

    deinit {
        VTPixelTransferSessionInvalidate(transfer)
    }

    func process(_ source: CVPixelBuffer, at time: CMTime, into destination: CVPixelBuffer) throws {
        let upscaled = try superResolution.upscale(source, at: time)
        let status = VTPixelTransferSessionTransferImage(transfer, from: upscaled, to: destination)
        guard status == noErr else {
            throw UpscaleError.scalerUnavailable("Resizing the ML output failed (\(status)).")
        }
        try dumps?.record(source: source, upscaled: upscaled, at: time)
    }
}

/// Availability and download of the on-device model behind `AppleMLEngine`.
@available(macOS 26.0, *)
public enum AppleMLModel {
    public static var isReady: Bool {
        (try? configuration(width: 1920, height: 1080).configurationModelStatus) == .ready
    }

    /// Asks macOS to download the model from Apple, reporting progress (0...1) about once a second.
    public static func download(onProgress: (Float) -> Void) throws {
        let configuration = try configuration(width: 1920, height: 1080)
        guard configuration.configurationModelStatus != .ready else { return }
        let done = DispatchSemaphore(value: 0)
        var failure: Error?
        configuration.downloadConfigurationModel { error in
            failure = error
            done.signal()
        }
        while done.wait(timeout: .now() + 1) == .timedOut {
            onProgress(configuration.configurationModelPercentageAvailable)
        }
        // The completion handler has reported VTFrameProcessorAssetDownloadFailed even when the model ended up
        // fully installed, so the model status is the source of truth.
        guard configuration.configurationModelStatus == .ready else {
            throw UpscaleError.io("Model download failed: \(failure?.localizedDescription ?? "unknown error")")
        }
    }

    static func configuration(width: Int, height: Int) throws -> VTSuperResolutionScalerConfiguration {
        guard VTSuperResolutionScalerConfiguration.isSupported else {
            throw UpscaleError.scalerUnavailable("Apple ML super-resolution isn't supported on this Mac (needs macOS 26+).")
        }
        guard let configuration = VTSuperResolutionScalerConfiguration(
            frameWidth: width, frameHeight: height, scaleFactor: AppleSuperResolution.scaleFactor, inputType: .video,
            // `.normal` is the only quality level VideoToolbox offers for super resolution (no `.quality` case).
            usePrecomputedFlow: false, qualityPrioritization: .normal,
            revision: VTSuperResolutionScalerConfiguration.defaultRevision)
        else {
            throw UpscaleError.scalerUnavailable("The ml engine accepts video up to 1920x1080; this one is \(width)x\(height).")
        }
        return configuration
    }
}

import AVFoundation
import CoreMedia
import CoreVideo
import Foundation
import VideoToolbox

/// Apple's ML video super-resolution (VideoToolbox `VTSuperResolutionScaler`).
///
/// The model only supports 4×, so each frame is upscaled 4× in half-float RGBA and then resized to the
/// requested size (e.g. 720p → 5120x2880 → 3840x2160; beyond 4× the resize finishes the job). The resize
/// writes 10-bit 4:2:0 so the model's half-float precision isn't lost to 8-bit banding. It uses the
/// previous frame for temporal stability, so frames must arrive in order.
@available(macOS 26.0, *)
final class AppleMLEngine: VideoFrameEngine {
    static let scaleFactor = 4

    let label = "Apple ML super-resolution (VideoToolbox, 4× then resized)"
    let decodeAttributes: [String: Any]
    let outputPixelFormat = PixelBuffers.tenBitYUV

    private let processor = VTFrameProcessor()
    private let pool: CVPixelBufferPool
    private let transfer: VTPixelTransferSession
    private var previousSource: VTFrameProcessorFrame?
    private var previousOutput: VTFrameProcessorFrame?

    /// `color` holds the AVVideoColor*Key tags the encoder is given, so the RGB → YCbCr conversion matches them.
    init(sourceWidth: Int, sourceHeight: Int, outputWidth: Int, outputHeight: Int, color: [String: String]) throws {
        let configuration = try AppleMLModel.configuration(width: sourceWidth, height: sourceHeight)
        guard configuration.configurationModelStatus == .ready else {
            throw UpscaleError.scalerUnavailable(
                "Apple's super-resolution model isn't installed. Run once with --download-model (macOS fetches it from Apple).")
        }
        try processor.startSession(configuration: configuration)

        var pool: CVPixelBufferPool?
        CVPixelBufferPoolCreate(nil, nil, configuration.destinationPixelBufferAttributes as CFDictionary, &pool)
        var transfer: VTPixelTransferSession?
        VTPixelTransferSessionCreate(allocator: nil, pixelTransferSessionOut: &transfer)
        guard let pool, let transfer else {
            processor.endSession()
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

        self.pool = pool
        self.transfer = transfer
        self.decodeAttributes = configuration.sourcePixelBufferAttributes
    }

    deinit {
        processor.endSession()
        VTPixelTransferSessionInvalidate(transfer)
    }

    func process(_ source: CVPixelBuffer, at time: CMTime, into destination: CVPixelBuffer) throws {
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

        let status = VTPixelTransferSessionTransferImage(transfer, from: upscaled, to: destination)
        guard status == noErr else {
            throw UpscaleError.scalerUnavailable("Resizing the ML output failed (\(status)).")
        }
        previousSource = sourceFrame
        previousOutput = outputFrame
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
            frameWidth: width, frameHeight: height, scaleFactor: AppleMLEngine.scaleFactor, inputType: .video,
            // `.normal` is the only quality level VideoToolbox offers for super resolution (no `.quality` case).
            usePrecomputedFlow: false, qualityPrioritization: .normal,
            revision: VTSuperResolutionScalerConfiguration.defaultRevision)
        else {
            throw UpscaleError.scalerUnavailable("The ml engine accepts video up to 1920x1080; this one is \(width)x\(height).")
        }
        return configuration
    }
}

import AVFoundation
import CoreMedia
import CoreVideo
import VideoToolbox

/// Decodes a video, upscales every frame (MetalFX or Apple ML), and re-encodes it as HEVC. Audio is copied untouched.
public enum VideoUpscaler {
    public struct Options {
        public var size: TargetSize = .uhd4K
        public var engine: Engine = .metalfx
        /// Average video bitrate in bits per second; `nil` picks one from the output resolution and frame rate.
        public var bitrate: Int?
        /// Optional excerpt, in seconds from the start of the input.
        public var start: Double?
        public var duration: Double?
        /// Polled once per frame; returning true stops the job with `CancellationError` and removes the partial file.
        public var isCancelled: (() -> Bool)?

        public init() {}
    }

    /// What `upscale` will do with a file, for showing it before starting.
    public struct SourceInfo {
        public let width, height: Int
        public let duration: Double
        public let fps: Double
        public let hasAudio: Bool
    }

    public static func inspect(_ url: URL) async throws -> SourceInfo {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw UpscaleError.noVideoTrack }
        let (size, rate) = try await track.load(.naturalSize, .nominalFrameRate)
        let duration = try await asset.load(.duration)
        let hasAudio = !(try await asset.loadTracks(withMediaType: .audio)).isEmpty
        return SourceInfo(
            width: Int(size.width), height: Int(size.height), duration: duration.seconds,
            fps: rate > 0 ? Double(rate) : 30, hasAudio: hasAudio)
    }

    public struct Progress {
        public let framesDone: Int
        public let framesTotal: Int
        public let elapsed: Double
    }

    public struct Summary {
        public let sourceWidth, sourceHeight, outputWidth, outputHeight: Int
        public let frames: Int
        public let bitrate: Int
        public let elapsed: Double
        public let engine: String
    }

    public static func upscale(
        input: URL, output: URL, options: Options, onProgress: @escaping (Progress) -> Void
    ) async throws -> Summary {
        let asset = AVURLAsset(url: input)
        guard let videoTrack = try await asset.loadTracks(withMediaType: .video).first else {
            throw UpscaleError.noVideoTrack
        }
        let audioTracks = try await asset.loadTracks(withMediaType: .audio)
        let (naturalSize, frameRate, transform, formats) = try await videoTrack.load(
            .naturalSize, .nominalFrameRate, .preferredTransform, .formatDescriptions)
        let assetDuration = try await asset.load(.duration)

        let (srcW, srcH) = (Int(naturalSize.width), Int(naturalSize.height))
        let (dstW, dstH) = options.size.resolve(sourceWidth: srcW, sourceHeight: srcH, evenDimensions: true)
        guard dstW >= srcW, dstH >= srcH, dstW * dstH > srcW * srcH else {
            throw UpscaleError.notAnUpscale(source: "\(srcW)x\(srcH)", target: "\(dstW)x\(dstH)")
        }
        let colorProperties = try colorProperties(of: formats.first)
        let fps = frameRate > 0 ? Double(frameRate) : 30
        let bitrate = options.bitrate ?? defaultBitrate(width: dstW, height: dstH, fps: fps)
        let engine: VideoFrameEngine
        switch options.engine {
        case .metalfx:
            engine = try FrameUpscaler(inputWidth: srcW, inputHeight: srcH, outputWidth: dstW, outputHeight: dstH)
        case .ml:
            guard #available(macOS 26.0, *) else {
                throw UpscaleError.scalerUnavailable("The ml engine needs macOS 26 or later.")
            }
            engine = try AppleMLEngine(
                sourceWidth: srcW, sourceHeight: srcH, outputWidth: dstW, outputHeight: dstH, color: colorProperties)
        }

        let start = CMTime(seconds: options.start ?? 0, preferredTimescale: 600)
        let end = options.duration.map { CMTimeAdd(start, CMTime(seconds: $0, preferredTimescale: 600)) } ?? assetDuration
        let range = CMTimeRange(start: start, end: CMTimeMinimum(end, assetDuration))

        // Reader: decode straight to the engine's input format; audio is read compressed for pass-through.
        let reader = try AVAssetReader(asset: asset)
        reader.timeRange = range
        let videoOutput = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: engine.decodeAttributes)
        videoOutput.alwaysCopiesSampleData = false
        reader.add(videoOutput)
        let audioOutputs = audioTracks.map { AVAssetReaderTrackOutput(track: $0, outputSettings: nil) }
        audioOutputs.forEach { reader.add($0) }

        // Writer: hardware HEVC at the target size, tagged with the source's colour space.
        let tenBit = engine.outputPixelFormat == PixelBuffers.tenBitYUV
        let writer = try AVAssetWriter(outputURL: output, fileType: output.pathExtension.lowercased() == "mov" ? .mov : .mp4)
        let videoInput = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.hevc,
            AVVideoWidthKey: dstW,
            AVVideoHeightKey: dstH,
            AVVideoColorPropertiesKey: colorProperties,
            AVVideoCompressionPropertiesKey: [
                AVVideoAverageBitRateKey: bitrate,
                AVVideoExpectedSourceFrameRateKey: fps,
                // Offline encode: spend the time on quality, not speed or power.
                kVTCompressionPropertyKey_PrioritizeEncodingSpeedOverQuality as String: false,
                kVTCompressionPropertyKey_RealTime as String: false,
                kVTCompressionPropertyKey_MaximizePowerEfficiency as String: false,
                AVVideoProfileLevelKey: (tenBit ? kVTProfileLevel_HEVC_Main10_AutoLevel : kVTProfileLevel_HEVC_Main_AutoLevel) as String,
            ],
        ])
        videoInput.transform = transform
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: videoInput,
            sourcePixelBufferAttributes: PixelBuffers.attributes(format: engine.outputPixelFormat, width: dstW, height: dstH))
        writer.add(videoInput)

        var audioPairs: [(AVAssetReaderTrackOutput, AVAssetWriterInput)] = []
        for (track, readerOutput) in zip(audioTracks, audioOutputs) {
            let hint = try await track.load(.formatDescriptions).first
            let writerInput = AVAssetWriterInput(mediaType: .audio, outputSettings: nil, sourceFormatHint: hint)
            guard writer.canAdd(writerInput) else {
                throw UpscaleError.unsupportedInput("This audio track can't be copied into a .\(output.pathExtension) file; try .mov.")
            }
            writer.add(writerInput)
            audioPairs.append((readerOutput, writerInput))
        }

        guard writer.startWriting() else { throw writer.error ?? UpscaleError.io("Could not start writing \(output.path).") }
        guard reader.startReading() else { throw reader.error ?? UpscaleError.io("Could not read \(input.path).") }
        writer.startSession(atSourceTime: range.start)

        let framesTotal = max(1, Int((range.duration.seconds * fps).rounded()))
        let clock = ContinuousClock.now
        let frames = FrameCounter()

        do {
            try await withThrowingTaskGroup(of: Void.self) { group in
                group.addTask {
                    try await drain(videoInput, writer: writer, label: "video") {
                        if options.isCancelled?() == true { throw CancellationError() }
                        guard let sample = videoOutput.copyNextSampleBuffer() else { return false }
                        guard let source = CMSampleBufferGetImageBuffer(sample) else { return true }
                        guard let pool = adaptor.pixelBufferPool else { throw UpscaleError.io("Writer has no pixel buffer pool.") }
                        var destination: CVPixelBuffer?
                        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &destination)
                        guard let destination else { throw UpscaleError.io("Could not allocate an output frame.") }

                        let time = CMSampleBufferGetPresentationTimeStamp(sample)
                        try engine.process(source, at: time, into: destination)
                        guard adaptor.append(destination, withPresentationTime: time) else {
                            throw writer.error ?? UpscaleError.io("Encoder rejected frame \(frames.value).")
                        }
                        frames.value += 1
                        onProgress(Progress(framesDone: frames.value, framesTotal: framesTotal, elapsed: clock.elapsedSeconds))
                        return true
                    }
                }
                for (index, (readerOutput, writerInput)) in audioPairs.enumerated() {
                    group.addTask {
                        try await drain(writerInput, writer: writer, label: "audio\(index)") {
                            guard let sample = readerOutput.copyNextSampleBuffer() else { return false }
                            guard writerInput.append(sample) else { throw writer.error ?? UpscaleError.io("Audio copy failed.") }
                            return true
                        }
                    }
                }
                try await group.waitForAll()
            }
            if reader.status == .failed { throw reader.error ?? UpscaleError.io("Decoding failed.") }
            await writer.finishWriting()
            guard writer.status == .completed else { throw writer.error ?? UpscaleError.io("Encoding failed.") }
        } catch {
            reader.cancelReading()
            writer.cancelWriting()
            throw error
        }

        return Summary(
            sourceWidth: srcW, sourceHeight: srcH, outputWidth: dstW, outputHeight: dstH,
            frames: frames.value, bitrate: bitrate, elapsed: clock.elapsedSeconds, engine: engine.label)
    }

    /// Feeds `input` from `next` (which returns false when its source is exhausted) until done.
    private static func drain(
        _ input: AVAssetWriterInput, writer: AVAssetWriter, label: String, next: @escaping () throws -> Bool
    ) async throws {
        let queue = DispatchQueue(label: "metalfx-upscale.\(label)")
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            var finished = false
            input.requestMediaDataWhenReady(on: queue) {
                while !finished && input.isReadyForMoreMediaData {
                    do {
                        if try next() { continue }
                        finished = true
                        input.markAsFinished()
                        continuation.resume()
                    } catch {
                        finished = true
                        if writer.status == .writing { input.markAsFinished() }
                        continuation.resume(throwing: error)
                    }
                }
            }
        }
    }

    /// ~0.07 bits per pixel per frame: about 14.5 Mbps for 4K at 25 fps.
    static func defaultBitrate(width: Int, height: Int, fps: Double) -> Int {
        Int(Double(width * height) * fps * 0.07)
    }

    /// The source's colour tags, defaulting to BT.709 (what untagged HD video almost always is).
    /// HDR transfer functions are rejected because this pipeline is 8-bit SDR.
    private static func colorProperties(of format: CMFormatDescription?) throws -> [String: String] {
        func tag(_ key: CFString) -> String? {
            format.flatMap { CMFormatDescriptionGetExtension($0, extensionKey: key) as? String }
        }
        let transfer = tag(kCMFormatDescriptionExtension_TransferFunction) ?? AVVideoTransferFunction_ITU_R_709_2
        if transfer == (kCVImageBufferTransferFunction_SMPTE_ST_2084_PQ as String)
            || transfer == (kCVImageBufferTransferFunction_ITU_R_2100_HLG as String) {
            throw UpscaleError.unsupportedInput("HDR (PQ/HLG) video isn't supported by this 8-bit SDR pipeline.")
        }
        return [
            AVVideoColorPrimariesKey: tag(kCMFormatDescriptionExtension_ColorPrimaries) ?? AVVideoColorPrimaries_ITU_R_709_2,
            AVVideoTransferFunctionKey: transfer,
            AVVideoYCbCrMatrixKey: tag(kCMFormatDescriptionExtension_YCbCrMatrix) ?? AVVideoYCbCrMatrix_ITU_R_709_2,
        ]
    }
}

/// Only the video drain queue mutates this, so it needs no locking.
private final class FrameCounter {
    var value = 0
}

extension ContinuousClock.Instant {
    var elapsedSeconds: Double {
        let d = ContinuousClock.now - self
        return Double(d.components.seconds) + Double(d.components.attoseconds) / 1e18
    }
}

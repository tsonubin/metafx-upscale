import AVFoundation
import CoreVideo
import Foundation
import Testing
@testable import UpscaleCore

private let modelReady: Bool = {
    if #available(macOS 26.0, *) { return AppleMLModel.isReady }
    return false
}()

/// Runs a whole (tiny) `ml` job with dumps, so it needs Apple's model; it is skipped where the model isn't installed.
struct DumpJobTests {
    /// A 25 fps H.264 clip of a moving gradient.
    private func makeClip(at url: URL, width: Int, height: Int, frames: Int) throws {
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
            AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height,
        ])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        writer.add(input)
        #expect(writer.startWriting())
        writer.startSession(atSourceTime: .zero)
        for index in 0..<frames {
            while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.005) }
            var buffer: CVPixelBuffer?
            CVPixelBufferPoolCreatePixelBuffer(nil, try #require(adaptor.pixelBufferPool), &buffer)
            let frame = try #require(buffer)
            CVPixelBufferLockBaseAddress(frame, [])
            let base = try #require(CVPixelBufferGetBaseAddress(frame)).assumingMemoryBound(to: UInt8.self)
            let stride = CVPixelBufferGetBytesPerRow(frame)
            for y in 0..<height {
                for x in 0..<width {
                    let pixel = base + y * stride + 4 * x
                    pixel[0] = UInt8((x + 3 * index) % 256)
                    pixel[1] = UInt8((2 * y + index) % 256)
                    pixel[2] = UInt8((x + y) % 256)
                    pixel[3] = 255
                }
            }
            CVPixelBufferUnlockBaseAddress(frame, [])
            #expect(adaptor.append(frame, withPresentationTime: CMTime(value: Int64(index), timescale: 25)))
        }
        input.markAsFinished()
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
        #expect(writer.status == .completed)
    }

    private func readMeta(_ dir: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: dir.appendingPathComponent("meta.json"))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private final class Polls {
        var count = 0
    }

    private func withTempDirectory(_ body: (URL) async throws -> Void) async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("DumpJobTests-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try await body(dir)
    }

    private func options(_ dir: URL) -> VideoUpscaler.Options {
        var options = VideoUpscaler.Options()
        options.size = .exact(width: 1280, height: 720)
        options.engine = .ml
        options.dumps = DumpOptions(
            source: dir.appendingPathComponent("s"), base: dir.appendingPathComponent("b"),
            flow: dir.appendingPathComponent("f"),
            crop: CropRect(x: 100, y: 50, width: 64, height: 32))
        return options
    }

    @Test(.enabled(if: modelReady))
    func dumpsOneFramePerProcessedFrame() async throws {
        try await withTempDirectory { dir in
            let clip = dir.appendingPathComponent("clip.mp4")
            try makeClip(at: clip, width: 640, height: 360, frames: 6)
            _ = try await VideoUpscaler.upscale(
                input: clip, output: dir.appendingPathComponent("out.mp4"), options: options(dir), onProgress: { _ in })

            for (name, width, height, scale) in [("s", 64, 32, 1), ("b", 128, 64, 2), ("f", 64, 32, 1)] {
                let meta = try readMeta(dir.appendingPathComponent(name))
                #expect(meta["frames"] as? Int == 6)
                #expect(meta["width"] as? Int == width && meta["height"] as? Int == height)
                #expect(meta["scale"] as? Int == scale)
                #expect(meta["crop"] as? [Int] == [100, 50, 64, 32])
                #expect(meta["source_size"] as? [Int] == [640, 360])
                let pts = try #require(meta["pts"] as? [Double])
                #expect(pts.count == 6 && abs(pts[1] - 0.04) < 1e-6)
            }
            let size = try Data(contentsOf: dir.appendingPathComponent("b/000005.f16")).count
            #expect(size == 3 * 128 * 64 * 2)
            let flowMeta = try readMeta(dir.appendingPathComponent("f"))
            #expect(flowMeta["kind"] as? String == "flow" && flowMeta["channels"] as? Int == 4)
            #expect(flowMeta["valid"] as? [Bool] == [false, true, true, true, true, true])
            #expect(try Data(contentsOf: dir.appendingPathComponent("f/000005.f16")).count == 4 * 64 * 32 * 2)
        }
    }

    @Test(.enabled(if: modelReady))
    func cancelledJobStillWritesMeta() async throws {
        try await withTempDirectory { dir in
            let clip = dir.appendingPathComponent("clip.mp4")
            try makeClip(at: clip, width: 640, height: 360, frames: 12)
            var options = options(dir)
            let polls = Polls()
            options.isCancelled = {
                polls.count += 1
                return polls.count > 4 // let 4 frames through, then cancel
            }
            await #expect(throws: CancellationError.self) {
                _ = try await VideoUpscaler.upscale(
                    input: clip, output: dir.appendingPathComponent("out.mp4"), options: options, onProgress: { _ in })
            }

            for name in ["s", "b", "f"] {
                let meta = try readMeta(dir.appendingPathComponent(name))
                #expect(meta["frames"] as? Int == 4)
                #expect((meta["valid"] as? [Bool])?.count == 4)
            }
        }
    }
}

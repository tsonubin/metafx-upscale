import CoreMedia
import CoreVideo
import Foundation
import Testing
@testable import UpscaleCore

struct DumpSetTests {
    private static let half = kCVPixelFormatType_64RGBAHalf

    /// A 64RGBAHalf buffer whose pixel (x, y) has the channels returned by `value`.
    private func makeBuffer(
        width: Int, height: Int, format: OSType = half, value: (Int, Int) -> (Float16, Float16, Float16, Float16)
    ) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, format, PixelBuffers.attributes(format: format) as CFDictionary, &buffer)
        let created = try #require(buffer)
        guard format == Self.half else { return created }
        CVPixelBufferLockBaseAddress(created, [])
        defer { CVPixelBufferUnlockBaseAddress(created, []) }
        let stride = CVPixelBufferGetBytesPerRow(created)
        let base = try #require(CVPixelBufferGetBaseAddress(created))
        for y in 0..<height {
            let row = (base + y * stride).assumingMemoryBound(to: Float16.self)
            for x in 0..<width {
                let (r, g, b, a) = value(x, y)
                row[4 * x] = r
                row[4 * x + 1] = g
                row[4 * x + 2] = b
                row[4 * x + 3] = a
            }
        }
        return created
    }

    private func withTempDirectory(_ body: (URL) throws -> Void) throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("DumpSetTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try body(dir)
    }

    private func readMeta(_ dir: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: dir.appendingPathComponent("meta.json"))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    private func readHalves(_ url: URL) throws -> [Float16] {
        let data = try Data(contentsOf: url)
        return (0..<data.count / 2).map { Float16(bitPattern: UInt16(data[2 * $0]) | UInt16(data[2 * $0 + 1]) << 8) }
    }

    @Test func extractsPlanarRGBFromCrop() throws {
        // Pixel (x, y): R = x, G = y, B = x + y, A = 99 (must be dropped).
        let buffer = try makeBuffer(width: 8, height: 6) { x, y in (Float16(x), Float16(y), Float16(x + y), 99) }
        let planes = try DumpSet.rgbPlanes(of: buffer, crop: CropRect(x: 2, y: 1, width: 3, height: 2))
        #expect(planes.count == 3 * 3 * 2)
        let xs: [Float16] = [2, 3, 4, 2, 3, 4]
        let ys: [Float16] = [1, 1, 1, 2, 2, 2]
        #expect(Array(planes[0..<6]) == xs)
        #expect(Array(planes[6..<12]) == ys)
        #expect(Array(planes[12..<18]) == zip(xs, ys).map { $0 + $1 })
    }

    @Test func rejectsOtherPixelFormatsAndOutOfRangeCrops() throws {
        let bgra = try makeBuffer(width: 8, height: 6, format: kCVPixelFormatType_32BGRA) { _, _ in (0, 0, 0, 0) }
        #expect(throws: UpscaleError.self) {
            try DumpSet.rgbPlanes(of: bgra, crop: CropRect(x: 0, y: 0, width: 8, height: 6))
        }
        let half = try makeBuffer(width: 8, height: 6) { _, _ in (0, 0, 0, 1) }
        #expect(throws: UpscaleError.self) {
            try DumpSet.rgbPlanes(of: half, crop: CropRect(x: 6, y: 0, width: 3, height: 6))
        }
    }

    @Test func recordsSourceAndBaseWithScaledCrop() throws {
        try withTempDirectory { root in
            let (sourceDir, baseDir) = (root.appendingPathComponent("s"), root.appendingPathComponent("b"))
            let crop = CropRect(x: 2, y: 1, width: 4, height: 2)
            let dumps = try DumpSet(
                options: DumpOptions(source: sourceDir, base: baseDir, crop: crop),
                sourceWidth: 8, sourceHeight: 6, outputWidth: 24, outputHeight: 18)
            #expect(dumps.scale == 3)

            // The "4x" frame is 32x24: uniform colour, so the resize to 24x18 keeps it.
            let source = try makeBuffer(width: 8, height: 6) { x, y in (Float16(x) / 8, Float16(y) / 6, 0.5, 1) }
            let upscaled = try makeBuffer(width: 32, height: 24) { _, _ in (0.25, 0.5, 0.75, 1) }
            try dumps.record(source: source, upscaled: upscaled, at: CMTime(seconds: 0.5, preferredTimescale: 600))
            try dumps.record(source: source, upscaled: upscaled, at: CMTime(seconds: 0.54, preferredTimescale: 600))
            try dumps.finish()

            let sourceMeta = try readMeta(sourceDir)
            #expect(sourceMeta["kind"] as? String == "source")
            #expect(sourceMeta["frames"] as? Int == 2)
            #expect(sourceMeta["width"] as? Int == 4 && sourceMeta["height"] as? Int == 2)
            #expect(sourceMeta["crop"] as? [Int] == [2, 1, 4, 2])
            #expect(sourceMeta["scale"] as? Int == 1)
            #expect(sourceMeta["source_size"] as? [Int] == [8, 6])
            let pts = try #require(sourceMeta["pts"] as? [Double])
            #expect(abs(pts[0] - 0.5) < 1e-9 && abs(pts[1] - 0.54) < 1e-9)

            let baseMeta = try readMeta(baseDir)
            #expect(baseMeta["kind"] as? String == "base")
            #expect(baseMeta["frames"] as? Int == 2)
            #expect(baseMeta["channels"] as? Int == 3)
            #expect(baseMeta["width"] as? Int == 12 && baseMeta["height"] as? Int == 6)
            #expect(baseMeta["crop"] as? [Int] == [2, 1, 4, 2])
            #expect(baseMeta["scale"] as? Int == 3)
            #expect(baseMeta["source_size"] as? [Int] == [8, 6])

            let base = try readHalves(baseDir.appendingPathComponent("000001.f16"))
            #expect(base.count == 3 * 12 * 6)
            for (channel, expected) in [Float16(0.25), 0.5, 0.75].enumerated() {
                let plane = base[channel * 72..<(channel + 1) * 72]
                #expect(plane.allSatisfy { abs(Float($0) - Float(expected)) < 0.01 })
            }
        }
    }

    @Test func defaultCropIsTheFullFrame() throws {
        try withTempDirectory { root in
            let sourceDir = root.appendingPathComponent("s")
            let dumps = try DumpSet(
                options: DumpOptions(source: sourceDir), sourceWidth: 8, sourceHeight: 6, outputWidth: 16, outputHeight: 12)
            #expect(dumps.crop == CropRect(x: 0, y: 0, width: 8, height: 6))
            let source = try makeBuffer(width: 8, height: 6) { _, _ in (0.5, 0.5, 0.5, 1) }
            let upscaled = try makeBuffer(width: 32, height: 24) { _, _ in (0, 0, 0, 1) }
            try dumps.record(source: source, upscaled: upscaled, at: .zero)
            try dumps.finish()
            let meta = try readMeta(sourceDir)
            #expect(meta["frames"] as? Int == 1)
            #expect(meta["width"] as? Int == 8 && meta["height"] as? Int == 6)
        }
    }

    @Test func aRunThatRecordedNoFramesLeavesNoMeta() throws {
        // Something failing before the first frame must not leave a meta.json, which marks a dump as finished.
        try withTempDirectory { root in
            let dirs = ["s", "b"].map { root.appendingPathComponent($0) }
            let dumps = try DumpSet(
                options: DumpOptions(source: dirs[0], base: dirs[1]),
                sourceWidth: 8, sourceHeight: 6, outputWidth: 16, outputHeight: 12)
            try dumps.finish()
            for dir in dirs {
                #expect(FileManager.default.fileExists(atPath: dir.path))
                #expect(!FileManager.default.fileExists(atPath: dir.appendingPathComponent("meta.json").path))
            }
        }
    }

    @Test func finishAfterAFailedFrameKeepsTheEarlierOnes() throws {
        try withTempDirectory { root in
            let sourceDir = root.appendingPathComponent("s")
            let dumps = try DumpSet(
                options: DumpOptions(source: sourceDir), sourceWidth: 8, sourceHeight: 6, outputWidth: 16, outputHeight: 12)
            let good = try makeBuffer(width: 8, height: 6) { _, _ in (0.5, 0.5, 0.5, 1) }
            let bad = try makeBuffer(width: 8, height: 6, format: kCVPixelFormatType_32BGRA) { _, _ in (0, 0, 0, 0) }
            let upscaled = try makeBuffer(width: 32, height: 24) { _, _ in (0, 0, 0, 1) }
            try dumps.record(source: good, upscaled: upscaled, at: .zero)
            #expect(throws: UpscaleError.self) { try dumps.record(source: bad, upscaled: upscaled, at: .zero) }
            try dumps.finish()
            try dumps.finish() // calling it again is harmless
            #expect(try readMeta(sourceDir)["frames"] as? Int == 1)
        }
    }

    @Test(.enabled(if: opticalFlowSupported))
    func recordsFlowOnTheSourceCrop() throws {
        // Content moves 4 px right per frame. The flow is cropped to the source window and is in source pixels.
        let (width, height) = flowTestSize
        let first = try NoiseFrames.pair(width: width, height: height, dx: 4, dy: 0)
        let third = try NoiseFrames.pair(width: width, height: height, dx: 8, dy: 0).current
        let frames = [first.previous, first.current, third]
        try withTempDirectory { root in
            let (sourceDir, flowDir) = (root.appendingPathComponent("s"), root.appendingPathComponent("f"))
            let crop = CropRect(x: 100, y: 50, width: 64, height: 32)
            let dumps = try DumpSet(
                options: DumpOptions(source: sourceDir, flow: flowDir, crop: crop),
                sourceWidth: width, sourceHeight: height, outputWidth: 2 * width, outputHeight: 2 * height)
            for (index, frame) in frames.enumerated() {
                try dumps.record(source: frame, upscaled: frame, at: CMTime(value: Int64(index), timescale: 25))
            }
            try dumps.finish()

            let meta = try readMeta(flowDir)
            #expect(meta["kind"] as? String == "flow")
            #expect(meta["frames"] as? Int == 3)
            #expect(meta["channels"] as? Int == 4)
            #expect(meta["width"] as? Int == 64 && meta["height"] as? Int == 32)
            #expect(meta["crop"] as? [Int] == [100, 50, 64, 32])
            #expect(meta["scale"] as? Int == 1)
            #expect(meta["source_size"] as? [Int] == [width, height])
            #expect(meta["valid"] as? [Bool] == [false, true, true])
            #expect(meta["flow_channels"] as? [String] == ["back_x", "back_y", "fwd_x", "fwd_y"])
            #expect(meta["flow_units"] as? String == "source_px")
            let pts = try #require(meta["pts"] as? [Double])
            #expect(pts.count == 3 && abs(pts[2] - 0.08) < 1e-9)
            #expect(try readMeta(sourceDir)["frames"] as? Int == 3)

            // Frame 0 has no previous frame: zeros.
            let plane = 64 * 32
            let zeros = try readHalves(flowDir.appendingPathComponent("000000.f16"))
            #expect(zeros.count == 4 * plane && zeros.allSatisfy { $0 == 0 })
            for index in 1...2 {
                let flow = try readHalves(flowDir.appendingPathComponent(String(format: "%06d.f16", index)))
                #expect(flow.count == 4 * plane)
                let backX = median(flow[0..<plane]), backY = median(flow[plane..<2 * plane])
                let forwardX = median(flow[2 * plane..<3 * plane]), forwardY = median(flow[3 * plane..<4 * plane])
                #expect(backX > -4.5 && backX < -3.5, "frame \(index) back x was \(backX)")
                #expect(forwardX > 3.5 && forwardX < 4.5, "frame \(index) forward x was \(forwardX)")
                #expect(abs(backY) < 0.5 && abs(forwardY) < 0.5, "frame \(index) y was \(backY), \(forwardY)")
            }
        }
    }

    @Test(.enabled(if: opticalFlowSupported))
    func flowForTheFullFrameHasTheSourceGrid() throws {
        let (width, height) = flowTestSize
        let frames = try NoiseFrames.pair(width: width, height: height, dx: 4, dy: 0)
        try withTempDirectory { root in
            let flowDir = root.appendingPathComponent("f")
            let dumps = try DumpSet(
                options: DumpOptions(flow: flowDir),
                sourceWidth: width, sourceHeight: height, outputWidth: 2 * width, outputHeight: 2 * height)
            try dumps.record(source: frames.previous, upscaled: frames.previous, at: .zero)
            try dumps.record(source: frames.current, upscaled: frames.current, at: CMTime(value: 1, timescale: 25))
            try dumps.finish()
            let meta = try readMeta(flowDir)
            #expect(meta["width"] as? Int == width && meta["height"] as? Int == height)
            #expect(meta["crop"] as? [Int] == [0, 0, width, height])
            #expect(try readHalves(flowDir.appendingPathComponent("000001.f16")).count == 4 * width * height)
        }
    }

    @Test(.enabled(if: opticalFlowSupported))
    func flowDumpIsRefusedForSizesWhoseFlowCannotBeTurnedIntoSourcePixels() throws {
        try withTempDirectory { root in
            let dirs = ["s", "f"].map { root.appendingPathComponent($0) }
            #expect(throws: UpscaleError.self) {
                try DumpSet(
                    options: DumpOptions(source: dirs[0], flow: dirs[1]),
                    sourceWidth: 256, sourceHeight: 144, outputWidth: 512, outputHeight: 288)
            }
            // The estimator is set up before any directory is created.
            #expect(dirs.allSatisfy { !FileManager.default.fileExists(atPath: $0.path) })
        }
    }
}

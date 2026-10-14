import CoreMedia
import CoreVideo
import Foundation
import Testing
import VideoToolbox
@testable import UpscaleCore

/// Whether this Mac has VideoToolbox optical flow; the tests that run it are skipped otherwise.
let opticalFlowSupported: Bool = {
    if #available(macOS 26.0, *) { return VTOpticalFlowConfiguration.isSupported }
    return false
}()

/// Deterministic noise, so a failing run can be repeated.
private struct SplitMix64: RandomNumberGenerator {
    var state: UInt64

    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

/// Two frames of blurred random noise where the second is the first with its content moved by (dx, dy) pixels
/// (content moving right/down is positive). Noise has texture at every scale, which optical flow needs.
enum NoiseFrames {
    private static let pad = 16

    static func pair(
        width: Int, height: Int, dx: Int, dy: Int, format: OSType = kCVPixelFormatType_64RGBAHalf
    ) throws -> (previous: CVPixelBuffer, current: CVPixelBuffer) {
        precondition(abs(dx) <= pad && abs(dy) <= pad)
        let (fieldWidth, fieldHeight) = (width + 2 * pad, height + 2 * pad)
        var random = SplitMix64(state: 42)
        var field = (0..<fieldWidth * fieldHeight).map { _ in Float.random(in: 0...1, using: &random) }
        for _ in 0..<3 { // three 3x3 box blurs, as a horizontal and a vertical 3-tap pass each
            var across = field
            for y in 0..<fieldHeight {
                for x in 0..<fieldWidth {
                    let (from, to) = (max(0, x - 1), min(fieldWidth - 1, x + 1))
                    var sum: Float = 0
                    for nx in from...to { sum += field[y * fieldWidth + nx] }
                    across[y * fieldWidth + x] = sum / Float(to - from + 1)
                }
            }
            var down = across
            for y in 0..<fieldHeight {
                let (from, to) = (max(0, y - 1), min(fieldHeight - 1, y + 1))
                for x in 0..<fieldWidth {
                    var sum: Float = 0
                    for ny in from...to { sum += across[ny * fieldWidth + x] }
                    down[y * fieldWidth + x] = sum / Float(to - from + 1)
                }
            }
            field = down
        }
        let (low, high) = (field.min() ?? 0, field.max() ?? 1)
        func level(_ x: Int, _ y: Int) -> Float { (field[y * fieldWidth + x] - low) / (high - low) }

        let previous = try makeFrame(width: width, height: height, format: format) { x, y in level(x + pad, y + pad) }
        let current = try makeFrame(width: width, height: height, format: format) { x, y in
            level(x + pad - dx, y + pad - dy)
        }
        return (previous, current)
    }

    /// A gray frame (64RGBAHalf or 32BGRA, IOSurface-backed) whose pixel (x, y) has brightness `level(x, y)` in 0...1.
    static func makeFrame(
        width: Int, height: Int, format: OSType, level: (Int, Int) -> Float
    ) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, format, PixelBuffers.attributes(format: format) as CFDictionary, &buffer)
        let frame = try #require(buffer)
        CVPixelBufferLockBaseAddress(frame, [])
        defer { CVPixelBufferUnlockBaseAddress(frame, []) }
        let stride = CVPixelBufferGetBytesPerRow(frame)
        let base = try #require(CVPixelBufferGetBaseAddress(frame))
        for y in 0..<height {
            if format == kCVPixelFormatType_64RGBAHalf {
                let row = (base + y * stride).assumingMemoryBound(to: Float16.self)
                for x in 0..<width {
                    let value = Float16(level(x, y))
                    (row[4 * x], row[4 * x + 1], row[4 * x + 2], row[4 * x + 3]) = (value, value, value, 1)
                }
            } else {
                precondition(format == kCVPixelFormatType_32BGRA)
                let row = (base + y * stride).assumingMemoryBound(to: UInt8.self)
                for x in 0..<width {
                    let value = UInt8((level(x, y) * 255).rounded())
                    (row[4 * x], row[4 * x + 1], row[4 * x + 2], row[4 * x + 3]) = (value, value, value, 255)
                }
            }
        }
        return frame
    }
}

func median(_ values: ArraySlice<Float16>) -> Float {
    let sorted = values.map(Float.init).sorted()
    return sorted.isEmpty ? .nan : sorted[sorted.count / 2]
}

/// A source size whose flow grid covers the whole frame at a whole-number scale (80x45 for this one, a quarter of the
/// frame). The smallest 16:9 size at or above 256x144 that does: VideoToolbox accepts 256x144 too, but returns a padded
/// 80x60 grid there (see `rejectsSizesWhoseGridDoesNotCoverTheFrame`).
let flowTestSize = (width: 320, height: 180)

struct OpticalFlowTests {
    @available(macOS 26.0, *)
    private func estimate(dx: Int, dy: Int) throws -> (backward: [Float16], forward: [Float16], width: Int, height: Int, valid: Bool) {
        let estimator = try OpticalFlowEstimator(width: flowTestSize.width, height: flowTestSize.height)
        let frames = try NoiseFrames.pair(width: flowTestSize.width, height: flowTestSize.height, dx: dx, dy: dy)
        return try estimator.flow(previous: frames.previous, current: frames.current, sequential: false)
    }

    /// Medians of the x and y planes of a 2xHxW flow.
    private func medians(_ planes: [Float16]) -> (x: Float, y: Float) {
        let half = planes.count / 2
        return (median(planes[0..<half]), median(planes[half...]))
    }

    @Test(.enabled(if: opticalFlowSupported))
    func detectsShift() throws {
        guard #available(macOS 26.0, *) else { return }
        // Content moves +4 px in x from the previous frame to the current one.
        let result = try estimate(dx: 4, dy: 0)
        #expect(result.backward.count == 2 * result.width * result.height)
        #expect(result.forward.count == result.backward.count)
        let (backward, forward) = (medians(result.backward), medians(result.forward))
        #expect(backward.x > -4.5 && backward.x < -3.5, "backward x was \(backward.x)")
        #expect(forward.x > 3.5 && forward.x < 4.5, "forward x was \(forward.x)")
        #expect(abs(backward.y) < 0.5, "backward y was \(backward.y)")
        #expect(abs(forward.y) < 0.5, "forward y was \(forward.y)")
    }

    @Test(.enabled(if: opticalFlowSupported))
    func detectsVerticalShift() throws {
        guard #available(macOS 26.0, *) else { return }
        // Down is positive y, and the y plane is in source pixels too.
        let result = try estimate(dx: 0, dy: 6)
        let (backward, forward) = (medians(result.backward), medians(result.forward))
        #expect(backward.y > -6.5 && backward.y < -5.5, "backward y was \(backward.y)")
        #expect(forward.y > 5.5 && forward.y < 6.5, "forward y was \(forward.y)")
        #expect(abs(backward.x) < 0.5, "backward x was \(backward.x)")
        #expect(abs(forward.x) < 0.5, "forward x was \(forward.x)")
    }

    @Test(.enabled(if: opticalFlowSupported))
    func convertsFramesInOtherPixelFormats() throws {
        guard #available(macOS 26.0, *) else { return }
        let estimator = try OpticalFlowEstimator(width: flowTestSize.width, height: flowTestSize.height)
        let frames = try NoiseFrames.pair(
            width: flowTestSize.width, height: flowTestSize.height, dx: 4, dy: 0, format: kCVPixelFormatType_32BGRA)
        let result = try estimator.flow(previous: frames.previous, current: frames.current, sequential: false)
        let forward = medians(result.forward)
        #expect(forward.x > 3.5 && forward.x < 4.5, "forward x was \(forward.x)")
    }

    @Test(.enabled(if: opticalFlowSupported))
    func sequentialSubmissionsKeepWorking() throws {
        guard #available(macOS 26.0, *) else { return }
        // Three frames moving +3 px each: both pairs, the second one submitted as a continuation, see +3.
        let (width, height) = flowTestSize
        let first = try NoiseFrames.pair(width: width, height: height, dx: 3, dy: 0)
        let third = try NoiseFrames.pair(width: width, height: height, dx: 6, dy: 0).current
        let estimator = try OpticalFlowEstimator(width: width, height: height)
        let one = try estimator.flow(previous: first.previous, current: first.current, sequential: false)
        let two = try estimator.flow(previous: first.current, current: third, sequential: true)
        for result in [one, two] {
            let forward = medians(result.forward)
            #expect(forward.x > 2.5 && forward.x < 3.5, "forward x was \(forward.x)")
        }
    }

    /// The values of `plane` (a flat W×H plane) in the `blockWidth`×`blockHeight` block whose top-left is (x, y).
    private func block(_ plane: ArraySlice<Float16>, width: Int, x: Int, y: Int, blockWidth: Int, blockHeight: Int) -> ArraySlice<Float16> {
        let rows = (y..<y + blockHeight).map { row in plane.dropFirst(row * width + x).prefix(blockWidth) }
        return ArraySlice(rows.flatMap { $0 })
    }

    @Test(.enabled(if: opticalFlowSupported))
    func flowIsRightInAllFourCornersToo() throws {
        guard #available(macOS 26.0, *) else { return }
        // The whole field must carry the shift, not just its median: check blocks at each corner of the 80x45 grid.
        let result = try estimate(dx: 8, dy: 6)
        let (width, height) = (result.width, result.height)
        let plane = width * height
        for (name, field, expectedX, expectedY) in [("backward", result.backward, Float(-8), Float(-6)), ("forward", result.forward, 8, 6)] {
            for (corner, x, y) in [("top-left", 0, 0), ("top-right", width - 10, 0), ("bottom-left", 0, height - 6), ("bottom-right", width - 10, height - 6)] {
                let xs = block(field[0..<plane], width: width, x: x, y: y, blockWidth: 10, blockHeight: 6)
                let ys = block(field[plane...], width: width, x: x, y: y, blockWidth: 10, blockHeight: 6)
                #expect(abs(median(xs) - expectedX) < 1, "\(name) \(corner) x median was \(median(xs))")
                #expect(abs(median(ys) - expectedY) < 1, "\(name) \(corner) y median was \(median(ys))")
                let worst = max(xs.map { abs(Float($0) - expectedX) }.max() ?? 0, ys.map { abs(Float($0) - expectedY) }.max() ?? 0)
                #expect(worst < 3, "\(name) \(corner): a value was \(worst) px from the shift")
            }
        }
    }

    @Test(.enabled(if: opticalFlowSupported))
    func detectsShiftAtFullHD() throws {
        guard #available(macOS 26.0, *) else { return }
        // 1920x1080 gets a 240x135 grid, an eighth of the frame: a 16 px move is 2 grid pixels.
        let estimator = try OpticalFlowEstimator(width: 1920, height: 1080)
        #expect(estimator.flowWidth == 240 && estimator.flowHeight == 135)
        let frames = try NoiseFrames.pair(width: 1920, height: 1080, dx: 16, dy: 8)
        let result = try estimator.flow(previous: frames.previous, current: frames.current, sequential: false)
        let (width, height) = (result.width, result.height)
        let plane = width * height
        #expect(result.valid)
        for (name, field, expectedX, expectedY) in [("backward", result.backward, Float(-16), Float(-8)), ("forward", result.forward, 16, 8)] {
            #expect(abs(median(field[0..<plane]) - expectedX) < 1.5, "\(name) x median was \(median(field[0..<plane]))")
            #expect(abs(median(field[plane...]) - expectedY) < 1.5, "\(name) y median was \(median(field[plane...]))")
            // The bottom-right corner of the frame.
            let xs = block(field[0..<plane], width: width, x: width - 16, y: height - 9, blockWidth: 16, blockHeight: 9)
            let ys = block(field[plane...], width: width, x: width - 16, y: height - 9, blockWidth: 16, blockHeight: 9)
            #expect(abs(median(xs) - expectedX) < 1.5, "\(name) bottom-right x median was \(median(xs))")
            #expect(abs(median(ys) - expectedY) < 1.5, "\(name) bottom-right y median was \(median(ys))")
        }
    }

    @Test(.enabled(if: opticalFlowSupported))
    func rejectsFramesOfTheWrongSize() throws {
        guard #available(macOS 26.0, *) else { return }
        let estimator = try OpticalFlowEstimator(width: flowTestSize.width, height: flowTestSize.height)
        let frames = try NoiseFrames.pair(width: 160, height: 90, dx: 1, dy: 0)
        #expect(throws: UpscaleError.self) {
            try estimator.flow(previous: frames.previous, current: frames.current, sequential: false)
        }
    }

    @Test(.enabled(if: opticalFlowSupported), arguments: [
        (256, 144), (854, 480), (720, 480), (1280, 544), (1024, 576), (1366, 768), (2560, 1440),
    ])
    func rejectsSizesWhoseGridDoesNotCoverTheFrame(width: Int, height: Int) throws {
        guard #available(macOS 26.0, *) else { return }
        // VideoToolbox pads the flow grid for these (256x144 gets 80x60, 854x480 and 720x480 get 240x135 and 180x135),
        // and the flow covers only the top-left part of it. At 854x480 the right 11% and bottom 11% of the frame
        // are garbage, and a per-axis rescale of 256x144 reads 3 px for a 4 px move on y. They are refused instead of
        // dumping wrong flow.
        #expect(throws: UpscaleError.self) { try OpticalFlowEstimator(width: width, height: height) }
    }

    @Test(.enabled(if: opticalFlowSupported), arguments: [
        (320, 180), (640, 360), (960, 540), (1280, 720), (1920, 1080), (480, 270), (1440, 1080), (3840, 2160),
    ])
    func acceptsSizesWhoseGridCoversTheFrame(width: Int, height: Int) throws {
        guard #available(macOS 26.0, *) else { return }
        // The grid is a quarter of the frame (an eighth at 1080p and 4K), rounded up; 480x270 gets 120x67.
        let estimator = try OpticalFlowEstimator(width: width, height: height)
        let ratio = width / estimator.flowWidth
        #expect(ratio == 4 || ratio == 8)
        #expect(abs(estimator.flowHeight - (height + ratio - 1) / ratio) <= 1)
    }

    @Test(.enabled(if: opticalFlowSupported))
    func rejectsSizesVideoToolboxCannotDo() {
        guard #available(macOS 26.0, *) else { return }
        #expect(throws: UpscaleError.self) { try OpticalFlowEstimator(width: 100_000, height: 100_000) }
    }
}

struct FlowGridTests {
    private func twoComponentBuffer(
        width: Int, height: Int, format: OSType, value: (Int, Int) -> (Float, Float)
    ) throws -> CVPixelBuffer {
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, format, PixelBuffers.attributes(format: format) as CFDictionary, &buffer)
        let created = try #require(buffer)
        CVPixelBufferLockBaseAddress(created, [])
        defer { CVPixelBufferUnlockBaseAddress(created, []) }
        let stride = CVPixelBufferGetBytesPerRow(created)
        let base = try #require(CVPixelBufferGetBaseAddress(created))
        for y in 0..<height {
            for x in 0..<width {
                let (u, v) = value(x, y)
                if format == kCVPixelFormatType_TwoComponent16Half {
                    let row = (base + y * stride).assumingMemoryBound(to: Float16.self)
                    (row[2 * x], row[2 * x + 1]) = (Float16(u), Float16(v))
                } else {
                    let row = (base + y * stride).assumingMemoryBound(to: Float.self)
                    (row[2 * x], row[2 * x + 1]) = (u, v)
                }
            }
        }
        return created
    }

    @Test(arguments: [kCVPixelFormatType_TwoComponent16Half, kCVPixelFormatType_TwoComponent32Float])
    func readsPlanesAndScalesBothAxes(format: OSType) throws {
        // Pixel (x, y): u = x, v = 10 + y.
        let buffer = try twoComponentBuffer(width: 4, height: 3, format: format) { x, y in (Float(x), Float(10 + y)) }
        let (planes, finite) = try FlowGrid.planes(of: buffer, scale: 4)
        try #require(planes.count == 2 * 4 * 3)
        #expect(finite)
        #expect(Array(planes[0..<12]) == (0..<3).flatMap { _ in (0..<4).map { Float16($0 * 4) } })
        #expect(Array(planes[12..<24]) == (0..<3).flatMap { y in (0..<4).map { _ in Float16((10 + y) * 4) } })
    }

    @Test func readerZeroesValuesThatAreNotFiniteHalfFloats() throws {
        let values: [Float] = [.nan, .infinity, -.infinity, 1e6, -70000, 3]
        let buffer = try twoComponentBuffer(width: 6, height: 1, format: kCVPixelFormatType_TwoComponent32Float) { x, _ in
            (values[x], values[5 - x])
        }
        let (planes, finite) = try FlowGrid.planes(of: buffer, scale: 1)
        #expect(planes == [0, 0, 0, 0, 0, 3, /* y */ 3, 0, 0, 0, 0, 0])
        #expect(!finite) // the pair is reported as untracked
        // The scale can push a finite value out of range too.
        let scaled = try FlowGrid.planes(of: buffer, scale: 100)
        #expect(scaled.planes[5] == 300 && !scaled.finite)
        let calm = try twoComponentBuffer(width: 2, height: 1, format: kCVPixelFormatType_TwoComponent16Half) { x, _ in (Float(x), -2) }
        #expect(try FlowGrid.planes(of: calm, scale: 1000).finite)
    }

    @Test func readerRejectsOtherFormats() throws {
        let bgra = try NoiseFrames.makeFrame(width: 4, height: 3, format: kCVPixelFormatType_32BGRA) { _, _ in 0 }
        #expect(throws: UpscaleError.self) { try FlowGrid.planes(of: bgra, scale: 1) }
    }

    @Test func resampleIsTheCropWhenTheGridIsOneToOne() {
        // A 6x4 field with u = x + 10 * y, v = -x: cropping x 1..<4, y 1..<3 picks those pixels exactly.
        let (width, height) = (6, 4)
        let field: [Float16] =
            (0..<height).flatMap { y in (0..<width).map { x in Float16(x + 10 * y) } }
            + (0..<height).flatMap { _ in (0..<width).map { x in Float16(-x) } }
        let crop = CropRect(x: 1, y: 1, width: 3, height: 2)
        let out = FlowGrid.resample(field, width: width, height: height, scale: 1, crop: crop)
        #expect(out == [11, 12, 13, 21, 22, 23, -1, -2, -3, -1, -2, -3])
    }

    @Test func resampleUsesTheSameScaleOnBothAxes() throws {
        // A 2-wide, 3-tall grid at 4 source px per grid px: u = 0, v = the grid row. Source row 4 has its centre at grid
        // row (4 + 0.5) / 4 - 0.5 = 0.625; row 11 is past the last grid row's centre (2) and clamps to it.
        let field: [Float16] = [0, 0, 0, 0, 0, 0, /* v */ 0, 0, 1, 1, 2, 2]
        let out = FlowGrid.resample(field, width: 2, height: 3, scale: 4, crop: CropRect(x: 0, y: 4, width: 1, height: 1))
        #expect(out.map(Float.init) == [0, 0.625])
        let last = FlowGrid.resample(field, width: 2, height: 3, scale: 4, crop: CropRect(x: 7, y: 11, width: 1, height: 1))
        #expect(last.map(Float.init) == [0, 2])
    }

    @Test func resampleInterpolatesALowerResolutionGridAtSourcePixelCentres() throws {
        // A 2x2 grid over a 4x4 source: grid pixel (i, j) is centred on source pixel (2i + 0.5, 2j + 0.5).
        // u = 10 * i, v = j. Source column 0 and 3 lie outside the outer grid centres (clamped); 1 and 2 are interpolated.
        let field: [Float16] = [0, 10, 0, 10, /* v */ 0, 0, 1, 1]
        let out = FlowGrid.resample(
            field, width: 2, height: 2, scale: 2, crop: CropRect(x: 0, y: 0, width: 4, height: 1))
        try #require(out.count == 8)
        // Grid x for source column c is (c + 0.5) / 2 - 0.5 = -0.25, 0.25, 0.75, 1.25 -> clamped 0, 0.25, 0.75, 1.
        #expect(out[0..<4].map(Float.init) == [0, 2.5, 7.5, 10])
        // Row 0 sits at grid y = -0.25, clamped to 0, where v = 0.
        #expect(out[4..<8].map(Float.init) == [0, 0, 0, 0])
    }
}

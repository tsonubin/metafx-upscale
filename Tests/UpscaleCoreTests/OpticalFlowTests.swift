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
        for _ in 0..<3 { // three 3x3 box blurs
            var blurred = field
            for y in 0..<fieldHeight {
                for x in 0..<fieldWidth {
                    var sum: Float = 0
                    var count: Float = 0
                    for ny in max(0, y - 1)...min(fieldHeight - 1, y + 1) {
                        for nx in max(0, x - 1)...min(fieldWidth - 1, x + 1) {
                            sum += field[ny * fieldWidth + nx]
                            count += 1
                        }
                    }
                    blurred[y * fieldWidth + x] = sum / count
                }
            }
            field = blurred
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

/// A source size that VideoToolbox gives a flow buffer for with a whole-frame, same-scale-on-both-axes layout (80x45
/// for this one, a quarter of the frame). The smallest 16:9 size at or above 256x144 that does: VideoToolbox accepts
/// 256x144 too, but returns a padded 80x60 buffer there (see `rejectsPaddedFlowBuffers`).
let flowTestSize = (width: 320, height: 180)

struct OpticalFlowTests {
    @available(macOS 26.0, *)
    private func estimate(dx: Int, dy: Int) throws -> (backward: [Float16], forward: [Float16], width: Int, height: Int) {
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

    @Test(.enabled(if: opticalFlowSupported))
    func rejectsFramesOfTheWrongSize() throws {
        guard #available(macOS 26.0, *) else { return }
        let estimator = try OpticalFlowEstimator(width: flowTestSize.width, height: flowTestSize.height)
        let frames = try NoiseFrames.pair(width: 160, height: 90, dx: 1, dy: 0)
        #expect(throws: UpscaleError.self) {
            try estimator.flow(previous: frames.previous, current: frames.current, sequential: false)
        }
    }

    @Test(.enabled(if: opticalFlowSupported))
    func rejectsPaddedFlowBuffers() throws {
        guard #available(macOS 26.0, *) else { return }
        // For 256x144 VideoToolbox returns an 80x60 flow buffer: the flow sits in its top-left 64x36 and the values
        // are scaled by 256/80 on both axes (a 4 px move reads 1.25, not 1.667 on y), so a per-axis rescale from the
        // buffer size would be wrong. Frame sizes like that are refused instead of dumping wrong flow.
        #expect(throws: UpscaleError.self) { try OpticalFlowEstimator(width: 256, height: 144) }
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
    func readsPlanesAndScalesPerAxis(format: OSType) throws {
        // Pixel (x, y): u = x, v = 10 + y.
        let buffer = try twoComponentBuffer(width: 4, height: 3, format: format) { x, y in (Float(x), Float(10 + y)) }
        let planes = try FlowGrid.planes(of: buffer, scaleX: 4, scaleY: 2)
        try #require(planes.count == 2 * 4 * 3)
        #expect(Array(planes[0..<12]) == (0..<3).flatMap { _ in (0..<4).map { Float16($0 * 4) } })
        #expect(Array(planes[12..<24]) == (0..<3).flatMap { y in (0..<4).map { _ in Float16((10 + y) * 2) } })
    }

    @Test func readerZeroesValuesThatAreNotFiniteHalfFloats() throws {
        let values: [Float] = [.nan, .infinity, -.infinity, 1e6, -70000, 3]
        let buffer = try twoComponentBuffer(width: 6, height: 1, format: kCVPixelFormatType_TwoComponent32Float) { x, _ in
            (values[x], values[5 - x])
        }
        let planes = try FlowGrid.planes(of: buffer, scaleX: 1, scaleY: 1)
        #expect(planes == [0, 0, 0, 0, 0, 3, /* y */ 3, 0, 0, 0, 0, 0])
        // The scale can push a finite value out of range too.
        #expect(try FlowGrid.planes(of: buffer, scaleX: 100, scaleY: 1)[5] == 300)
    }

    @Test func readerRejectsOtherFormats() throws {
        let bgra = try NoiseFrames.makeFrame(width: 4, height: 3, format: kCVPixelFormatType_32BGRA) { _, _ in 0 }
        #expect(throws: UpscaleError.self) { try FlowGrid.planes(of: bgra, scaleX: 1, scaleY: 1) }
    }

    @Test func resampleIsTheCropWhenTheGridMatchesTheSource() {
        // A 6x4 field with u = x + 10 * y, v = -x: cropping x 1..<4, y 1..<3 picks those pixels exactly.
        let (width, height) = (6, 4)
        let field: [Float16] =
            (0..<height).flatMap { y in (0..<width).map { x in Float16(x + 10 * y) } }
            + (0..<height).flatMap { _ in (0..<width).map { x in Float16(-x) } }
        let crop = CropRect(x: 1, y: 1, width: 3, height: 2)
        let out = FlowGrid.resample(field, width: width, height: height, sourceWidth: 6, sourceHeight: 4, crop: crop)
        #expect(out == [11, 12, 13, 21, 22, 23, -1, -2, -3, -1, -2, -3])
    }

    @Test func resampleInterpolatesALowerResolutionGridAtSourcePixelCentres() throws {
        // A 2x2 grid over a 4x4 source: grid pixel (i, j) is centred on source pixel (2i + 0.5, 2j + 0.5).
        // u = 10 * i, v = j. Source column 0 and 3 lie outside the outer grid centres (clamped); 1 and 2 are interpolated.
        let field: [Float16] = [0, 10, 0, 10, /* v */ 0, 0, 1, 1]
        let out = FlowGrid.resample(
            field, width: 2, height: 2, sourceWidth: 4, sourceHeight: 4, crop: CropRect(x: 0, y: 0, width: 4, height: 1))
        try #require(out.count == 8)
        // Grid x for source column c is (c + 0.5) / 2 - 0.5 = -0.25, 0.25, 0.75, 1.25 -> clamped 0, 0.25, 0.75, 1.
        #expect(out[0..<4].map(Float.init) == [0, 2.5, 7.5, 10])
        // Row 0 sits at grid y = -0.25, clamped to 0, where v = 0.
        #expect(out[4..<8].map(Float.init) == [0, 0, 0, 0])
    }
}

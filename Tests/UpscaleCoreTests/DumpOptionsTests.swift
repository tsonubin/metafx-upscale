import Foundation
import Testing
@testable import UpscaleCore

struct DumpOptionsTests {
    @Test func scaleIsInteger() throws {
        #expect(try baseScale(sourceWidth: 1280, sourceHeight: 720, outputWidth: 3840, outputHeight: 2160) == 3)
        #expect(try baseScale(sourceWidth: 1920, sourceHeight: 1080, outputWidth: 3840, outputHeight: 2160) == 2)
        #expect(try baseScale(sourceWidth: 640, sourceHeight: 360, outputWidth: 1920, outputHeight: 1080) == 3)
        #expect(throws: UpscaleError.self) {
            try baseScale(sourceWidth: 640, sourceHeight: 480, outputWidth: 2880, outputHeight: 2160)
        }
    }

    @Test func scaleNeedsTheSameRatioOnBothAxes() {
        #expect(throws: UpscaleError.self) {
            try baseScale(sourceWidth: 1280, sourceHeight: 720, outputWidth: 3840, outputHeight: 1440)
        }
        #expect(throws: UpscaleError.self) {
            try baseScale(sourceWidth: 1280, sourceHeight: 720, outputWidth: 640, outputHeight: 360)
        }
    }

    private func options(engine: Engine, crop: CropRect? = nil) -> VideoUpscaler.Options {
        var options = VideoUpscaler.Options()
        options.engine = engine
        options.dumps = DumpOptions(source: URL(fileURLWithPath: "/tmp/never-written"), crop: crop)
        return options
    }

    @Test func dumpsRequireMLEngine() {
        #expect(throws: UpscaleError.self) {
            try VideoUpscaler.validate(
                options(engine: .metalfx), sourceWidth: 1280, sourceHeight: 720, outputWidth: 3840, outputHeight: 2160)
        }
    }

    @Test func dumpsRequireIntegerScale() {
        #expect(throws: UpscaleError.self) {
            try VideoUpscaler.validate(
                options(engine: .ml), sourceWidth: 640, sourceHeight: 480, outputWidth: 2880, outputHeight: 2160)
        }
    }

    @Test func cropMustFitSource() {
        let crop = CropRect(x: 1000, y: 600, width: 320, height: 180)
        #expect(throws: UpscaleError.self) {
            try VideoUpscaler.validate(
                options(engine: .ml, crop: crop), sourceWidth: 1280, sourceHeight: 720, outputWidth: 3840, outputHeight: 2160)
        }
    }

    @Test func acceptsValidDumpsAndNoDumps() throws {
        let crop = CropRect(x: 480, y: 270, width: 320, height: 180)
        try VideoUpscaler.validate(
            options(engine: .ml, crop: crop), sourceWidth: 1280, sourceHeight: 720, outputWidth: 3840, outputHeight: 2160)
        // Without dumps neither the engine nor the scale is constrained.
        var plain = VideoUpscaler.Options()
        plain.engine = .metalfx
        try VideoUpscaler.validate(plain, sourceWidth: 640, sourceHeight: 480, outputWidth: 2880, outputHeight: 2160)
    }
}

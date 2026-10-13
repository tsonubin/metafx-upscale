import Testing
@testable import UpscaleCore

struct TargetSizeTests {
    @Test func presetFitsLandscapeSource() throws {
        let size = try TargetSize.parse("4k").resolve(sourceWidth: 1280, sourceHeight: 720, evenDimensions: true)
        #expect(size.width == 3840 && size.height == 2160)
    }

    @Test func presetFollowsPortraitOrientation() throws {
        let size = try TargetSize.parse("4K").resolve(sourceWidth: 1080, sourceHeight: 1920, evenDimensions: true)
        #expect(size.width == 2160 && size.height == 3840)
    }

    @Test func presetKeepsWidescreenAspect() throws {
        let size = try TargetSize.parse("4k").resolve(sourceWidth: 1920, sourceHeight: 800, evenDimensions: true)
        #expect(size.width == 3840 && size.height == 1600)
    }

    @Test func scaleFactor() throws {
        let size = try TargetSize.parse("1.5x").resolve(sourceWidth: 1280, sourceHeight: 720, evenDimensions: true)
        #expect(size.width == 1920 && size.height == 1080)
    }

    @Test func exactSizeRoundsToEvenForVideo() throws {
        let size = try TargetSize.parse("2001x1001").resolve(sourceWidth: 1280, sourceHeight: 720, evenDimensions: true)
        #expect(size.width == 2000 && size.height == 1000)
    }

    @Test func exactSizeKeptForImages() throws {
        let size = try TargetSize.parse("2001x1001").resolve(sourceWidth: 1280, sourceHeight: 720, evenDimensions: false)
        #expect(size.width == 2001 && size.height == 1001)
    }

    @Test(arguments: ["", "foo", "0x100", "-2x", "x100"])
    func rejectsInvalidSizes(_ text: String) {
        #expect(throws: UpscaleError.self) { try TargetSize.parse(text) }
    }
}

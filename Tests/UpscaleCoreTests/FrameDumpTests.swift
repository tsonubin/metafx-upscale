import Foundation
import Testing
@testable import UpscaleCore

struct FrameDumpTests {
    private func makeWriter(
        _ directory: URL, kind: String = "source", channels: Int = 3, width: Int = 4, height: Int = 2,
        extra: [String: Any] = [:]
    ) throws -> FrameDumpWriter {
        try FrameDumpWriter(
            directory: directory, kind: kind, channels: channels, width: width, height: height,
            sourceSize: (1280, 720), scale: 3, crop: CropRect(x: 10, y: 20, width: 320, height: 180), extra: extra)
    }

    private func withTempDirectory(_ body: (URL) throws -> Void) throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("FrameDumpTests-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: dir) }
        try body(dir)
    }

    private func readMeta(_ dir: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: dir.appendingPathComponent("meta.json"))
        return try #require(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    @Test func writesPlanarHalfFloats() throws {
        try withTempDirectory { dir in
            let writer = try makeWriter(dir)
            let first = (0..<24).map { Float16($0) * 0.5 }
            let second = (0..<24).map { Float16(100 - $0) }
            try writer.write(first, pts: 0)
            try writer.write(second, pts: 0.04)
            try writer.finish()

            for (name, expected) in [("000000.f16", first), ("000001.f16", second)] {
                let data = try Data(contentsOf: dir.appendingPathComponent(name))
                #expect(data.count == 48)
                let decoded = (0..<24).map { i in
                    Float16(bitPattern: UInt16(data[2 * i]) | UInt16(data[2 * i + 1]) << 8)
                }
                #expect(decoded == expected)
            }

            let meta = try readMeta(dir)
            #expect(meta["frames"] as? Int == 2)
            #expect(meta["pts"] as? [Double] == [0, 0.04])
            #expect(meta["valid"] as? [Bool] == [true, true])
            #expect(meta["kind"] as? String == "source")
            #expect(meta["channels"] as? Int == 3)
            #expect(meta["width"] as? Int == 4)
            #expect(meta["height"] as? Int == 2)
            #expect(meta["layout"] as? String == "CHW")
            #expect(meta["dtype"] as? String == "float16")
            #expect(meta["source_size"] as? [Int] == [1280, 720])
            #expect(meta["scale"] as? Int == 3)
            #expect(meta["crop"] as? [Int] == [10, 20, 320, 180])
        }
    }

    @Test func recordsValidFlagAndExtraKeys() throws {
        try withTempDirectory { dir in
            let extra: [String: Any] = ["flow_channels": ["back_x", "back_y", "fwd_x", "fwd_y"], "flow_units": "source_px"]
            let writer = try makeWriter(dir, kind: "flow", channels: 4, extra: extra)
            try writer.write([Float16](repeating: 1, count: 32), pts: 0, valid: false)
            try writer.write([Float16](repeating: 2, count: 32), pts: 0.04)
            try writer.finish()

            let meta = try readMeta(dir)
            #expect(meta["valid"] as? [Bool] == [false, true])
            #expect(meta["flow_channels"] as? [String] == ["back_x", "back_y", "fwd_x", "fwd_y"])
            #expect(meta["flow_units"] as? String == "source_px")
            #expect(meta["kind"] as? String == "flow")
        }
    }

    @Test func rejectsFrameOfWrongSize() throws {
        try withTempDirectory { dir in
            let writer = try makeWriter(dir)
            #expect(throws: UpscaleError.self) { try writer.write([Float16](repeating: 0, count: 23), pts: 0) }
        }
    }

    @Test func cropParses() throws {
        #expect(try CropRect.parse("10,20,320,180") == CropRect(x: 10, y: 20, width: 320, height: 180))
        #expect(throws: UpscaleError.self) { try CropRect.parse("1,2,3") }
        #expect(throws: UpscaleError.self) { try CropRect.parse("-1,0,10,10") }
    }

    @Test(arguments: ["", "a,b,c,d", "1,2,3,4,5", "0,0,0,10", "0,0,10,0"])
    func cropRejectsMalformedText(_ text: String) {
        #expect(throws: UpscaleError.self) { try CropRect.parse(text) }
    }
}

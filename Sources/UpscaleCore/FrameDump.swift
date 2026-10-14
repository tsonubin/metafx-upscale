import Foundation

/// A crop window in source pixels, before any scaling.
public struct CropRect: Equatable, Sendable {
    public var x: Int
    public var y: Int
    public var width: Int
    public var height: Int

    public init(x: Int, y: Int, width: Int, height: Int) {
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    }

    /// Parses "X,Y,W,H". The origin must be non-negative and the size positive.
    public static func parse(_ text: String) throws -> CropRect {
        let parts = text.split(separator: ",", omittingEmptySubsequences: false).map { Int($0) }
        guard parts.count == 4, let x = parts[0], let y = parts[1], let w = parts[2], let h = parts[3],
              x >= 0, y >= 0, w > 0, h > 0 else {
            throw UpscaleError.invalidCrop(text)
        }
        return CropRect(x: x, y: y, width: w, height: h)
    }
}

/// Writes frames as raw little-endian float16 planar CHW files (`000000.f16`, ...) plus `meta.json`.
/// The Python reader in `detailnet/dumps.py` reads the same layout.
public final class FrameDumpWriter {
    private let directory: URL
    private let channels: Int
    private let width: Int
    private let height: Int
    private let header: [String: Any]
    private var times: [Double] = []
    private var validFlags: [Bool] = []

    /// `extra` keys are added to meta.json; they never replace the standard keys.
    public init(
        directory: URL, kind: String, channels: Int, width: Int, height: Int,
        sourceSize: (Int, Int), scale: Int, crop: CropRect, extra: [String: Any] = [:]
    ) throws {
        guard channels > 0, width > 0, height > 0 else {
            throw UpscaleError.io("Dump frames need positive channels, width and height.")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.directory = directory
        self.channels = channels
        self.width = width
        self.height = height
        header = extra.merging([
            "kind": kind,
            "channels": channels,
            "width": width,
            "height": height,
            "layout": "CHW",
            "dtype": "float16",
            "source_size": [sourceSize.0, sourceSize.1],
            "scale": scale,
            "crop": [crop.x, crop.y, crop.width, crop.height],
        ]) { _, standard in standard }
    }

    public func write(_ chw: [Float16], pts: Double, valid: Bool = true) throws {
        guard chw.count == channels * width * height else {
            throw UpscaleError.io("Dump frame has \(chw.count) values; expected \(channels * width * height).")
        }
        let name = String(format: "%06d.f16", times.count)
        // Apple silicon is little-endian, so the in-memory bytes are the file bytes.
        try chw.withUnsafeBytes { try Data($0).write(to: directory.appendingPathComponent(name)) }
        times.append(pts)
        validFlags.append(valid)
    }

    /// Writes (or rewrites) meta.json for the frames written so far. With no frames it writes nothing: a dump
    /// that has a meta.json is a finished one, and a run that failed before its first frame must not look like that.
    public func finish() throws {
        guard !times.isEmpty else { return }
        var meta = header
        meta["frames"] = times.count
        meta["pts"] = times
        meta["valid"] = validFlags
        let data = try JSONSerialization.data(withJSONObject: meta, options: [.sortedKeys, .prettyPrinted])
        try data.write(to: directory.appendingPathComponent("meta.json"))
    }
}

import CoreMedia
import CoreVideo

/// Which upscaler the video pipeline uses.
public enum Engine: String, Sendable {
    /// MetalFX spatial scaler: fast, edge-adaptive, any output size.
    case metalfx
    /// Apple's ML super-resolution (VideoToolbox): slower, reconstructs texture, fixed 4× then resized.
    case ml
}

/// Turns one decoded frame into one upscaled frame for the video pipeline.
protocol VideoFrameEngine: AnyObject {
    /// Shown in the summary, e.g. "MetalFX spatial on Apple M5 Max".
    var label: String { get }
    /// Pixel buffer attributes the decoder should produce for this engine.
    var decodeAttributes: [String: Any] { get }
    /// Pixel format of the upscaled frames handed to the encoder; 10-bit formats are encoded as HEVC Main10.
    var outputPixelFormat: OSType { get }
    /// Upscales `source` into `destination` (`outputPixelFormat` at the output size). Frames arrive in presentation order.
    func process(_ source: CVPixelBuffer, at time: CMTime, into destination: CVPixelBuffer) throws
}

extension FrameUpscaler: VideoFrameEngine {
    var label: String { "MetalFX spatial on \(deviceName)" }
    var decodeAttributes: [String: Any] { PixelBuffers.bgraAttributes() }
    var outputPixelFormat: OSType { kCVPixelFormatType_32BGRA }

    func process(_ source: CVPixelBuffer, at time: CMTime, into destination: CVPixelBuffer) throws {
        try upscale(source, into: destination)
    }
}

enum PixelBuffers {
    static let tenBitYUV = kCVPixelFormatType_420YpCbCr10BiPlanarVideoRange

    /// Metal-compatible, IOSurface-backed 32BGRA, optionally at a fixed size.
    static func bgraAttributes(width: Int? = nil, height: Int? = nil) -> [String: Any] {
        attributes(format: kCVPixelFormatType_32BGRA, width: width, height: height)
    }

    /// Metal-compatible, IOSurface-backed buffers of `format`, optionally at a fixed size.
    static func attributes(format: OSType, width: Int? = nil, height: Int? = nil) -> [String: Any] {
        var attributes: [String: Any] = [
            kCVPixelBufferPixelFormatTypeKey as String: format,
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ]
        if let width, let height {
            attributes[kCVPixelBufferWidthKey as String] = width
            attributes[kCVPixelBufferHeightKey as String] = height
        }
        return attributes
    }
}

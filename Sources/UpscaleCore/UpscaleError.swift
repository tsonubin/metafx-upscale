public enum UpscaleError: Error, CustomStringConvertible {
    case invalidSize(String)
    case notAnUpscale(source: String, target: String)
    case metalUnavailable
    case scalerUnavailable(String)
    case noVideoTrack
    case unsupportedInput(String)
    case io(String)

    public var description: String {
        switch self {
        case let .invalidSize(text): "Invalid size '\(text)'. Use 4k, 1440p, 1080p, 8k, WIDTHxHEIGHT (e.g. 3840x2160) or a factor like 2x."
        case let .notAnUpscale(source, target): "Target \(target) is not larger than the source \(source); MetalFX only upscales."
        case .metalUnavailable: "No Metal GPU is available."
        case let .scalerUnavailable(reason): reason
        case .noVideoTrack: "The input has no video track."
        case let .unsupportedInput(reason): reason
        case let .io(reason): reason
        }
    }
}

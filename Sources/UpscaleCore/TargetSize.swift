/// The output dimensions requested on the command line.
public enum TargetSize: Equatable, Sendable {
    /// Fit inside a box whose long/short sides follow the source's orientation (e.g. 4k → 3840×2160 or 2160×3840).
    case box(long: Int, short: Int)
    /// Exact output dimensions.
    case exact(width: Int, height: Int)
    /// Multiply both source dimensions.
    case scale(Double)

    public static let uhd4K = TargetSize.box(long: 3840, short: 2160)

    public static func parse(_ text: String) throws -> TargetSize {
        let s = text.lowercased()
        switch s {
        case "4k", "2160p", "uhd": return .uhd4K
        case "8k", "4320p": return .box(long: 7680, short: 4320)
        case "1440p", "qhd": return .box(long: 2560, short: 1440)
        case "1080p", "fhd": return .box(long: 1920, short: 1080)
        default: break
        }
        if s.hasSuffix("x"), let factor = Double(s.dropLast()), factor > 0 {
            return .scale(factor)
        }
        let parts = s.split(separator: "x")
        if parts.count == 2, let w = Int(parts[0]), let h = Int(parts[1]), w > 0, h > 0 {
            return .exact(width: w, height: h)
        }
        throw UpscaleError.invalidSize(text)
    }

    /// Resolves to pixel dimensions for a source. Video needs even dimensions for 4:2:0 encoding.
    public func resolve(sourceWidth: Int, sourceHeight: Int, evenDimensions: Bool) -> (width: Int, height: Int) {
        let (sw, sh) = (Double(sourceWidth), Double(sourceHeight))
        let (w, h): (Double, Double)
        switch self {
        case let .exact(width, height):
            (w, h) = (Double(width), Double(height))
        case let .scale(factor):
            (w, h) = (sw * factor, sh * factor)
        case let .box(long, short):
            let (bw, bh) = sourceWidth >= sourceHeight ? (Double(long), Double(short)) : (Double(short), Double(long))
            let k = min(bw / sw, bh / sh)
            (w, h) = (sw * k, sh * k)
        }
        var (rw, rh) = (Int(w.rounded()), Int(h.rounded()))
        if evenDimensions {
            rw -= rw % 2
            rh -= rh % 2
        }
        return (rw, rh)
    }
}

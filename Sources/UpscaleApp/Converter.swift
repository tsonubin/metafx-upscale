import Foundation
import UpscaleCore

/// Converts files AVFoundation can't open (WebM, MKV, AV1, Opus…) into a near-lossless ProRes + AAC .mov with ffmpeg.
enum Converter {
    static var ffmpeg: URL? {
        ["/opt/homebrew/bin/ffmpeg", "/usr/local/bin/ffmpeg"]
            .map(URL.init(fileURLWithPath:))
            .first { FileManager.default.isExecutableFile(atPath: $0.path) }
    }

    static func toProRes(_ input: URL) async throws -> URL {
        guard let ffmpeg else {
            throw UpscaleError.unsupportedInput("This format needs ffmpeg to convert it first (brew install ffmpeg).")
        }
        let output = FileManager.default.temporaryDirectory
            .appending(path: "\(UUID().uuidString)-\(input.deletingPathExtension().lastPathComponent).mov")
        let process = Process()
        process.executableURL = ffmpeg
        process.arguments = [
            "-v", "error", "-y", "-i", input.path,
            "-c:v", "prores_ks", "-profile:v", "3", "-pix_fmt", "yuv422p10le",
            "-c:a", "aac", "-b:a", "256k", output.path,
        ]
        let errors = Pipe()
        process.standardError = errors
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            process.terminationHandler = { _ in continuation.resume() }
            do { try process.run() } catch { continuation.resume(throwing: error) }
        }
        guard process.terminationStatus == 0 else {
            let message = String(decoding: errors.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            try? FileManager.default.removeItem(at: output)
            throw UpscaleError.unsupportedInput("ffmpeg couldn't convert this file: \(message.prefix(200))")
        }
        return output
    }
}

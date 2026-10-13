import CoreGraphics
import CoreVideo
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// Upscales a still image with MetalFX, keeping its colour profile and orientation tag.
public enum ImageUpscaler {
    public struct Summary {
        public let sourceWidth, sourceHeight, outputWidth, outputHeight: Int
        public let gpuSeconds: Double
        public let device: String
    }

    public static func upscale(input: URL, output: URL, size: TargetSize) throws -> Summary {
        guard let source = CGImageSourceCreateWithURL(input as CFURL, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else {
            throw UpscaleError.unsupportedInput("Could not decode image \(input.path).")
        }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any] ?? [:]

        let (srcW, srcH) = (image.width, image.height)
        let (dstW, dstH) = size.resolve(sourceWidth: srcW, sourceHeight: srcH, evenDimensions: false)
        guard dstW >= srcW, dstH >= srcH, dstW * dstH > srcW * srcH else {
            throw UpscaleError.notAnUpscale(source: "\(srcW)x\(srcH)", target: "\(dstW)x\(dstH)")
        }

        let colorSpace = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        let hasAlpha = ![.none, .noneSkipFirst, .noneSkipLast].contains(image.alphaInfo)
        let alpha: CGImageAlphaInfo = hasAlpha ? .premultipliedFirst : .noneSkipFirst
        let bitmapInfo = CGBitmapInfo.byteOrder32Little.rawValue | alpha.rawValue

        let upscaler = try FrameUpscaler(inputWidth: srcW, inputHeight: srcH, outputWidth: dstW, outputHeight: dstH)
        let sourceBuffer = try makeBuffer(width: srcW, height: srcH)
        let outputBuffer = try makeBuffer(width: dstW, height: dstH)

        try withContext(sourceBuffer, colorSpace: colorSpace, bitmapInfo: bitmapInfo) { context in
            context.draw(image, in: CGRect(x: 0, y: 0, width: srcW, height: srcH))
        }
        let clock = ContinuousClock.now
        try upscaler.upscale(sourceBuffer, into: outputBuffer)
        let gpuSeconds = clock.elapsedSeconds
        let result = try withContext(outputBuffer, colorSpace: colorSpace, bitmapInfo: bitmapInfo) { $0.makeImage() }
        guard let result else { throw UpscaleError.io("Could not read back the upscaled image.") }

        let type = UTType(filenameExtension: output.pathExtension) ?? .png
        guard let destination = CGImageDestinationCreateWithURL(output as CFURL, type.identifier as CFString, 1, nil) else {
            throw UpscaleError.io("Can't write .\(output.pathExtension) images; use png, jpg, heic or tiff.")
        }
        var outputProperties: [CFString: Any] = [kCGImageDestinationLossyCompressionQuality: 0.95]
        outputProperties[kCGImagePropertyOrientation] = properties[kCGImagePropertyOrientation]
        CGImageDestinationAddImage(destination, result, outputProperties as CFDictionary)
        guard CGImageDestinationFinalize(destination) else { throw UpscaleError.io("Could not write \(output.path).") }

        return Summary(
            sourceWidth: srcW, sourceHeight: srcH, outputWidth: dstW, outputHeight: dstH,
            gpuSeconds: gpuSeconds, device: upscaler.deviceName)
    }

    private static func makeBuffer(width: Int, height: Int) throws -> CVPixelBuffer {
        let attributes: [String: Any] = [
            kCVPixelBufferMetalCompatibilityKey as String: true,
            kCVPixelBufferIOSurfacePropertiesKey as String: [String: Any](),
        ]
        var buffer: CVPixelBuffer?
        CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32BGRA, attributes as CFDictionary, &buffer)
        guard let buffer else { throw UpscaleError.io("Could not allocate a \(width)x\(height) image buffer.") }
        return buffer
    }

    private static func withContext<T>(
        _ buffer: CVPixelBuffer, colorSpace: CGColorSpace, bitmapInfo: UInt32, _ body: (CGContext) -> T
    ) throws -> T {
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: CVPixelBufferGetWidth(buffer), height: CVPixelBufferGetHeight(buffer),
            bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: colorSpace, bitmapInfo: bitmapInfo)
        else {
            throw UpscaleError.io("Could not create a drawing context for the image.")
        }
        return body(context)
    }
}

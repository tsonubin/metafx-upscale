import CoreVideo
import Foundation
import Metal
import MetalFX

/// Upscales BGRA pixel buffers with the MetalFX spatial scaler, keeping every frame on the GPU.
///
/// Both buffers must be 32BGRA, IOSurface-backed, and sized as configured. One instance handles one
/// input/output size pair; it is not thread-safe.
public final class FrameUpscaler {
    public let inputWidth: Int
    public let inputHeight: Int
    public let outputWidth: Int
    public let outputHeight: Int
    public let deviceName: String

    private let queue: MTLCommandQueue
    private let scaler: MTLFXSpatialScaler
    /// MetalFX only writes to private-storage textures, so the result is blitted into the destination buffer.
    private let scaledTexture: MTLTexture
    private let textureCache: CVMetalTextureCache

    public init(inputWidth: Int, inputHeight: Int, outputWidth: Int, outputHeight: Int) throws {
        guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
            throw UpscaleError.metalUnavailable
        }
        guard MTLFXSpatialScalerDescriptor.supportsDevice(device) else {
            throw UpscaleError.scalerUnavailable("MetalFX spatial scaling is not supported on \(device.name).")
        }

        let descriptor = MTLFXSpatialScalerDescriptor()
        descriptor.inputWidth = inputWidth
        descriptor.inputHeight = inputHeight
        descriptor.outputWidth = outputWidth
        descriptor.outputHeight = outputHeight
        descriptor.colorTextureFormat = .bgra8Unorm
        descriptor.outputTextureFormat = .bgra8Unorm
        // Decoded video and photos are gamma-encoded, which is what perceptual mode expects.
        descriptor.colorProcessingMode = .perceptual
        guard let scaler = descriptor.makeSpatialScaler(device: device) else {
            throw UpscaleError.scalerUnavailable(
                "MetalFX could not create a \(inputWidth)x\(inputHeight) → \(outputWidth)x\(outputHeight) scaler.")
        }

        let textureDescriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm, width: outputWidth, height: outputHeight, mipmapped: false)
        textureDescriptor.storageMode = .private
        textureDescriptor.usage = scaler.outputTextureUsage
        guard let scaledTexture = device.makeTexture(descriptor: textureDescriptor) else {
            throw UpscaleError.scalerUnavailable("Could not allocate a \(outputWidth)x\(outputHeight) GPU texture.")
        }

        var cache: CVMetalTextureCache?
        guard CVMetalTextureCacheCreate(nil, nil, device, nil, &cache) == kCVReturnSuccess, let cache else {
            throw UpscaleError.metalUnavailable
        }

        self.inputWidth = inputWidth
        self.inputHeight = inputHeight
        self.outputWidth = outputWidth
        self.outputHeight = outputHeight
        self.deviceName = device.name
        self.queue = queue
        self.scaler = scaler
        self.scaledTexture = scaledTexture
        self.textureCache = cache
    }

    public func upscale(_ source: CVPixelBuffer, into destination: CVPixelBuffer) throws {
        // The CVMetalTexture wrappers must outlive the GPU work, so they stay in scope until it completes.
        let input = try texture(for: source, width: inputWidth, height: inputHeight, usage: scaler.colorTextureUsage)
        let output = try texture(for: destination, width: outputWidth, height: outputHeight, usage: .shaderRead)
        guard let inputTexture = CVMetalTextureGetTexture(input),
              let outputTexture = CVMetalTextureGetTexture(output),
              let commandBuffer = queue.makeCommandBuffer()
        else {
            throw UpscaleError.scalerUnavailable("Could not wrap pixel buffers as Metal textures.")
        }

        scaler.colorTexture = inputTexture
        scaler.inputContentWidth = inputWidth
        scaler.inputContentHeight = inputHeight
        scaler.outputTexture = scaledTexture
        scaler.encode(commandBuffer: commandBuffer)

        guard let blit = commandBuffer.makeBlitCommandEncoder() else {
            throw UpscaleError.scalerUnavailable("Could not create a blit encoder.")
        }
        blit.copy(from: scaledTexture, to: outputTexture)
        blit.endEncoding()

        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        if let error = commandBuffer.error {
            throw UpscaleError.scalerUnavailable("GPU upscale failed: \(error.localizedDescription)")
        }
    }

    private func texture(for buffer: CVPixelBuffer, width: Int, height: Int, usage: MTLTextureUsage) throws -> CVMetalTexture {
        guard CVPixelBufferGetWidth(buffer) == width, CVPixelBufferGetHeight(buffer) == height else {
            throw UpscaleError.io(
                "Pixel buffer is \(CVPixelBufferGetWidth(buffer))x\(CVPixelBufferGetHeight(buffer)), expected \(width)x\(height).")
        }
        let attributes = [kCVMetalTextureUsage: NSNumber(value: usage.rawValue)] as CFDictionary
        var texture: CVMetalTexture?
        let status = CVMetalTextureCacheCreateTextureFromImage(
            nil, textureCache, buffer, attributes, .bgra8Unorm, width, height, 0, &texture)
        guard status == kCVReturnSuccess, let texture else {
            throw UpscaleError.io("CVMetalTextureCache failed (\(status)); is the buffer 32BGRA and IOSurface-backed?")
        }
        return texture
    }
}

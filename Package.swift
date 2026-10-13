// swift-tools-version:6.0
import PackageDescription

let package = Package(
    name: "metalfx-upscale",
    platforms: [.macOS(.v14)],
    products: [
        .executable(name: "metalfx-upscale", targets: ["metalfx-upscale"]),
        .executable(name: "Upscale", targets: ["UpscaleApp"]),
    ],
    targets: [
        .target(name: "UpscaleCore"),
        .executableTarget(name: "metalfx-upscale", dependencies: ["UpscaleCore"]),
        .executableTarget(name: "UpscaleApp", dependencies: ["UpscaleCore"]),
        .testTarget(name: "UpscaleCoreTests", dependencies: ["UpscaleCore"]),
    ],
    swiftLanguageModes: [.v5]
)

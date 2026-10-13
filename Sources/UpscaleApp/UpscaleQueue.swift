import AppKit
import Foundation
import Observation
import UniformTypeIdentifiers
import UpscaleCore

/// The app's single setting: the highest-quality output Apple's AI upscaler can produce.
enum Quality {
    static let size = TargetSize.uhd4K
    static let bitrate = 50_000_000
    static let summary = "Apple AI super-resolution · 4K · HEVC 10-bit · 50 Mbps · quality over speed"
}

@MainActor @Observable
final class UpscaleQueue {
    enum Status: Equatable {
        case waiting, converting, upscaling, done, failed(String)
    }

    enum ModelState: Equatable {
        case checking, ready, missing, downloading(Double), unsupported(String)
    }

    struct Item: Identifiable {
        let id = UUID()
        let url: URL
        var info: VideoUpscaler.SourceInfo?
        /// True once `info` was looked up; `info == nil` after that means AVFoundation can't read the file.
        var inspected = false
        var status = Status.waiting
        var progress = 0.0
        var fps = 0.0
        var secondsLeft: Double?
        var output: URL?

        var name: String { url.lastPathComponent }
    }

    var items: [Item] = []
    var isRunning = false
    var model = ModelState.checking
    var outputFolder: URL {
        didSet { UserDefaults.standard.set(outputFolder.path, forKey: "outputFolder") }
    }

    private var cancelFlag = CancelFlag()
    private var activity: NSObjectProtocol?

    init() {
        let saved = UserDefaults.standard.string(forKey: "outputFolder").map { URL(fileURLWithPath: $0) }
        outputFolder = saved ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: "Movies/Upscaled")
        checkModel()
    }

    // MARK: Queue

    /// Adds video files; folders are searched (one level) for videos.
    func add(_ urls: [URL]) {
        let files = urls.flatMap { url -> [URL] in
            guard url.hasDirectoryPath else { return [url] }
            let contents = (try? FileManager.default.contentsOfDirectory(
                at: url, includingPropertiesForKeys: nil, options: .skipsHiddenFiles)) ?? []
            return contents.sorted { $0.lastPathComponent < $1.lastPathComponent }
        }
        for url in files where Self.isVideo(url) && !items.contains(where: { $0.url == url }) {
            let item = Item(url: url)
            items.append(item)
            Task {
                let info = try? await VideoUpscaler.inspect(url)
                update(item.id) { $0.info = info; $0.inspected = true }
            }
        }
    }

    func remove(_ id: Item.ID) {
        guard let item = items.first(where: { $0.id == id }), item.status != .upscaling, item.status != .converting else { return }
        items.removeAll { $0.id == id }
    }

    func clearFinished() {
        items.removeAll { $0.status == .done }
    }

    func start() {
        guard !isRunning, model == .ready else { return }
        isRunning = true
        cancelFlag = CancelFlag()
        activity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled], reason: "Upscaling videos")
        Task {
            while !cancelFlag.isSet, let next = items.first(where: { $0.status == .waiting }) {
                await process(next.id)
            }
            isRunning = false
            if let activity { ProcessInfo.processInfo.endActivity(activity) }
            activity = nil
        }
    }

    func stop() {
        cancelFlag.set()
    }

    // MARK: Processing

    private func process(_ id: Item.ID) async {
        guard let item = items.first(where: { $0.id == id }) else { return }
        let stem = item.url.deletingPathExtension().lastPathComponent
        let destination = outputFolder.appending(path: "\(stem) - 4K.mp4")
        let partial = outputFolder.appending(path: ".\(stem) - 4K.partial.mp4")
        let fm = FileManager.default

        if fm.fileExists(atPath: destination.path) {
            update(id) { $0.status = .done; $0.progress = 1; $0.output = destination }
            return
        }
        do {
            try fm.createDirectory(at: outputFolder, withIntermediateDirectories: true)
            var input = item.url
            var intermediate: URL?
            let readable = item.inspected ? item.info != nil : (try? await VideoUpscaler.inspect(item.url)) != nil
            if !readable {
                update(id) { $0.status = .converting }
                intermediate = try await Converter.toProRes(item.url)
                input = intermediate!
            }
            defer { if let intermediate { try? fm.removeItem(at: intermediate) } }

            update(id) { $0.status = .upscaling; $0.progress = 0 }
            var options = VideoUpscaler.Options()
            options.engine = .ml
            options.size = Quality.size
            options.bitrate = Quality.bitrate
            let flag = cancelFlag
            options.isCancelled = { flag.isSet }
            let throttle = Throttle(interval: 0.5)
            try? fm.removeItem(at: partial)

            _ = try await VideoUpscaler.upscale(input: input, output: partial, options: options) { p in
                guard throttle.shouldFire() || p.framesDone == p.framesTotal else { return }
                let fps = Double(p.framesDone) / max(p.elapsed, 0.001)
                let left = Double(max(p.framesTotal - p.framesDone, 0)) / max(fps, 0.001)
                let fraction = min(1, Double(p.framesDone) / Double(p.framesTotal))
                Task { @MainActor in
                    self.update(id) { $0.progress = fraction; $0.fps = fps; $0.secondsLeft = left }
                }
            }

            let source = try await VideoUpscaler.inspect(input)
            let result = try await VideoUpscaler.inspect(partial)
            guard abs(source.duration - result.duration) < 0.5 else {
                throw UpscaleError.io("Output is \(Int(result.duration)) s long but the source is \(Int(source.duration)) s.")
            }
            try fm.moveItem(at: partial, to: destination)
            update(id) { $0.status = .done; $0.progress = 1; $0.secondsLeft = nil; $0.output = destination }
        } catch is CancellationError {
            try? fm.removeItem(at: partial)
            update(id) { $0.status = .waiting; $0.progress = 0; $0.secondsLeft = nil }
        } catch {
            try? fm.removeItem(at: partial)
            update(id) { $0.status = .failed("\(error)"); $0.secondsLeft = nil }
        }
    }

    private func update(_ id: Item.ID, _ change: (inout Item) -> Void) {
        guard let index = items.firstIndex(where: { $0.id == id }) else { return }
        change(&items[index])
    }

    // MARK: Model

    func checkModel() {
        guard #available(macOS 26.0, *) else {
            model = .unsupported("Apple's AI upscaler needs macOS 26 or later.")
            return
        }
        model = AppleMLModel.isReady ? .ready : .missing
    }

    func downloadModel() {
        guard #available(macOS 26.0, *), model == .missing else { return }
        model = .downloading(0)
        Task.detached {
            do {
                try AppleMLModel.download { fraction in
                    Task { @MainActor in self.model = .downloading(Double(fraction)) }
                }
                await MainActor.run { self.model = .ready }
            } catch {
                await MainActor.run { self.model = .missing }
            }
        }
    }

    // MARK: Helpers

    static func isVideo(_ url: URL) -> Bool {
        if ["webm", "mkv"].contains(url.pathExtension.lowercased()) { return true }
        return UTType(filenameExtension: url.pathExtension)?.conforms(to: .audiovisualContent) == true
    }
}

/// Thread-safe stop flag shared with the background encode loop.
final class CancelFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var isSet: Bool { lock.withLock { value } }
    func set() { lock.withLock { value = true } }
}

/// Lets roughly one progress update through per interval.
final class Throttle: @unchecked Sendable {
    private let lock = NSLock()
    private let interval: TimeInterval
    private var last = Date.distantPast
    init(interval: TimeInterval) { self.interval = interval }
    func shouldFire() -> Bool {
        lock.withLock {
            guard Date().timeIntervalSince(last) >= interval else { return false }
            last = Date()
            return true
        }
    }
}

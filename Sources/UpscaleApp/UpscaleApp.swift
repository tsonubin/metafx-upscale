import AppKit
import SwiftUI

@main
struct UpscaleApp: App {
    @NSApplicationDelegateAdaptor private var delegate: AppDelegate
    @State private var queue = UpscaleQueue()

    var body: some Scene {
        Window("Upscale", id: "main") {
            ContentView(queue: queue)
                .onAppear {
                    delegate.queue = queue
                    LaunchArguments.apply(CommandLine.arguments, to: queue)
                }
        }
        .defaultSize(width: 760, height: 520)
        .windowResizability(.contentMinSize)
    }
}

final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var queue: UpscaleQueue?

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.regular)
        NSApp.activate()
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { true }

    /// Upscaling takes hours; don't lose it to a stray ⌘Q.
    @MainActor func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        guard let queue, queue.isRunning else { return .terminateNow }
        let alert = NSAlert()
        alert.messageText = "Upscaling is in progress"
        alert.informativeText = "Quitting stops the current video; it will start over next time."
        alert.addButton(withTitle: "Keep Upscaling")
        alert.addButton(withTitle: "Quit")
        return alert.runModal() == .alertSecondButtonReturn ? .terminateNow : .terminateCancel
    }
}

/// `Upscale --add FILE_OR_FOLDER... [--output DIR] [--start]`, for scripting and testing.
enum LaunchArguments {
    @MainActor static func apply(_ arguments: [String], to queue: UpscaleQueue) {
        var files: [URL] = []
        var start = false
        var iterator = arguments.dropFirst().makeIterator()
        var adding = false
        while let arg = iterator.next() {
            switch arg {
            case "--add": adding = true
            case "--start": start = true; adding = false
            case "--output":
                adding = false
                if let dir = iterator.next() { queue.outputFolder = URL(fileURLWithPath: dir) }
            default:
                if adding { files.append(URL(fileURLWithPath: arg)) }
            }
        }
        queue.add(files)
        if start { queue.start() }
    }
}

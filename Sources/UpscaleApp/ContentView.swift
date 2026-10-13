import AppKit
import SwiftUI
import UniformTypeIdentifiers
import UpscaleCore

struct ContentView: View {
    @Bindable var queue: UpscaleQueue
    @State private var importing = false
    @State private var dropTargeted = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            Group {
                if queue.items.isEmpty { dropZone } else { list }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .dropDestination(for: URL.self) { urls, _ in
                queue.add(urls)
                return true
            } isTargeted: { dropTargeted = $0 }
            Divider()
            footer
        }
        .frame(minWidth: 620, minHeight: 420)
        .fileImporter(
            isPresented: $importing,
            allowedContentTypes: [.movie, .video, .folder] + ["webm", "mkv"].compactMap { UTType(filenameExtension: $0) },
            allowsMultipleSelection: true
        ) { result in
            if case let .success(urls) = result { queue.add(urls) }
        }
    }

    private var header: some View {
        HStack(alignment: .firstTextBaseline) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Upscale to 4K").font(.title2.weight(.semibold))
                Text(Quality.summary).font(.caption).foregroundStyle(.secondary)
            }
            Spacer()
            modelBadge
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    @ViewBuilder private var modelBadge: some View {
        switch queue.model {
        case .checking:
            ProgressView().controlSize(.small)
        case .ready:
            Label("AI model ready", systemImage: "checkmark.seal.fill")
                .font(.caption).foregroundStyle(.green)
        case .missing:
            Button("Download AI model") { queue.downloadModel() }
                .help("macOS downloads Apple's super-resolution model from Apple (one time).")
        case let .downloading(fraction):
            ProgressView(value: fraction) { Text("Downloading model").font(.caption) }
                .frame(width: 150)
        case let .unsupported(reason):
            Label(reason, systemImage: "exclamationmark.triangle.fill")
                .font(.caption).foregroundStyle(.orange)
        }
    }

    private var dropZone: some View {
        VStack(spacing: 10) {
            Image(systemName: "film.stack").font(.system(size: 40, weight: .light))
            Text("Drop videos or a folder here").font(.headline)
            Button("Add Videos…") { importing = true }
        }
        .foregroundStyle(dropTargeted ? Color.accentColor : .secondary)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(
            RoundedRectangle(cornerRadius: 14)
                .strokeBorder(style: StrokeStyle(lineWidth: 1.5, dash: [6, 5]))
                .foregroundStyle(dropTargeted ? Color.accentColor : Color.secondary.opacity(0.4))
                .padding(20)
        )
    }

    private var list: some View {
        List {
            ForEach(queue.items) { item in
                ItemRow(item: item, onRemove: { queue.remove(item.id) })
            }
        }
        .listStyle(.inset)
        .overlay {
            if dropTargeted {
                RoundedRectangle(cornerRadius: 8).stroke(Color.accentColor, lineWidth: 2).padding(4)
            }
        }
    }

    private var footer: some View {
        HStack(spacing: 12) {
            Button("Add Videos…") { importing = true }
            Button {
                chooseOutputFolder()
            } label: {
                Label(queue.outputFolder.lastPathComponent, systemImage: "folder")
            }
            .help("Save to \(queue.outputFolder.path)")
            Spacer()
            if !queue.items.isEmpty {
                Text(progressSummary).font(.callout).foregroundStyle(.secondary).monospacedDigit()
            }
            if queue.isRunning {
                Button("Stop", role: .destructive) { queue.stop() }
                    .keyboardShortcut(.cancelAction)
            } else {
                Button("Start") { queue.start() }
                    .keyboardShortcut(.defaultAction)
                    .buttonStyle(.borderedProminent)
                    .disabled(queue.model != .ready || !queue.items.contains { $0.status == .waiting })
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
    }

    private var progressSummary: String {
        let done = queue.items.filter { $0.status == .done }.count
        return "\(done) of \(queue.items.count) done"
    }

    private func chooseOutputFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        panel.directoryURL = queue.outputFolder
        panel.prompt = "Save Here"
        if panel.runModal() == .OK, let url = panel.url { queue.outputFolder = url }
    }
}

private struct ItemRow: View {
    let item: UpscaleQueue.Item
    let onRemove: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            statusIcon.frame(width: 22)
            VStack(alignment: .leading, spacing: 4) {
                Text(item.name).lineLimit(1).truncationMode(.middle)
                detail
            }
            Spacer(minLength: 8)
            trailing
        }
        .padding(.vertical, 6)
        .contextMenu {
            if let output = item.output {
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([output]) }
            }
            Button("Show Original in Finder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
            if item.status != .upscaling && item.status != .converting {
                Button("Remove", role: .destructive, action: onRemove)
            }
        }
    }

    @ViewBuilder private var statusIcon: some View {
        switch item.status {
        case .waiting: Image(systemName: "clock").foregroundStyle(.secondary)
        case .converting, .upscaling: ProgressView().controlSize(.small)
        case .done: Image(systemName: "checkmark.circle.fill").foregroundStyle(.green)
        case .failed: Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.red)
        }
    }

    @ViewBuilder private var detail: some View {
        switch item.status {
        case .upscaling:
            VStack(alignment: .leading, spacing: 3) {
                ProgressView(value: item.progress)
                Text(runningText).font(.caption).foregroundStyle(.secondary).monospacedDigit()
            }
        case .converting:
            Text("Converting to a format the AI can read…").font(.caption).foregroundStyle(.secondary)
        case let .failed(message):
            Text(message).font(.caption).foregroundStyle(.red).lineLimit(2)
        default:
            Text(sizeText).font(.caption).foregroundStyle(.secondary).monospacedDigit()
        }
    }

    @ViewBuilder private var trailing: some View {
        switch item.status {
        case .done:
            if let output = item.output {
                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([output]) }
                    .buttonStyle(.link)
            }
        case .waiting, .failed:
            Button(action: onRemove) { Image(systemName: "xmark.circle.fill").foregroundStyle(.tertiary) }
                .buttonStyle(.plain)
                .help("Remove from the list")
        default:
            EmptyView()
        }
    }

    private var sizeText: String {
        guard item.inspected else { return "Reading…" }
        guard let info = item.info else { return "Will be converted first (WebM/MKV) · output 4K" }
        let out = Quality.size.resolve(sourceWidth: info.width, sourceHeight: info.height, evenDimensions: true)
        return "\(info.width)×\(info.height) → \(out.width)×\(out.height) · \(Format.duration(info.duration))"
    }

    private var runningText: String {
        var parts = ["\(Int(item.progress * 100))%"]
        if item.fps > 0 { parts.append(String(format: "%.1f fps", item.fps)) }
        if let left = item.secondsLeft, item.fps > 0 { parts.append("\(Format.duration(left)) left") }
        return parts.joined(separator: " · ")
    }
}

enum Format {
    static func duration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        if s >= 3600 { return String(format: "%dh %02dm", s / 3600, s / 60 % 60) }
        return String(format: "%d:%02d", s / 60, s % 60)
    }
}

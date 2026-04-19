import SwiftUI
import AVFoundation

/// Minimal UI to turn this phone into a pipeline-stage worker.
///
/// Flow:
///   1. Load GGUF model from Documents or bundle.
///   2. Enter layer range (e.g. 0..14 for the first shard).
///   3. Tap "Start Stage" — loads layer slice, starts HTTP server on port 50001.
///   4. Coordinator (Mac dashboard) registers this worker, routes inference.
///
/// Also displays the coordinator discovery URL if one is entered manually
/// or learned from /discover later. For now this is a one-way display; a
/// future revision can scan a QR code from the dashboard.
struct PipelineWorkerView: View {
    @StateObject private var vm = PipelineWorkerViewModel()
    @State private var modelPath: URL?
    @State private var layerStart: String = "0"
    @State private var layerEnd: String = "14"
    @State private var isFirst: Bool = true
    @State private var isLast: Bool = false
    @State private var port: UInt16 = 50001

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()
            if vm.running {
                runningView
            } else {
                setupView
            }
        }
        .onAppear { loadExistingModel() }
    }

    // MARK: - Setup

    var setupView: some View {
        ScrollView {
            VStack(spacing: 20) {
                Text("PIPELINE STAGE")
                    .font(.system(size: 28, weight: .black, design: .monospaced))
                    .foregroundColor(Color(hex: "#F5A623"))
                    .padding(.top, 60)

                Text("one slice of a bigger model")
                    .font(.caption).foregroundColor(.gray)

                // Model status
                VStack(alignment: .leading, spacing: 4) {
                    Text("MODEL").font(.caption2).foregroundColor(.gray)
                    if let p = modelPath {
                        Text(p.lastPathComponent)
                            .font(.system(size: 11, design: .monospaced))
                            .foregroundColor(.green)
                    } else {
                        Text("no GGUF found in Documents or bundle")
                            .foregroundColor(.red).font(.caption)
                    }
                }
                .padding().frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.gray.opacity(0.1)).cornerRadius(8)

                // Layer range
                VStack(alignment: .leading, spacing: 6) {
                    Text("LAYER RANGE").font(.caption2).foregroundColor(.gray)
                    HStack {
                        TextField("start", text: $layerStart)
                            .keyboardType(.numberPad)
                            .textFieldStyle(.roundedBorder)
                        Text("..").foregroundColor(.gray)
                        TextField("end (exclusive)", text: $layerEnd)
                            .keyboardType(.numberPad)
                            .textFieldStyle(.roundedBorder)
                    }
                    Text("e.g. 0..14 for first shard, 14..28 for second")
                        .font(.caption2).foregroundColor(.gray)
                }
                .padding().frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.gray.opacity(0.1)).cornerRadius(8)

                // Role toggles
                HStack {
                    Toggle("First Stage", isOn: $isFirst)
                        .font(.caption).foregroundColor(.white)
                    Toggle("Last Stage", isOn: $isLast)
                        .font(.caption).foregroundColor(.white)
                }
                .padding().background(Color.gray.opacity(0.1)).cornerRadius(8)

                if let err = vm.error {
                    Text(err).foregroundColor(.red).font(.caption)
                        .multilineTextAlignment(.center).padding(.horizontal)
                }

                Button {
                    start()
                } label: {
                    Text("Start Stage")
                        .font(.system(size: 16, weight: .bold, design: .monospaced))
                        .foregroundColor(.black)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color(hex: "#F5A623"))
                        .cornerRadius(10)
                }
                .disabled(modelPath == nil || vm.starting)
                .opacity(modelPath == nil ? 0.3 : 1.0)

                if vm.starting {
                    ProgressView().tint(Color(hex: "#F5A623"))
                }

                Spacer()
            }
            .padding()
        }
    }

    // MARK: - Running

    var runningView: some View {
        VStack(spacing: 0) {
            VStack(spacing: 4) {
                Text("STAGE")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.gray)
                Text("L \(vm.layerStart)..\(vm.layerEnd)")
                    .font(.system(size: 42, weight: .black, design: .monospaced))
                    .foregroundColor(Color(hex: "#F5A623"))
                Text("of \(vm.numLayersTotal) total")
                    .font(.caption2).foregroundColor(.gray)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 32)
            .background(Color(hex: "#F5A623").opacity(0.1))

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 12) {
                PMetricCard(label: "FORWARDS", value: "\(vm.forwards)", pulsing: vm.forwards > 0)
                PMetricCard(label: "TOK/S", value: String(format: "%.1f", vm.lastTps))
                PMetricCard(label: "LAST",   value: vm.lastKind.uppercased())
                PMetricCard(label: "ROLE",
                            value: vm.isFirst && vm.isLast ? "FIRST+LAST" :
                                   vm.isFirst ? "FIRST" :
                                   vm.isLast ? "LAST" : "MIDDLE")
                PMetricCard(label: "PORT", value: String(vm.port))
                PMetricCard(label: "HIDDEN", value: String(vm.hiddenSize))
            }
            .padding()

            if let url = vm.localURL {
                VStack(spacing: 4) {
                    Text("REGISTER ME")
                        .font(.caption2).foregroundColor(.gray)
                    Text(url)
                        .font(.system(size: 12, design: .monospaced))
                        .foregroundColor(Color(hex: "#00ff88"))
                        .textSelection(.enabled)
                }
                .padding()
            }

            // Server status — red if NWListener fails (usually Local Network permission)
            Text(vm.listenerState)
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(vm.listenerState.contains("ready") ? .green :
                                 vm.listenerState.contains("fail") ? .red : .yellow)
                .padding(.horizontal)

            Spacer()

            Text("this phone holds layers [\(vm.layerStart)..\(vm.layerEnd))\nof \(vm.modelName).\nhidden states flow over HTTP between shards.")
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding()
        }
    }

    // MARK: - Helpers

    func loadExistingModel() {
        if let bundled = Bundle.main.url(forResource: "Qwen3-0.6B-Q4_K_M", withExtension: "gguf") {
            modelPath = bundled
            return
        }
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let files = (try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil)) ?? []
        modelPath = files.first(where: { $0.pathExtension == "gguf" })
    }

    func start() {
        guard let p = modelPath else { return }
        guard let ls = Int32(layerStart), let le = Int32(layerEnd), ls < le else {
            vm.error = "invalid layer range"
            return
        }
        Task { await vm.start(modelPath: p, layerStart: ls, layerEnd: le,
                              isFirst: isFirst, isLast: isLast, port: port) }
    }
}

@MainActor
final class PipelineWorkerViewModel: ObservableObject {
    @Published var running = false
    @Published var starting = false
    @Published var error: String?

    @Published var layerStart: Int32 = 0
    @Published var layerEnd: Int32 = 0
    @Published var numLayersTotal: Int32 = 0
    @Published var isFirst: Bool = true
    @Published var isLast: Bool = false
    @Published var port: UInt16 = 50001
    @Published var hiddenSize: Int32 = 0
    @Published var modelName: String = ""
    @Published var forwards: Int = 0
    @Published var activeSeqs: Int = 0
    @Published var lastTps: Double = 0
    @Published var lastKind: String = "—"
    @Published var localURL: String? = nil
    @Published var listenerState: String = "not started"

    private var stage: PipelineStage?
    private var server: PipelineWorkerServer?
    private var audioSession: AVAudioSession?

    private func activateAudioSession() {
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
        try? session.setActive(true)
        audioSession = session
    }

    func start(modelPath: URL, layerStart: Int32, layerEnd: Int32,
               isFirst: Bool, isLast: Bool, port: UInt16) async {
        starting = true
        error = nil
        do {
            let stg = try PipelineStage(
                modelPath: modelPath.path,
                layerStart: layerStart,
                layerEnd: layerEnd,
                isFirst: isFirst,
                isLast: isLast)

            self.stage = stg
            self.layerStart = layerStart
            self.layerEnd = layerEnd
            self.numLayersTotal = await stg.numLayersTotal
            self.hiddenSize = await stg.hiddenSize
            self.isFirst = isFirst
            self.isLast = isLast
            self.port = port
            self.modelName = modelPath.deletingPathExtension().lastPathComponent

            // Keep the app alive when backgrounded — iOS suspends foreground
            // apps quickly and kills any NWListener. A silent audio session
            // under UIBackgroundModes=audio keeps us running.
            activateAudioSession()

            let srv = PipelineWorkerServer(port: port, stage: stg)
            srv.onListenerState = { [weak self] s in
                Task { @MainActor in self?.listenerState = s }
            }
            srv.onForwardComplete = { [weak self] tokens, tps, kind in
                Task { @MainActor in
                    guard let self else { return }
                    self.forwards += 1
                    self.lastTps = tps
                    self.lastKind = kind
                }
            }
            try await srv.start()
            self.server = srv

            // Best-effort WiFi IP for the UI — not authoritative, just human-readable.
            self.localURL = "http://\(bestGuessIP()):\(port)"

            running = true
            starting = false
        } catch {
            self.error = "start failed: \(error)"
            self.starting = false
        }
    }
}

struct PMetricCard: View {
    let label: String
    let value: String
    var pulsing: Bool = false
    @State private var glow = false
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 9, weight: .medium, design: .monospaced))
                .foregroundColor(.gray)
            Text(value)
                .font(.system(size: 15, weight: .bold, design: .monospaced))
                .foregroundColor(pulsing && glow ? Color(hex: "#F5A623") : .white)
                .lineLimit(1).minimumScaleFactor(0.5)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.white.opacity(0.05))
        .cornerRadius(8)
        .onChange(of: value) { _ in
            guard pulsing else { return }
            glow = true
            DispatchQueue.main.asyncAfter(deadline: .now() + 0.4) { glow = false }
        }
    }
}

/// Non-authoritative best-effort IP for display. Uses getifaddrs.
private func bestGuessIP() -> String {
    var ifaddr: UnsafeMutablePointer<ifaddrs>?
    guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return "0.0.0.0" }
    defer { freeifaddrs(ifaddr) }
    var ptr: UnsafeMutablePointer<ifaddrs>? = first
    while let p = ptr {
        defer { ptr = p.pointee.ifa_next }
        let addr = p.pointee.ifa_addr.pointee
        guard addr.sa_family == UInt8(AF_INET) else { continue }
        let name = String(cString: p.pointee.ifa_name)
        // en0 = WiFi on iPhone; en1 = ethernet on iPad sometimes
        guard name == "en0" || name == "en1" else { continue }
        var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
        if getnameinfo(p.pointee.ifa_addr,
                       socklen_t(p.pointee.ifa_addr.pointee.sa_len),
                       &host, socklen_t(host.count),
                       nil, 0, NI_NUMERICHOST) == 0 {
            let s = String(cString: host)
            if !s.isEmpty { return s }
        }
    }
    return "0.0.0.0"
}

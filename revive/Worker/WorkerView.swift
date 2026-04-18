import SwiftUI
import AVFoundation

/// Shown on every worker phone. Keeps the app in the foreground
/// and displays live inference status.
@MainActor
class WorkerViewModel: ObservableObject {
    @Published var role: AgentRole = .drafter
    @Published var modelName: String = "Loading..."
    @Published var status: WorkerStatus = .idle
    @Published var requestsServed: Int = 0
    @Published var currentTPS: Double = 0
    @Published var batteryPercent: Int = 0
    @Published var thermalState: String = "nominal"
    @Published var isGenerating: Bool = false

    private var inferenceHandler: InferenceHandler?
    private var httpServer: HTTPServer?
    private var advertiser: BonjourAdvertiser?
    private var audioSession: AVAudioSession?

    func setup(role: AgentRole, modelPath: URL, port: UInt16) async {
        self.role = role
        self.modelName = modelPath.deletingPathExtension().lastPathComponent
        self.status = .idle

        // Keep app alive via silent audio session
        activateAudioSession()

        // Load llama model
        guard let context = try? await LlamaContext.create_context(path: modelPath.path) else {
            modelName = "Failed to load model"
            return
        }

        let handler = InferenceHandler(llamaContext: context, role: role, modelName: modelName)
        inferenceHandler = handler

        // Start HTTP server
        let server = HTTPServer(port: port) { [weak self] request in
            guard let self, let handler = self.inferenceHandler else {
                return ("Error: handler not ready", DeviceMetrics.snapshot(
                    tokensGenerated: 0, tokensPerSecond: 0,
                    timeToFirstTokenMs: 0, totalTimeMs: 0))
            }
            await MainActor.run { self.isGenerating = true; self.status = .generating }
            let result = await handler.handle(request)
            await MainActor.run {
                self.isGenerating = false
                self.status = ProcessInfo.processInfo.thermalState == .serious ? .hot : .idle
                self.requestsServed += 1
                self.currentTPS = result.1.tokensPerSecond
            }
            return result
        }

        try? await server.start()
        httpServer = server

        // Advertise via Bonjour
        let ramMb = Int(ProcessInfo.processInfo.physicalMemory / (1024 * 1024))
        let adv = BonjourAdvertiser(role: role, model: modelName, port: port, ramMb: ramMb)
        adv.start()
        advertiser = adv

        // Start telemetry loop
        startTelemetryLoop()
    }

    private func startTelemetryLoop() {
        Task {
            while true {
                batteryPercent = DeviceMetrics.batteryPercent
                thermalState = DeviceMetrics.thermalStateString
                if thermalState == "serious" || thermalState == "critical" {
                    status = .hot
                }
                try? await Task.sleep(nanoseconds: 3_000_000_000)
            }
        }
    }

    private func activateAudioSession() {
        let session = AVAudioSession.sharedInstance()
        try? session.setCategory(.playback, mode: .default, options: [.mixWithOthers])
        try? session.setActive(true)
        audioSession = session
    }
}

struct WorkerView: View {
    @StateObject var vm = WorkerViewModel()
    @State private var modelPath: URL?
    @State private var selectedRole: AgentRole = .drafter
    @State private var port: UInt16 = 50001
    @State private var isSetup = false

    var body: some View {
        ZStack {
            Color.black.ignoresSafeArea()

            if isSetup {
                runningView
            } else {
                setupView
            }
        }
    }

    // MARK: - Setup screen

    var setupView: some View {
        VStack(spacing: 24) {
            Text("REVIVE")
                .font(.system(size: 48, weight: .black, design: .monospaced))
                .foregroundColor(.green)

            Text("Worker Node Setup")
                .font(.title2)
                .foregroundColor(.gray)

            VStack(alignment: .leading, spacing: 8) {
                Text("Role").foregroundColor(.gray).font(.caption)
                Picker("Role", selection: $selectedRole) {
                    ForEach(AgentRole.allCases.filter { $0 != .aggregator }, id: \.self) {
                        Text($0.displayName).tag($0)
                    }
                }
                .pickerStyle(.wheel)
                .frame(height: 120)
                .clipped()
            }
            .padding()
            .background(Color.gray.opacity(0.15))
            .cornerRadius(12)

            Button("Load Model from Files") {
                // In production: use UIDocumentPickerViewController
                // For hackathon: models pre-placed in Documents directory
                let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                let files = (try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil))
                modelPath = files?.first(where: { $0.pathExtension == "gguf" })
            }
            .buttonStyle(.bordered)
            .tint(.green)

            if let path = modelPath {
                Text("✓ \(path.lastPathComponent)")
                    .foregroundColor(.green)
                    .font(.caption)
            }

            Button("Start Worker") {
                guard let path = modelPath else { return }
                isSetup = true
                Task {
                    await vm.setup(role: selectedRole, modelPath: path, port: port)
                }
            }
            .buttonStyle(.borderedProminent)
            .tint(.green)
            .disabled(modelPath == nil)
        }
        .padding()
    }

    // MARK: - Running screen

    var runningView: some View {
        VStack(spacing: 0) {
            // Role header
            VStack(spacing: 4) {
                Text(vm.role.displayName.uppercased())
                    .font(.system(size: 32, weight: .black, design: .monospaced))
                    .foregroundColor(Color(hex: vm.role.color))

                PulsingDot(isActive: vm.isGenerating, color: Color(hex: vm.role.color))
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 32)
            .background(Color(hex: vm.role.color).opacity(0.1))

            // Metrics grid
            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 16) {
                MetricCard(label: "STATUS",    value: vm.status.rawValue.uppercased(), color: statusColor)
                MetricCard(label: "MODEL",     value: vm.modelName, color: .white)
                MetricCard(label: "REQUESTS",  value: "\(vm.requestsServed)", color: .white)
                MetricCard(label: "TOK/S",     value: String(format: "%.1f", vm.currentTPS), color: .green)
                MetricCard(label: "BATTERY",   value: "\(vm.batteryPercent)%", color: batteryColor)
                MetricCard(label: "THERMAL",   value: vm.thermalState, color: thermalColor)
            }
            .padding()

            Spacer()

            // Model description
            Text(vm.role.systemPrompt)
                .font(.caption)
                .foregroundColor(.gray)
                .multilineTextAlignment(.center)
                .padding()
        }
    }

    var statusColor: Color {
        switch vm.status {
        case .generating: return .yellow
        case .hot:        return .red
        case .offline:    return .gray
        case .idle:       return .green
        }
    }

    var batteryColor: Color {
        vm.batteryPercent < 20 ? .red : vm.batteryPercent < 50 ? .yellow : .green
    }

    var thermalColor: Color {
        switch vm.thermalState {
        case "nominal", "fair": return .green
        case "serious":         return .yellow
        case "critical":        return .red
        default:                return .gray
        }
    }
}

// MARK: - Subviews

struct MetricCard: View {
    let label: String
    let value: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(label)
                .font(.system(size: 10, weight: .medium, design: .monospaced))
                .foregroundColor(.gray)
            Text(value)
                .font(.system(size: 16, weight: .bold, design: .monospaced))
                .foregroundColor(color)
                .lineLimit(1)
                .minimumScaleFactor(0.5)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(Color.white.opacity(0.05))
        .cornerRadius(8)
    }
}

struct PulsingDot: View {
    let isActive: Bool
    let color: Color
    @State private var scale: CGFloat = 1.0

    var body: some View {
        Circle()
            .fill(isActive ? color : Color.gray)
            .frame(width: 12, height: 12)
            .scaleEffect(scale)
            .onChange(of: isActive) { active in
                if active {
                    withAnimation(.easeInOut(duration: 0.6).repeatForever(autoreverses: true)) {
                        scale = 1.5
                    }
                } else {
                    withAnimation { scale = 1.0 }
                }
            }
    }
}


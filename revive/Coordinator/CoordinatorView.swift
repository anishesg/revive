import SwiftUI
import Combine

@MainActor
class CoordinatorViewModel: ObservableObject {
    @Published var mode: SwarmMode = .swarm
    @Published var chatMessages: [ChatEntry] = []
    @Published var isQuerying: Bool = false
    @Published var lastResult: SwarmResult?
    @Published var modelLoaded: Bool = false

    let swarmManager = SwarmManager()
    private var aggregator: Aggregator?
    private var router: QueryRouter?
    private var webServer: CoordinatorWebServer?
    private var cancellable: AnyCancellable?

    struct ChatEntry: Identifiable {
        let id = UUID()
        let role: String  // "user", "swarm", agentRole rawValue
        let content: String
        let color: String?
        let model: String?
        let tps: Double?
    }

    func setup() {
        // Forward SwarmManager's changes to our own objectWillChange so SwiftUI re-renders
        cancellable = swarmManager.objectWillChange
            .receive(on: RunLoop.main)
            .sink { [weak self] _ in self?.objectWillChange.send() }

        let agg = Aggregator(modelName: "coordinator-model")
        self.aggregator = agg
        self.router = QueryRouter(swarmManager: swarmManager, aggregator: agg)

        // Try to load coordinator model if present
        Task {
            let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            let candidates = (try? FileManager.default.contentsOfDirectory(at: docs, includingPropertiesForKeys: nil))
                ?? []
            if let gguf = candidates.first(where: { $0.pathExtension == "gguf" }) {
                try? await agg.loadModel(path: gguf)
                await MainActor.run { self.modelLoaded = true }
                print("[Coordinator] Aggregator model loaded: \(gguf.lastPathComponent)")
            }
        }

        // Start web dashboard server
        webServer = CoordinatorWebServer(swarmManager: swarmManager)
        webServer?.start()
    }

    func sendQuery(_ text: String) {
        guard !text.trimmingCharacters(in: .whitespaces).isEmpty else { return }
        let query = text.trimmingCharacters(in: .whitespaces)

        chatMessages.append(ChatEntry(role: "user", content: query, color: nil, model: nil, tps: nil))
        isQuerying = true

        Task {
            guard let router = router else { return }
            let result = await router.route(query: query, mode: mode)

            await MainActor.run {
                // Add agent responses
                for response in result.agentResponses {
                    self.chatMessages.append(ChatEntry(
                        role: response.role.rawValue,
                        content: response.content,
                        color: response.role.color,
                        model: response.model,
                        tps: response.metrics.tokensPerSecond
                    ))
                }
                // Add final synthesized answer
                self.chatMessages.append(ChatEntry(
                    role: "swarm",
                    content: result.finalAnswer,
                    color: "#00ff88",
                    model: "MoA Synthesis",
                    tps: result.avgTPS
                ))
                self.lastResult = result
                self.isQuerying = false
            }
        }
    }

    func addManualWorker(host: String, port: Int, role: AgentRole, model: String) {
        swarmManager.addManualWorker(host: host, port: port, role: role, model: model)
    }
}

struct CoordinatorView: View {
    @StateObject private var vm = CoordinatorViewModel()
    @State private var inputText: String = ""
    @State private var showAddWorker: Bool = false
    @AppStorage("appMode") private var appMode: AppMode = .coordinator

    var body: some View {
        ZStack {
            Color(hex: "#0a0a0a").ignoresSafeArea()
            ScrollView {
                VStack(spacing: 0) {
                    headerBar
                    swarmOverview
                    workerGrid
                    chatSection
                    if let result = vm.lastResult {
                        metricsSection(result: result)
                    }
                }
                .padding(.horizontal, 16)
            }
        }
        .onAppear { vm.setup() }
        .sheet(isPresented: $showAddWorker) {
            AddWorkerSheet(vm: vm, isPresented: $showAddWorker)
        }
    }

    // MARK: - Header

    var headerBar: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 2) {
                Text("R E V I V E")
                    .font(.system(size: 22, weight: .black, design: .monospaced))
                    .foregroundColor(Color(hex: "#00ff88"))
                Text("Phone Swarm Collective Intelligence")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(.gray)
            }
            Spacer()
            modeToggle
            Button(action: { showAddWorker = true }) {
                Image(systemName: "plus.circle")
                    .foregroundColor(.gray)
            }
            Button(action: { appMode = .unset }) {
                Image(systemName: "arrow.left.circle")
                    .foregroundColor(.gray)
            }
        }
        .padding(.vertical, 14)
        .overlay(alignment: .bottom) {
            Rectangle().fill(Color(hex: "#222222")).frame(height: 1)
        }
    }

    var modeToggle: some View {
        HStack(spacing: 6) {
            ForEach([SwarmMode.swarm, SwarmMode.speed], id: \.self) { m in
                Button(action: { vm.mode = m }) {
                    Text(m == .swarm ? "⚡ Swarm" : "🚀 Speed")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(vm.mode == m ? Color(hex: "#00ff88") : .gray)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 5)
                        .background(
                            RoundedRectangle(cornerRadius: 6)
                                .fill(vm.mode == m ? Color(hex: "#00ff88").opacity(0.08) : Color(hex: "#111111"))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 6)
                                        .stroke(vm.mode == m ? Color(hex: "#00ff88") : Color(hex: "#222222"), lineWidth: 1)
                                )
                        )
                }
                .buttonStyle(.plain)
            }
        }
    }

    // MARK: - Swarm Overview

    var swarmOverview: some View {
        HStack(spacing: 20) {
            OverviewStat(label: "Nodes", value: "\(vm.swarmManager.workers.count)", color: "#00ff88")
            OverviewStat(label: "Active", value: "\(vm.swarmManager.workers.filter { $0.status == .generating }.count)", color: "#4a90d9")
            OverviewStat(label: "Tok/s", value: String(format: "%.0f", vm.swarmManager.workers.reduce(0) { $0 + $1.lastMetrics.tokensPerSecond }), color: "#f5a623")
            if vm.isQuerying {
                HStack(spacing: 6) {
                    ProgressView()
                        .scaleEffect(0.7)
                        .tint(Color(hex: "#00ff88"))
                    Text("querying...")
                        .font(.system(size: 11, design: .monospaced))
                        .foregroundColor(Color(hex: "#00ff88"))
                }
                .padding(.leading, 8)
            }
            Spacer()
            if !vm.modelLoaded {
                Text("⚠ No aggregator model")
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(Color(hex: "#f5a623"))
            }
        }
        .padding(.vertical, 12)
    }

    // MARK: - Worker Cards

    var workerGrid: some View {
        LazyVGrid(columns: [GridItem(.adaptive(minimum: 150))], spacing: 10) {
            ForEach(vm.swarmManager.workers) { worker in
                WorkerCard(worker: worker)
            }
            if vm.swarmManager.workers.isEmpty {
                emptySwarmCard
            }
        }
        .padding(.bottom, 16)
    }

    var emptySwarmCard: some View {
        VStack(spacing: 8) {
            Text("No workers")
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(.gray)
            Text("Launch the REVIVE app in Worker mode on other phones and connect to the same WiFi.")
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(Color(hex: "#444444"))
                .multilineTextAlignment(.center)
            Button("+ Add manually") { showAddWorker = true }
                .font(.system(size: 11, design: .monospaced))
                .foregroundColor(Color(hex: "#00ff88"))
        }
        .padding(16)
        .frame(maxWidth: .infinity)
        .background(Color(hex: "#111111"))
        .overlay(RoundedRectangle(cornerRadius: 10).stroke(Color(hex: "#222222"), lineWidth: 1))
        .cornerRadius(10)
    }

    // MARK: - Chat

    var chatSection: some View {
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(vm.chatMessages) { entry in
                            ChatBubble(entry: entry)
                                .id(entry.id)
                        }
                    }
                    .padding(16)
                }
                .frame(height: 280)
                .onChange(of: vm.chatMessages.count) { _ in
                    if let last = vm.chatMessages.last {
                        withAnimation { proxy.scrollTo(last.id, anchor: .bottom) }
                    }
                }
            }
            .background(Color(hex: "#111111"))
            .cornerRadius(12, corners: [.topLeft, .topRight])

            HStack(spacing: 0) {
                TextField("Ask the swarm anything...", text: $inputText)
                    .font(.system(size: 13, design: .monospaced))
                    .foregroundColor(.white)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 14)
                    .onSubmit { submitQuery() }

                Button("Send") { submitQuery() }
                    .font(.system(size: 13, weight: .bold, design: .monospaced))
                    .foregroundColor(.black)
                    .padding(.horizontal, 20)
                    .frame(maxHeight: .infinity)
                    .background(Color(hex: "#00ff88"))
                    .disabled(inputText.trimmingCharacters(in: .whitespaces).isEmpty || vm.isQuerying)
            }
            .background(Color(hex: "#111111"))
            .overlay(alignment: .top) {
                Rectangle().fill(Color(hex: "#222222")).frame(height: 1)
            }
            .cornerRadius(12, corners: [.bottomLeft, .bottomRight])
        }
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(hex: "#222222"), lineWidth: 1))
        .padding(.bottom, 16)
    }

    func submitQuery() {
        let text = inputText
        inputText = ""
        vm.sendQuery(text)
    }

    // MARK: - Metrics

    func metricsSection(result: SwarmResult) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("LAST QUERY METRICS")
                .font(.system(size: 10, weight: .bold, design: .monospaced))
                .foregroundColor(.gray)
                .kerning(2)

            HStack(spacing: 20) {
                OverviewStat(label: "Agents", value: "\(result.respondedCount)", color: "#4a90d9")
                OverviewStat(label: "Avg Tok/s", value: String(format: "%.1f", result.avgTPS), color: "#00ff88")
                OverviewStat(label: "Type", value: result.queryType.rawValue.uppercased(), color: "#f5a623")
                OverviewStat(label: "Mode", value: result.mode == .swarm ? "SWARM" : "SPEED", color: "#00ff88")
            }
        }
        .padding(16)
        .background(Color(hex: "#111111"))
        .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color(hex: "#222222"), lineWidth: 1))
        .cornerRadius(12)
        .padding(.bottom, 16)
    }
}

// MARK: - Supporting Views

struct WorkerCard: View {
    let worker: WorkerInfo

    var statusColor: Color {
        switch worker.status {
        case .generating: return Color(hex: "#00ff88")
        case .hot:        return Color(hex: "#ff4444")
        case .offline:    return .gray
        case .idle:       return Color(hex: "#444444")
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Text(worker.role.displayName.uppercased())
                    .font(.system(size: 10, weight: .bold, design: .monospaced))
                    .foregroundColor(Color(hex: worker.role.color))
                    .kerning(1)
                Spacer()
                Circle()
                    .fill(statusColor)
                    .frame(width: 7, height: 7)
                    .overlay(
                        Circle().fill(statusColor).frame(width: 7, height: 7)
                            .opacity(worker.status == .generating ? 0.5 : 0)
                            .scaleEffect(worker.status == .generating ? 1.5 : 1)
                            .animation(.easeInOut(duration: 0.8).repeatForever(), value: worker.status)
                    )
            }

            Text(worker.model)
                .font(.system(size: 9, design: .monospaced))
                .foregroundColor(.gray)
                .lineLimit(1)

            Text(String(format: "%.1f tok/s", worker.lastMetrics.tokensPerSecond))
                .font(.system(size: 18, weight: .bold, design: .monospaced))
                .foregroundColor(.white)

            Text("\(worker.host):\(worker.port)")
                .font(.system(size: 9, design: .monospaced))
                .foregroundColor(Color(hex: "#444444"))

            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    RoundedRectangle(cornerRadius: 2).fill(Color(hex: "#222222")).frame(height: 3)
                    RoundedRectangle(cornerRadius: 2)
                        .fill(Color(hex: worker.role.color))
                        .frame(width: geo.size.width * CGFloat(min(worker.weight, 1.0)), height: 3)
                }
            }
            .frame(height: 3)
        }
        .padding(12)
        .background(Color(hex: "#111111"))
        .overlay(
            RoundedRectangle(cornerRadius: 10)
                .stroke(
                    worker.status == .generating ? Color(hex: worker.role.color).opacity(0.6) : Color(hex: "#222222"),
                    lineWidth: 1
                )
        )
        .cornerRadius(10)
        .shadow(color: worker.status == .generating ? Color(hex: worker.role.color).opacity(0.15) : .clear, radius: 8)
    }
}

struct ChatBubble: View {
    let entry: CoordinatorViewModel.ChatEntry

    var roleLabel: String {
        switch entry.role {
        case "user":  return "YOU"
        case "swarm": return "SWARM · MoA"
        default:      return entry.role.uppercased()
        }
    }

    var bubbleColor: Color {
        switch entry.role {
        case "user":  return Color(hex: "#4a90d9").opacity(0.15)
        case "swarm": return Color(hex: "#00ff88").opacity(0.06)
        default:      return Color.white.opacity(0.03)
        }
    }

    var borderColor: Color {
        if let c = entry.color { return Color(hex: c) }
        return entry.role == "user" ? Color(hex: "#4a90d9") : Color(hex: "#333333")
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 8) {
                Text(roleLabel)
                    .font(.system(size: 10, design: .monospaced))
                    .foregroundColor(entry.color.map { Color(hex: $0) } ?? .gray)
                    .kerning(1)
                if let model = entry.model {
                    Text(model)
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(Color(hex: "#444444"))
                }
                if let tps = entry.tps, tps > 0 {
                    Spacer()
                    Text(String(format: "%.1f tok/s", tps))
                        .font(.system(size: 9, design: .monospaced))
                        .foregroundColor(Color(hex: "#444444"))
                }
            }

            Text(entry.content)
                .font(.system(size: 12, design: .monospaced))
                .foregroundColor(.white)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(bubbleColor)
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(borderColor.opacity(0.6), lineWidth: 1)
                )
                .cornerRadius(8)
        }
    }
}

struct OverviewStat: View {
    let label: String
    let value: String
    let color: String

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(value)
                .font(.system(size: 20, weight: .bold, design: .monospaced))
                .foregroundColor(Color(hex: color))
            Text(label)
                .font(.system(size: 10, design: .monospaced))
                .foregroundColor(.gray)
        }
    }
}

struct AddWorkerSheet: View {
    let vm: CoordinatorViewModel
    @Binding var isPresented: Bool
    @State private var host = ""
    @State private var port = "50001"
    @State private var selectedRole: AgentRole = .reasoner
    @State private var model = "phi-3-mini-3.8b-Q4_K_M"

    var body: some View {
        NavigationView {
            ZStack {
                Color(hex: "#0a0a0a").ignoresSafeArea()
                VStack(spacing: 20) {
                    VStack(alignment: .leading, spacing: 8) {
                        label("Host / IP Address")
                        TextField("192.168.1.x", text: $host)
                            .textInputAutocapitalization(.never)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(.body, design: .monospaced))
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        label("Port")
                        TextField("50001", text: $port)
                            .keyboardType(.numberPad)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(.body, design: .monospaced))
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        label("Role")
                        Picker("Role", selection: $selectedRole) {
                            ForEach(AgentRole.allCases.filter { $0 != .aggregator }, id: \.self) { role in
                                Text(role.displayName).tag(role)
                            }
                        }
                        .pickerStyle(.menu)
                        .tint(Color(hex: "#00ff88"))
                    }

                    VStack(alignment: .leading, spacing: 8) {
                        label("Model Name")
                        TextField("phi-3-mini-3.8b-Q4_K_M", text: $model)
                            .textInputAutocapitalization(.never)
                            .textFieldStyle(.roundedBorder)
                            .font(.system(.body, design: .monospaced))
                    }

                    Spacer()

                    Button("Add Worker") {
                        if let p = Int(port), !host.isEmpty {
                            vm.addManualWorker(host: host, port: p, role: selectedRole, model: model)
                            isPresented = false
                        }
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                    .background(Color(hex: "#00ff88"))
                    .foregroundColor(.black)
                    .font(.system(size: 14, weight: .bold, design: .monospaced))
                    .cornerRadius(10)
                    .disabled(host.isEmpty || port.isEmpty)
                }
                .padding(24)
            }
            .navigationTitle("Add Worker Manually")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { isPresented = false }
                        .tint(Color(hex: "#00ff88"))
                }
            }
        }
    }

    func label(_ text: String) -> some View {
        Text(text)
            .font(.system(size: 11, design: .monospaced))
            .foregroundColor(.gray)
    }
}

// MARK: - RoundedCorner helper

extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCornerShape(radius: radius, corners: corners))
    }
}

struct RoundedCornerShape: Shape {
    var radius: CGFloat
    var corners: UIRectCorner

    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(roundedRect: rect, byRoundingCorners: corners,
                                cornerRadii: CGSize(width: radius, height: radius))
        return Path(path.cgPath)
    }
}

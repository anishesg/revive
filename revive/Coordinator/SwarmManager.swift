import Foundation

/// Manages the swarm: discovers workers, fans out queries, collects responses.
@MainActor
class SwarmManager: ObservableObject {
    @Published var workers: [WorkerInfo] = []
    @Published var isQuerying: Bool = false

    private let browser = BonjourBrowser()
    private let session = URLSession.shared

    init() {
        browser.onWorkerDiscovered = { [weak self] worker in
            Task { @MainActor in
                guard let self else { return }
                if !self.workers.contains(where: { $0.name == worker.name }) {
                    self.workers.append(worker)
                    print("[SwarmManager] Worker joined: \(worker.role.displayName) @ \(worker.host):\(worker.port)")
                }
            }
        }
        browser.onWorkerLost = { [weak self] name in
            Task { @MainActor in
                guard let self else { return }
                self.workers.removeAll { $0.name == name }
                print("[SwarmManager] Worker left: \(name)")
            }
        }
        browser.start()
    }

    // MARK: - Manual worker registration (fallback for Android / no Bonjour)

    func addManualWorker(host: String, port: Int, role: AgentRole, model: String) {
        let worker = WorkerInfo(name: "\(role.rawValue)-manual", role: role, model: model,
                                host: host, port: port, ramMb: 3072)
        workers.append(worker)
    }

    // MARK: - Fan-out query

    /// Query a subset of workers in parallel. Returns all responses received before timeout.
    func query(
        prompt: String,
        roles: Set<AgentRole>,
        timeoutSeconds: TimeInterval = 10.0
    ) async -> [AgentResponse] {
        let targets = workers.filter { roles.contains($0.role) }
        guard !targets.isEmpty else { return [] }

        isQuerying = true
        defer { Task { @MainActor in self.isQuerying = false } }

        return await withTaskGroup(of: AgentResponse?.self) { group in
            for worker in targets {
                group.addTask { [weak self] in
                    guard let self else { return nil }
                    return await self.queryWorker(worker, prompt: prompt, timeout: timeoutSeconds)
                }
            }

            var responses: [AgentResponse] = []
            for await result in group {
                if let r = result { responses.append(r) }
            }
            return responses
        }
    }

    // MARK: - Single worker query

    private func queryWorker(_ worker: WorkerInfo, prompt: String, timeout: TimeInterval) async -> AgentResponse? {
        let url = worker.url.appendingPathComponent("v1/chat/completions")
        var request = URLRequest(url: url, timeoutInterval: timeout)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let body = CompletionRequest(
            model: worker.model,
            messages: [ChatMessage(role: "user", content: prompt)],
            max_tokens: 150,
            temperature: 0.7,
            stream: false,
            stop: ["<|im_end|>", "|im_end|", "</s>"]
        )
        request.httpBody = try? JSONEncoder().encode(body)

        do {
            let (data, _) = try await session.data(for: request)
            return parseWorkerResponse(data: data, worker: worker)
        } catch {
            print("[SwarmManager] Worker \(worker.role.rawValue) failed: \(error.localizedDescription)")
            // Mark worker as offline
            Task { @MainActor in
                if let idx = self.workers.firstIndex(where: { $0.id == worker.id }) {
                    self.workers[idx].status = .offline
                }
            }
            return nil
        }
    }

    private func parseWorkerResponse(data: Data, worker: WorkerInfo) -> AgentResponse? {
        struct ResponseEnvelope: Codable {
            struct Choice: Codable {
                struct Message: Codable { let content: String }
                let message: Message
            }
            let choices: [Choice]
            let metrics: ResponseMetrics?
        }

        guard let envelope = try? JSONDecoder().decode(ResponseEnvelope.self, from: data),
              let content = envelope.choices.first?.message.content else {
            return nil
        }

        let metrics = envelope.metrics ?? ResponseMetrics(
            tokensGenerated: 0, tokensPerSecond: 0,
            timeToFirstTokenMs: 0, totalTimeMs: 0,
            thermalState: "unknown", batteryPercent: -1, memoryUsedMb: 0
        )

        // Update worker metrics
        Task { @MainActor in
            if let idx = self.workers.firstIndex(where: { $0.id == worker.id }) {
                self.workers[idx].lastMetrics = metrics
                self.workers[idx].status = .idle
            }
        }

        return AgentResponse(role: worker.role, model: worker.model, content: content, metrics: metrics)
    }

    // MARK: - Health check

    func pingAllWorkers() async {
        for worker in workers {
            Task {
                let url = worker.url.appendingPathComponent("health")
                if let _ = try? await session.data(from: url) {
                    // alive
                } else {
                    Task { @MainActor in
                        if let idx = self.workers.firstIndex(where: { $0.id == worker.id }) {
                            self.workers[idx].status = .offline
                        }
                    }
                }
            }
        }
    }
}

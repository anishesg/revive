import Foundation

/// Routes queries to the right subset of workers based on query classification.
@MainActor
class QueryRouter {
    private let swarmManager: SwarmManager
    private let aggregator: Aggregator
    private var spotterWorker: WorkerInfo? {
        swarmManager.workers.first(where: { $0.role == .spotter })
    }

    init(swarmManager: SwarmManager, aggregator: Aggregator) {
        self.swarmManager = swarmManager
        self.aggregator = aggregator
    }

    // MARK: - Main entry point

    func route(query: String, mode: SwarmMode) async -> SwarmResult {
        if mode == .speed {
            return await routeSpeedMode(query: query)
        }

        // Step 1: Classify via Spotter (fast, 0.6B model)
        let queryType = await classifyQuery(query)
        print("[Router] Query classified as: \(queryType.rawValue)")

        // Step 2: Route to appropriate agents
        let targetRoles = queryType.targetRoles
        let responses = await swarmManager.query(
            prompt: query,
            roles: targetRoles,
            timeoutSeconds: queryType == .simpleFact ? 6.0 : 12.0
        )

        if responses.isEmpty {
            return SwarmResult(
                query: query,
                queryType: queryType,
                agentResponses: [],
                finalAnswer: "No agents responded. Check device connectivity.",
                mode: mode
            )
        }

        // Step 3: Aggregate (or pass through for simple facts)
        let finalAnswer: String
        if queryType.needsAggregation && responses.count > 1 {
            finalAnswer = await aggregator.synthesize(query: query, responses: responses)
        } else {
            finalAnswer = responses.first?.content ?? ""
        }

        return SwarmResult(
            query: query,
            queryType: queryType,
            agentResponses: responses,
            finalAnswer: finalAnswer,
            mode: mode
        )
    }

    // MARK: - Speed mode (speculative decoding pair)

    private func routeSpeedMode(query: String) async -> SwarmResult {
        // Use drafter → reasoner speculative decoding pair if available,
        // otherwise fall back to fastest single worker
        let drafter = swarmManager.workers.first(where: { $0.role == .drafter })
        let verifier = swarmManager.workers.sorted(by: { $0.weight > $1.weight }).first(where: { $0.role != .drafter })

        var targetRoles: Set<AgentRole> = []
        if let d = drafter { targetRoles.insert(d.role) }
        if let v = verifier { targetRoles.insert(v.role) }
        if targetRoles.isEmpty {
            targetRoles = Set(swarmManager.workers.prefix(1).map(\.role))
        }

        let responses = await swarmManager.query(
            prompt: query,
            roles: targetRoles.isEmpty ? [.reasoner] : targetRoles,
            timeoutSeconds: 8.0
        )

        // In speed mode, prefer the verifier's response
        let answer = responses.first(where: { $0.role == verifier?.role })?.content
                  ?? responses.first?.content
                  ?? "No response"

        return SwarmResult(
            query: query,
            queryType: .simpleFact,
            agentResponses: responses,
            finalAnswer: answer,
            mode: .speed
        )
    }

    // MARK: - Classification via Spotter

    private func classifyQuery(_ query: String) async -> QueryType {
        guard let spotter = spotterWorker else {
            return .complexReasoning // safe fallback
        }

        let classifyPrompt = "Classify this query: \(query)"
        let responses = await swarmManager.query(
            prompt: classifyPrompt,
            roles: [.spotter],
            timeoutSeconds: 3.0
        )

        let raw = responses.first?.content.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        let category = raw.components(separatedBy: .whitespacesAndNewlines).first?.uppercased() ?? ""

        switch category {
        case "SIMPLE_FACT":       return .simpleFact
        case "COMPLEX_REASONING": return .complexReasoning
        case "CREATIVE":          return .creative
        case "CODE":              return .code
        case "MATH":              return .math
        case "OPINION":           return .opinion
        default:                  return .complexReasoning
        }
    }
}

// MARK: - Result type

struct SwarmResult: Identifiable {
    let id = UUID()
    let query: String
    let queryType: QueryType
    let agentResponses: [AgentResponse]
    let finalAnswer: String
    let mode: SwarmMode

    var respondedCount: Int { agentResponses.count }

    /// Average tok/s across all agents
    var avgTPS: Double {
        guard !agentResponses.isEmpty else { return 0 }
        let total = agentResponses.reduce(0.0) { $0 + $1.metrics.tokensPerSecond }
        return total / Double(agentResponses.count)
    }
}

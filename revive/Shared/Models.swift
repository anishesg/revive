import Foundation

// MARK: - Agent Roles

enum AgentRole: String, Codable, CaseIterable {
    case reasoner    = "reasoner"
    case writer      = "writer"
    case concise     = "concise"
    case critic      = "critic"
    case factchecker = "factchecker"
    case drafter     = "drafter"
    case spotter     = "spotter"
    case aggregator  = "aggregator"

    var displayName: String {
        switch self {
        case .reasoner:    return "The Reasoner"
        case .writer:      return "The Writer"
        case .concise:     return "The Concise"
        case .critic:      return "The Critic"
        case .factchecker: return "The Factchecker"
        case .drafter:     return "The Drafter"
        case .spotter:     return "The Spotter"
        case .aggregator:  return "The Aggregator"
        }
    }

    var systemPrompt: String {
        switch self {
        case .reasoner:
            return "You are a rigorous analytical thinker. Think step by step. Show your reasoning chain explicitly. Prioritize logical correctness over brevity. If you are uncertain, say so."
        case .writer:
            return "You are an eloquent communicator. Write clear, well-structured, engaging responses. Use good prose. Prioritize readability and flow over exhaustive detail."
        case .concise:
            return "You are a master of brevity. Answer in as few words as possible while being complete and accurate. No fluff, no preamble, no filler phrases."
        case .critic:
            return "You are a devil's advocate. Identify flaws, edge cases, counterarguments, and unstated assumptions in the question and any obvious answers. Challenge everything."
        case .factchecker:
            return "You are a fact-checker. Focus only on verifiable, accurate information. Explicitly flag anything uncertain with [uncertain]. Cite specific numbers or sources when possible."
        case .drafter:
            return "You are a quick-response generator. Produce a fast first-pass answer. Speed and coverage matter more than polish. Just get the key points down."
        case .spotter:
            return "Classify the query into EXACTLY one of these categories (reply with only the category word): SIMPLE_FACT, COMPLEX_REASONING, CREATIVE, CODE, MATH, OPINION"
        case .aggregator:
            return "You are the Aggregator of a distributed AI swarm. You receive multiple responses from specialized agents and synthesize the single best answer by taking the strongest elements from each, resolving contradictions (prefer Factchecker and Critic over Drafter), using the Writer's clarity, and the Reasoner's logic. Output only the final synthesized answer."
        }
    }

    var color: String {
        switch self {
        case .reasoner:    return "#4A90D9"
        case .writer:      return "#7ED321"
        case .concise:     return "#F5A623"
        case .critic:      return "#D0021B"
        case .factchecker: return "#9B59B6"
        case .drafter:     return "#1ABC9C"
        case .spotter:     return "#E67E22"
        case .aggregator:  return "#2ECC71"
        }
    }
}

// MARK: - Query Classification

enum QueryType: String {
    case simpleFact        = "SIMPLE_FACT"
    case complexReasoning  = "COMPLEX_REASONING"
    case creative          = "CREATIVE"
    case code              = "CODE"
    case math              = "MATH"
    case opinion           = "OPINION"

    /// Which roles to activate for this query type
    var targetRoles: Set<AgentRole> {
        switch self {
        case .simpleFact:
            return [.reasoner, .concise]
        case .complexReasoning:
            return [.reasoner, .writer, .critic, .factchecker, .drafter]
        case .creative:
            return [.writer, .reasoner, .critic]
        case .code:
            return [.reasoner, .factchecker, .critic]
        case .math:
            return [.reasoner, .factchecker, .concise]
        case .opinion:
            return [.writer, .critic, .reasoner]
        }
    }

    var needsAggregation: Bool {
        return self != .simpleFact
    }
}

// MARK: - Network Models

struct ChatMessage: Codable {
    let role: String
    let content: String
}

struct CompletionRequest: Codable {
    let model: String?
    let messages: [ChatMessage]
    let max_tokens: Int?
    let temperature: Double?
    let stream: Bool?
}

struct AgentResponse: Codable, Identifiable {
    let id: UUID
    let role: AgentRole
    let model: String
    let content: String
    let metrics: ResponseMetrics

    init(role: AgentRole, model: String, content: String, metrics: ResponseMetrics) {
        self.id = UUID()
        self.role = role
        self.model = model
        self.content = content
        self.metrics = metrics
    }
}

struct ResponseMetrics: Codable {
    let tokensGenerated: Int
    let tokensPerSecond: Double
    let timeToFirstTokenMs: Int
    let totalTimeMs: Int
    let thermalState: String
    let batteryPercent: Int
    let memoryUsedMb: Int

    enum CodingKeys: String, CodingKey {
        case tokensGenerated   = "tokens_generated"
        case tokensPerSecond   = "tokens_per_second"
        case timeToFirstTokenMs = "time_to_first_token_ms"
        case totalTimeMs       = "total_time_ms"
        case thermalState      = "thermal_state"
        case batteryPercent    = "battery_percent"
        case memoryUsedMb      = "memory_used_mb"
    }
}

struct WorkerInfo: Codable, Identifiable {
    let id: UUID
    var name: String
    var role: AgentRole
    var model: String
    var host: String
    var port: Int
    var ramMb: Int
    var status: WorkerStatus
    var lastMetrics: ResponseMetrics

    init(name: String, role: AgentRole, model: String, host: String, port: Int, ramMb: Int) {
        self.id = UUID()
        self.name = name
        self.role = role
        self.model = model
        self.host = host
        self.port = port
        self.ramMb = ramMb
        self.status = .idle
        self.lastMetrics = ResponseMetrics(
            tokensGenerated: 0, tokensPerSecond: 0,
            timeToFirstTokenMs: 0, totalTimeMs: 0,
            thermalState: "nominal", batteryPercent: -1, memoryUsedMb: 0
        )
    }

    var url: URL {
        URL(string: "http://\(host):\(port)")!
    }

    /// Higher = more capable. Used for weighted aggregation.
    var weight: Double {
        let speedScore = min(lastMetrics.tokensPerSecond / 40.0, 1.0)
        let thermalPenalty: Double
        switch lastMetrics.thermalState {
        case "serious":  thermalPenalty = 0.5
        case "critical": thermalPenalty = 0.1
        default:         thermalPenalty = 1.0
        }
        return speedScore * thermalPenalty
    }
}

enum WorkerStatus: String, Codable {
    case idle       = "idle"
    case generating = "generating"
    case hot        = "hot"
    case offline    = "offline"
}

// MARK: - Swarm Query Modes

enum SwarmMode: String, CaseIterable, Hashable {
    case swarm = "swarm"
    case speed = "speed"
}

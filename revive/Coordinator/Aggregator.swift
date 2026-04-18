import Foundation

/// Runs MoA synthesis on the iPad's local model.
actor Aggregator {
    private var llamaContext: LlamaContext?
    private let modelName: String

    init(modelName: String) {
        self.modelName = modelName
    }

    func loadModel(path: URL) async throws {
        llamaContext = try LlamaContext.create_context(path: path.path)
        print("[Aggregator] Loaded \(path.lastPathComponent)")
    }

    // MARK: - MoA Synthesis

    func synthesize(query: String, responses: [AgentResponse]) async -> String {
        guard let ctx = llamaContext else {
            // Fallback: pick highest-weight response
            return responses.max(by: { a, b in
                a.metrics.tokensPerSecond < b.metrics.tokensPerSecond
            })?.content ?? responses.first?.content ?? ""
        }

        let prompt = buildAggregationPrompt(query: query, responses: responses)
        await ctx.clear()
        await ctx.completion_init(text: prompt)

        var output = ""
        var tokens = 0
        let maxTokens = 300

        while await !ctx.is_done && tokens < maxTokens {
            let token = await ctx.completion_loop()
            output += token
            tokens += 1
        }

        return output.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Prompt Construction

    private func buildAggregationPrompt(query: String, responses: [AgentResponse]) -> String {
        let agentSection = responses.map { resp in
            "[\(resp.role.displayName) — \(resp.model)]:\n\(resp.content)"
        }.joined(separator: "\n\n---\n\n")

        return """
        <|im_start|>system
        \(AgentRole.aggregator.systemPrompt)<|im_end|>
        <|im_start|>user
        Original question: \(query)

        Agent responses:

        \(agentSection)

        Synthesize the single best answer:<|im_end|>
        <|im_start|>assistant
        """
    }
}

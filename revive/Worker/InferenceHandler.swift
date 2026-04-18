import Foundation

/// Bridges an incoming CompletionRequest to LlamaContext inference.
/// Returns (generatedText, ResponseMetrics).
actor InferenceHandler {
    private let llamaContext: LlamaContext
    private let agentRole: AgentRole
    private let modelName: String

    init(llamaContext: LlamaContext, role: AgentRole, modelName: String) {
        self.llamaContext = llamaContext
        self.agentRole = role
        self.modelName = modelName
    }

    func handle(_ request: CompletionRequest) async -> (String, ResponseMetrics) {
        let userMessage = request.messages.last(where: { $0.role == "user" })?.content ?? ""
        let maxTokens = request.max_tokens ?? 150

        let prompt = buildPrompt(system: agentRole.systemPrompt, user: userMessage)

        let t0 = DispatchTime.now().uptimeNanoseconds
        var firstTokenTime: UInt64? = nil
        var generatedTokens = 0
        var output = ""

        // Reset context for fresh generation
        await llamaContext.clear()
        await llamaContext.completion_init(text: prompt)

        let tInit = DispatchTime.now().uptimeNanoseconds
        firstTokenTime = tInit

        while await !llamaContext.is_done && generatedTokens < maxTokens {
            let token = await llamaContext.completion_loop()
            output += token
            generatedTokens += 1
        }

        let t1 = DispatchTime.now().uptimeNanoseconds
        let totalMs = Int((t1 - t0) / 1_000_000)
        let firstTokenMs = Int(((firstTokenTime ?? t0) - t0) / 1_000_000)
        let genSeconds = Double(t1 - tInit) / 1_000_000_000.0
        let tps = genSeconds > 0 ? Double(generatedTokens) / genSeconds : 0

        let metrics = DeviceMetrics.snapshot(
            tokensGenerated: generatedTokens,
            tokensPerSecond: tps,
            timeToFirstTokenMs: firstTokenMs,
            totalTimeMs: totalMs
        )

        return (output, metrics)
    }

    // MARK: - Prompt formatting (ChatML)

    private func buildPrompt(system: String, user: String) -> String {
        """
        <|im_start|>system
        \(system)<|im_end|>
        <|im_start|>user
        \(user)<|im_end|>
        <|im_start|>assistant
        """
    }
}

import Foundation
import llama

/// A pipeline shard running on an iOS device. Owns one llama_context
/// configured with `il_first`/`il_last` so the graph only runs transformer
/// layers in `[layerStart, layerEnd)`.
///
/// On a non-first shard, `forward(...)` takes a pre-computed hidden-state
/// buffer via `batch.embd` (bytes: `n_tokens * n_embd * sizeof(float)`).
/// On a non-last shard, the final norm + lm_head are skipped and
/// `res->t_embd` carries the hidden state for the next shard (readable
/// via `llama_get_embeddings_ith`).
///
/// Correctness proof against the full-model reference is in
/// llama.cpp `tests/test-pipeline-split.cpp` (bit-exact match on splits
/// 1, 14, 27 for Qwen3-0.6B-Q4_K_M).
actor PipelineStage {
    // Immutable after init
    private let model: OpaquePointer
    private let context: OpaquePointer
    private let vocab: OpaquePointer

    let layerStart: Int32
    let layerEnd: Int32
    let isFirst: Bool
    let isLast: Bool
    let hiddenSize: Int32
    let numLayersTotal: Int32
    let vocabSize: Int32
    let modelPath: String

    // Reusable batch. Capacity sized at init.
    private var batch: llama_batch

    // Track positions per seq_id so coordinators can skip passing them
    // explicitly during decode. Keyed by (seq_id, llama_seq_id).
    private var seqIdToLlamaSeq: [String: llama_seq_id] = [:]
    private var nextLlamaSeq: llama_seq_id = 0
    private var positionsBySeq: [String: Int32] = [:]

    static let maxBatchTokens: Int32 = 512

    // MARK: - init / deinit

    init(modelPath: String,
         layerStart: Int32,
         layerEnd: Int32,
         isFirst: Bool,
         isLast: Bool,
         ctxSize: UInt32 = 4096) throws {
        llama_backend_init()

        var mparams = llama_model_default_params()
        #if targetEnvironment(simulator)
        mparams.n_gpu_layers = 0
        #endif
        guard let model = llama_model_load_from_file(modelPath, mparams) else {
            throw LlamaError.couldNotInitializeContext
        }
        self.model = model
        self.modelPath = modelPath
        self.vocab = llama_model_get_vocab(model)
        self.numLayersTotal = Int32(llama_model_n_layer(model))
        self.hiddenSize = Int32(llama_model_n_embd(model))
        self.vocabSize = Int32(llama_vocab_n_tokens(self.vocab))

        precondition(layerStart >= 0 && layerEnd <= numLayersTotal && layerStart < layerEnd,
                     "invalid layer range \(layerStart)..\(layerEnd) (model has \(numLayersTotal) layers)")

        self.layerStart = layerStart
        self.layerEnd = layerEnd
        self.isFirst = isFirst
        self.isLast = isLast

        let nThreads = Int32(max(1, min(6, ProcessInfo.processInfo.processorCount - 2)))
        var cparams = llama_context_default_params()
        cparams.n_ctx = ctxSize
        cparams.n_batch = UInt32(Self.maxBatchTokens)
        cparams.n_ubatch = UInt32(Self.maxBatchTokens)
        cparams.n_threads = nThreads
        cparams.n_threads_batch = nThreads
        // We need t_embd to be reachable via llama_get_embeddings_ith on non-last shards.
        cparams.embeddings = !isLast
        // New APIs from our llama.cpp fork:
        cparams.il_first = layerStart
        cparams.il_last = layerEnd
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED
        cparams.n_seq_max = 16

        guard let ctx = llama_init_from_model(model, cparams) else {
            llama_model_free(model)
            throw LlamaError.couldNotInitializeContext
        }
        self.context = ctx

        // IMPORTANT: `llama_batch_init(n_tokens, embd, n_seq_max)` allocates
        // EITHER `batch.token` (when embd == 0) OR `batch.embd` (when embd > 0),
        // never both. We only ever feed one mode per-stage, so pick the right
        // allocation at init time and never switch.
        if isFirst {
            // tokens path: embd==0 → token[] allocated, embd==NULL
            self.batch = llama_batch_init(Self.maxBatchTokens, 0, 16)
        } else {
            // hidden-state path: embd==n_embd → embd[] allocated, token==NULL
            self.batch = llama_batch_init(Self.maxBatchTokens, Int32(hiddenSize), 16)
        }
    }

    deinit {
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
    }

    // MARK: - seq_id management

    private func allocLlamaSeq(for seqId: String) -> llama_seq_id {
        if let id = seqIdToLlamaSeq[seqId] { return id }
        let id = nextLlamaSeq
        nextLlamaSeq &+= 1
        seqIdToLlamaSeq[seqId] = id
        positionsBySeq[seqId] = 0
        return id
    }

    /// Drop KV for this seq_id. Safe to call even if never registered.
    func reset(seqId: String) {
        guard let llamaSeq = seqIdToLlamaSeq.removeValue(forKey: seqId) else { return }
        positionsBySeq.removeValue(forKey: seqId)
        llama_memory_seq_rm(llama_get_memory(context), llamaSeq, 0, -1)
    }

    // MARK: - forward

    struct ForwardInput {
        let tokens: [llama_token]?   // first-stage only
        let hidden: Data?            // non-first-stage: n_tokens * hiddenSize * sizeof(Float32)
        let positions: [Int32]       // absolute position ids for these tokens
        /// If nil, auto-assign positions starting from positionsBySeq[seqId].
        /// If set, uses provided positions verbatim (coordinator knows better).
        let overrideSeqPosition: Bool
    }

    struct ForwardResult {
        let hidden: Data?         // non-last-stage: n_tokens * hiddenSize * sizeof(Float32)
        let logits: [Float]?      // last-stage only: last row's logits [vocab]
        let latencyMs: Double
        let nTokens: Int32
    }

    /// Run one forward pass of the shard over this ubatch. Updates internal
    /// position counter if `overrideSeqPosition` is false.
    func forward(seqId: String, input: ForwardInput) throws -> ForwardResult {
        let t0 = DispatchTime.now().uptimeNanoseconds
        let llamaSeq = allocLlamaSeq(for: seqId)

        let nTokens: Int32
        if isFirst {
            guard let toks = input.tokens, !toks.isEmpty else {
                throw PipelineError.missingTokens
            }
            nTokens = Int32(toks.count)
        } else {
            guard let h = input.hidden else {
                throw PipelineError.missingHidden
            }
            let expected = Int(input.positions.count) * Int(hiddenSize) * MemoryLayout<Float>.stride
            guard h.count == expected else {
                throw PipelineError.shapeMismatch(expected: expected, got: h.count)
            }
            nTokens = Int32(input.positions.count)
        }
        precondition(nTokens <= Self.maxBatchTokens, "batch exceeds capacity")
        precondition(Int32(input.positions.count) == nTokens, "positions size mismatch")

        // Populate batch. The batch's `token` OR `embd` buffer was allocated
        // exclusively at init time based on `isFirst`; the other is NULL.
        llama_batch_clear(&batch)
        batch.n_tokens = nTokens

        for i in 0..<Int(nTokens) {
            batch.pos[i] = llama_pos(input.positions[i])
            batch.n_seq_id[i] = 1
            if let slot = batch.seq_id[i] {
                slot[0] = llamaSeq
            }
            batch.logits[i] = (isLast && i == Int(nTokens) - 1) ? 1 : 0
        }

        if isFirst {
            guard let tokPtr = batch.token else { throw PipelineError.decodeFailed(rc: -2) }
            for i in 0..<Int(nTokens) {
                tokPtr[i] = input.tokens![i]
            }
        } else {
            guard let embdPtr = batch.embd else { throw PipelineError.decodeFailed(rc: -3) }
            input.hidden!.withUnsafeBytes { raw in
                if let src = raw.bindMemory(to: Float.self).baseAddress {
                    memcpy(embdPtr, src, Int(nTokens) * Int(hiddenSize) * MemoryLayout<Float>.stride)
                }
            }
        }

        let rc = llama_decode(context, batch)
        if rc != 0 {
            throw PipelineError.decodeFailed(rc: Int32(rc))
        }

        // Update positions counter if we're auto-tracking
        if !input.overrideSeqPosition {
            positionsBySeq[seqId] = (input.positions.last ?? -1) + 1
        }

        let t1 = DispatchTime.now().uptimeNanoseconds
        let latencyMs = Double(t1 - t0) / 1_000_000.0

        if isLast {
            guard let logitsPtr = llama_get_logits_ith(context, nTokens - 1) else {
                throw PipelineError.noLogits
            }
            let arr = Array(UnsafeBufferPointer(start: logitsPtr, count: Int(vocabSize)))
            return ForwardResult(hidden: nil, logits: arr, latencyMs: latencyMs, nTokens: nTokens)
        } else {
            // Read all n_tokens rows of hidden states into a Data blob.
            var out = Data(count: Int(nTokens) * Int(hiddenSize) * MemoryLayout<Float>.stride)
            out.withUnsafeMutableBytes { rawBuf in
                let dst = rawBuf.bindMemory(to: Float.self).baseAddress!
                for i in 0..<Int(nTokens) {
                    guard let e = llama_get_embeddings_ith(context, Int32(i)) else { continue }
                    for j in 0..<Int(hiddenSize) {
                        dst[i * Int(hiddenSize) + j] = e[j]
                    }
                }
            }
            return ForwardResult(hidden: out, logits: nil, latencyMs: latencyMs, nTokens: nTokens)
        }
    }
}

enum PipelineError: Error {
    case missingTokens
    case missingHidden
    case shapeMismatch(expected: Int, got: Int)
    case decodeFailed(rc: Int32)
    case noLogits
}

package com.revive.worker

/**
 * JNI bridge to llama.cpp native library.
 * Provides tokenization, inference, and model management.
 */
object LlamaJNI {
    init {
        System.loadLibrary("revive-llama")
    }

    external fun loadModel(modelPath: String, nCtx: Int, nThreads: Int, nGpuLayers: Int): Long
    external fun freeModel(contextPtr: Long)
    external fun complete(contextPtr: Long, prompt: String, maxTokens: Int, temperature: Float): String
    external fun modelInfo(contextPtr: Long): String
    external fun bench(contextPtr: Long, pp: Int, tg: Int): String
}

data class InferenceResult(
    val content: String,
    val tokensGenerated: Int,
    val tokensPerSecond: Double,
    val timeToFirstTokenMs: Int,
    val totalTimeMs: Int,
)

class LlamaEngine {
    private var contextPtr: Long = 0
    private var modelName: String = ""

    val isLoaded: Boolean get() = contextPtr != 0L

    fun load(modelPath: String, nCtx: Int = 2048, nGpuLayers: Int = 0): Boolean {
        if (contextPtr != 0L) {
            LlamaJNI.freeModel(contextPtr)
        }
        val threads = maxOf(1, Runtime.getRuntime().availableProcessors() - 1)
        contextPtr = LlamaJNI.loadModel(modelPath, nCtx, threads, nGpuLayers)
        modelName = modelPath.substringAfterLast("/").removeSuffix(".gguf")
        return contextPtr != 0L
    }

    fun complete(prompt: String, maxTokens: Int = 150, temperature: Float = 0.7f): InferenceResult {
        if (contextPtr == 0L) {
            return InferenceResult("", 0, 0.0, 0, 0)
        }

        val startMs = System.currentTimeMillis()
        val result = LlamaJNI.complete(contextPtr, prompt, maxTokens, temperature)
        val totalMs = (System.currentTimeMillis() - startMs).toInt()

        val tokens = result.split(" ").size
        val tps = if (totalMs > 0) tokens.toDouble() / (totalMs / 1000.0) else 0.0

        return InferenceResult(
            content = result,
            tokensGenerated = tokens,
            tokensPerSecond = tps,
            timeToFirstTokenMs = minOf(totalMs / 4, 200),
            totalTimeMs = totalMs,
        )
    }

    fun release() {
        if (contextPtr != 0L) {
            LlamaJNI.freeModel(contextPtr)
            contextPtr = 0
        }
    }

    fun getModelName(): String = modelName
}

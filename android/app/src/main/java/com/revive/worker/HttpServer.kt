package com.revive.worker

import android.content.Context
import android.util.Log
import com.google.gson.Gson
import io.ktor.http.*
import io.ktor.serialization.gson.*
import io.ktor.server.application.*
import io.ktor.server.engine.*
import io.ktor.server.netty.*
import io.ktor.server.plugins.contentnegotiation.*
import io.ktor.server.request.*
import io.ktor.server.response.*
import io.ktor.server.routing.*

/**
 * HTTP server exposing the REVIVE worker API.
 * Compatible with the protocol spec: /health, /v1/chat/completions, /metrics
 */
class ReviveHttpServer(
    private val context: Context,
    private val engine: LlamaEngine,
    private val role: String,
    private val port: Int,
) {
    companion object {
        private const val TAG = "ReviveHttpServer"
    }

    private var server: ApplicationEngine? = null
    private val gson = Gson()
    private val startTime = System.currentTimeMillis()

    fun start() {
        server = embeddedServer(Netty, port = port, host = "0.0.0.0") {
            install(ContentNegotiation) { gson() }

            routing {
                get("/health") {
                    call.respond(mapOf(
                        "status" to "ok",
                        "role" to role,
                        "platform" to "android",
                        "uptime" to ((System.currentTimeMillis() - startTime) / 1000),
                    ))
                }

                get("/metrics") {
                    call.respond(DeviceMetrics.snapshot(context))
                }

                post("/v1/chat/completions") {
                    val body = call.receiveText()
                    val request = gson.fromJson(body, CompletionRequest::class.java)

                    val systemPrompt = AgentRoles.systemPrompt(role)
                    val userMessage = request.messages.lastOrNull { it.role == "user" }?.content ?: ""
                    val prompt = buildChatMLPrompt(systemPrompt, userMessage)

                    val result = engine.complete(
                        prompt = prompt,
                        maxTokens = request.max_tokens ?: 150,
                        temperature = request.temperature?.toFloat() ?: 0.7f,
                    )

                    val metrics = DeviceMetrics.snapshot(
                        context,
                        tokensGenerated = result.tokensGenerated,
                        tokensPerSecond = result.tokensPerSecond,
                        timeToFirstTokenMs = result.timeToFirstTokenMs,
                        totalTimeMs = result.totalTimeMs,
                    )

                    call.respond(mapOf(
                        "choices" to listOf(mapOf("message" to mapOf("role" to "assistant", "content" to result.content))),
                        "metrics" to metrics,
                    ))
                }
            }
        }.start(wait = false)

        Log.i(TAG, "HTTP server started on port $port")
    }

    fun stop() {
        server?.stop(1000, 2000)
    }

    private fun buildChatMLPrompt(system: String, user: String): String {
        return "<|im_start|>system\n$system<|im_end|>\n<|im_start|>user\n$user<|im_end|>\n<|im_start|>assistant\n"
    }
}

data class ChatMessage(val role: String, val content: String)

data class CompletionRequest(
    val model: String? = null,
    val messages: List<ChatMessage> = emptyList(),
    val max_tokens: Int? = 150,
    val temperature: Double? = 0.7,
    val stream: Boolean? = false,
)

object AgentRoles {
    private val prompts = mapOf(
        "reasoner" to "You are a rigorous analytical thinker. Think step by step. Show your reasoning chain explicitly.",
        "writer" to "You are an eloquent communicator. Write clear, well-structured, engaging responses.",
        "concise" to "You are a master of brevity. Answer in as few words as possible while being complete and accurate.",
        "critic" to "You are a devil's advocate. Identify flaws, edge cases, counterarguments.",
        "factchecker" to "You are a fact-checker. Focus only on verifiable, accurate information.",
        "drafter" to "You are a quick-response generator. Produce a fast first-pass answer.",
        "spotter" to "Classify the query into EXACTLY one category: SIMPLE_FACT, COMPLEX_REASONING, CREATIVE, CODE, MATH, OPINION",
    )

    fun systemPrompt(role: String): String = prompts[role] ?: prompts["drafter"]!!

    val allRoles = prompts.keys.toList()
}

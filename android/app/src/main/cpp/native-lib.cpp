#include <jni.h>
#include <string>
#include <vector>
#include <android/log.h>

#include "llama.h"

#define TAG "ReviveLlama"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, TAG, __VA_ARGS__)

struct ReviveContext {
    llama_model *model;
    llama_context *ctx;
    const llama_vocab *vocab;
    llama_sampler *sampler;
    int n_ctx;
};

extern "C" {

JNIEXPORT jlong JNICALL
Java_com_revive_worker_LlamaJNI_loadModel(JNIEnv *env, jobject, jstring model_path, jint n_ctx, jint n_threads, jint n_gpu_layers) {
    const char *path = env->GetStringUTFChars(model_path, nullptr);
    LOGI("Loading model: %s", path);

    llama_backend_init();

    auto model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model *model = llama_model_load_from_file(path, model_params);
    env->ReleaseStringUTFChars(model_path, path);

    if (!model) {
        LOGE("Failed to load model");
        return 0;
    }

    auto ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_ctx;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    llama_context *ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOGE("Failed to create context");
        llama_model_free(model);
        return 0;
    }

    auto sparams = llama_sampler_chain_default_params();
    llama_sampler *sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.4f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(1234));

    auto *rc = new ReviveContext{model, ctx, llama_model_get_vocab(model), sampler, n_ctx};
    LOGI("Model loaded successfully");
    return reinterpret_cast<jlong>(rc);
}

JNIEXPORT void JNICALL
Java_com_revive_worker_LlamaJNI_freeModel(JNIEnv *, jobject, jlong ptr) {
    auto *rc = reinterpret_cast<ReviveContext *>(ptr);
    if (rc) {
        llama_sampler_free(rc->sampler);
        llama_free(rc->ctx);
        llama_model_free(rc->model);
        delete rc;
        llama_backend_free();
    }
}

JNIEXPORT jstring JNICALL
Java_com_revive_worker_LlamaJNI_complete(JNIEnv *env, jobject, jlong ptr, jstring jprompt, jint max_tokens, jfloat temperature) {
    auto *rc = reinterpret_cast<ReviveContext *>(ptr);
    if (!rc) return env->NewStringUTF("");

    const char *prompt = env->GetStringUTFChars(jprompt, nullptr);

    // Tokenize
    int n_prompt = strlen(prompt);
    int n_tokens_max = n_prompt + 128;
    std::vector<llama_token> tokens(n_tokens_max);
    int n_tokens = llama_tokenize(rc->vocab, prompt, n_prompt, tokens.data(), n_tokens_max, true, false);
    env->ReleaseStringUTFChars(jprompt, prompt);

    if (n_tokens < 0) {
        LOGE("Tokenization failed");
        return env->NewStringUTF("");
    }
    tokens.resize(n_tokens);

    // Clear KV cache
    llama_memory_clear(llama_get_memory(rc->ctx), true);

    // Evaluate prompt
    llama_batch batch = llama_batch_init(512, 0, 1);
    for (int i = 0; i < n_tokens; i++) {
        batch.token[batch.n_tokens] = tokens[i];
        batch.pos[batch.n_tokens] = i;
        batch.n_seq_id[batch.n_tokens] = 1;
        batch.seq_id[batch.n_tokens][0] = 0;
        batch.logits[batch.n_tokens] = (i == n_tokens - 1) ? 1 : 0;
        batch.n_tokens++;
    }

    if (llama_decode(rc->ctx, batch) != 0) {
        LOGE("Decode failed");
        llama_batch_free(batch);
        return env->NewStringUTF("");
    }

    // Generate
    std::string result;
    int n_cur = n_tokens;
    char buf[128];

    for (int i = 0; i < max_tokens; i++) {
        llama_token new_token = llama_sampler_sample(rc->sampler, rc->ctx, batch.n_tokens - 1);

        if (llama_vocab_is_eog(rc->vocab, new_token) || n_cur >= rc->n_ctx) {
            break;
        }

        int n = llama_token_to_piece(rc->vocab, new_token, buf, sizeof(buf), 0, false);
        if (n > 0) {
            result.append(buf, n);
        }

        // Prepare next batch
        batch.n_tokens = 0;
        batch.token[0] = new_token;
        batch.pos[0] = n_cur;
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;
        batch.n_tokens = 1;

        if (llama_decode(rc->ctx, batch) != 0) {
            LOGE("Decode failed during generation");
            break;
        }
        n_cur++;
    }

    llama_batch_free(batch);
    return env->NewStringUTF(result.c_str());
}

JNIEXPORT jstring JNICALL
Java_com_revive_worker_LlamaJNI_modelInfo(JNIEnv *env, jobject, jlong ptr) {
    auto *rc = reinterpret_cast<ReviveContext *>(ptr);
    if (!rc) return env->NewStringUTF("no model");
    char buf[256] = {0};
    llama_model_desc(rc->model, buf, sizeof(buf));
    return env->NewStringUTF(buf);
}

JNIEXPORT jstring JNICALL
Java_com_revive_worker_LlamaJNI_bench(JNIEnv *env, jobject, jlong ptr, jint pp, jint tg) {
    // Simplified bench stub
    return env->NewStringUTF("bench not implemented on Android");
}

} // extern "C"

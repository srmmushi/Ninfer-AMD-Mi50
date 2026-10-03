#ifndef NINFER_C_API_H_
#define NINFER_C_API_H_
// ---------------------------------------------------------------------------
// NInfer-HIP C API (extern "C"): the integration surface for external
// schedulers such as vLLM. It exposes model loading, tokenizer access,
// per-sequence prefill/decode, a batched decode step (continuous batching),
// KV-cache introspection and sampling — everything a PagedAttention-style
// scheduler needs without exposing any C++ types.
// ---------------------------------------------------------------------------

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ninfer_engine ninfer_engine_t;
typedef struct ninfer_session ninfer_session_t;

typedef struct ninfer_options {
  int max_context;      // KV capacity per sequence (tokens)
  int prefill_chunk;    // rows processed per prefill call
  int num_sequences;    // concurrent KV slots (continuous batching width)
  int gpu_count;        // number of entries in `gpus`
  const int* gpus;      // device ids for layer-split pipeline parallelism
} ninfer_options_t;

// ---- engine lifecycle -----------------------------------------------------
// Returns 0 on success, negative on error.
int ninfer_load(const char* model_dir, const ninfer_options_t* options,
                ninfer_engine_t** out_engine);
void ninfer_engine_free(ninfer_engine_t* engine);

int ninfer_vocab_size(ninfer_engine_t* engine);
int ninfer_max_context(ninfer_engine_t* engine);
int ninfer_num_sequences(ninfer_engine_t* engine);
int ninfer_device_count(ninfer_engine_t* engine);

// KV cache introspection (PagedAttention scheduling inputs).
long long ninfer_kv_bytes_per_token(ninfer_engine_t* engine);
long long ninfer_kv_total_bytes(ninfer_engine_t* engine);
// Elements between consecutive sequence slabs inside one layer (used by the
// scheduler when it maps logical blocks onto ninfer slots).
long long ninfer_kv_seq_stride_elems(ninfer_engine_t* engine);

// ---- tokenizer ------------------------------------------------------------
// Returns the number of ids written, or negative on error.
int ninfer_tokenize(ninfer_engine_t* engine, const char* text,
                    long long* out_ids, int capacity);
int ninfer_detokenize(ninfer_engine_t* engine, const long long* ids, int count,
                      char* out_text, int capacity);

// ---- sessions (one per in-flight request) ---------------------------------
ninfer_session_t* ninfer_session_create(ninfer_engine_t* engine);
void ninfer_session_free(ninfer_session_t* session);
void ninfer_session_reset(ninfer_session_t* session);
int ninfer_session_slot(ninfer_session_t* session);
int ninfer_session_length(ninfer_session_t* session);

// Prefill `count` tokens into the session (chunked internally). Writes the
// logits of the final token to `out_logits` (capacity >= vocab_size).
int ninfer_prefill(ninfer_session_t* session, const long long* tokens,
                   int count, float* out_logits, int capacity);

// Decode one token. Writes the resulting logits to `out_logits`.
int ninfer_decode(ninfer_session_t* session, long long token,
                  float* out_logits, int capacity);

// Batched decode: one token per session, executed as a single forward pass
// (continuous batching). `out_logits` must hold n * vocab_size floats, row i
// belonging to sessions[i].
int ninfer_decode_batch(ninfer_engine_t* engine, ninfer_session_t** sessions,
                        const long long* tokens, int n, float* out_logits,
                        int capacity);

// ---- sampling -------------------------------------------------------------
// Returns the sampled token id, or negative on error.
long long ninfer_sample(ninfer_engine_t* engine, const float* logits,
                        float temperature, int top_k, float top_p,
                        float repetition_penalty, unsigned long long seed);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // NINFER_C_API_H_

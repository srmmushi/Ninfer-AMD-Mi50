#include "api/ninfer_c_api.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/log.h"
#include "common/util.h"
#include "core/device.h"
#include "frontend/tokenizer.h"
#include "models/model.h"
#include "runtime/engine.h"

using namespace ninfer;

// Slot allocator: num_sequences KV slabs are reserved at load time and
// handed out round-robin to sessions. These structs must live in the global
// namespace so they match the forward declaration in ninfer_c_api.h
// (typedef struct ninfer_engine ninfer_engine_t).
struct ninfer_engine {
  std::unique_ptr<Model> model;
  std::unique_ptr<Tokenizer> tokenizer;
  int next_slot = 0;
};

struct ninfer_session {
  ninfer_engine* engine = nullptr;
  int slot = 0;
  int length = 0;  // tokens currently resident in this slot
};

namespace {

// Copies `vocab` floats of the logits row `row` from the head device.
int copy_logits_row(ninfer_engine* eng, const float* device_logits, int row,
                    float* out, int capacity) {
  int V = eng->model->config().vocab_size;
  if (capacity < V) return -1;
  const float* row_ptr = device_logits + static_cast<int64_t>(row) * V;
  DeviceGuard guard(eng->model->output_device());
  HIP_CHECK(hipStreamSynchronize(0));
  HIP_CHECK(hipMemcpyAsync(out, row_ptr, static_cast<size_t>(V) * sizeof(float),
                           hipMemcpyDeviceToHost, 0));
  HIP_CHECK(hipStreamSynchronize(0));
  return 0;
}

}  // namespace

extern "C" {

int ninfer_load(const char* model_dir, const ninfer_options_t* options,
                ninfer_engine_t** out_engine) {
  try {
    init_device();
    auto eng = new ninfer_engine();
    ModelOptions mo;
    if (options) {
      mo.max_context = options->max_context > 0 ? options->max_context : 32768;
      mo.prefill_chunk =
          options->prefill_chunk > 0 ? options->prefill_chunk : 512;
      mo.num_sequences = options->num_sequences > 0 ? options->num_sequences : 1;
      mo.gpu_ids.clear();
      for (int i = 0; i < options->gpu_count; ++i) {
        mo.gpu_ids.push_back(options->gpus[i]);
      }
      if (mo.gpu_ids.empty()) mo.gpu_ids.push_back(0);
      if (options->quant) mo.quant = options->quant;
      if (options->parallel) mo.parallel = options->parallel;
    }
    eng->tokenizer =
        std::make_unique<Tokenizer>(path_join(model_dir, "tokenizer.json"));
    eng->model = std::make_unique<Model>(model_dir, mo);
    *out_engine = eng;
    return 0;
  } catch (const std::exception& e) {
    LOG_ERROR("ninfer_load: %s", e.what());
    return -1;
  }
}

void ninfer_engine_free(ninfer_engine_t* engine) { delete engine; }

int ninfer_vocab_size(ninfer_engine_t* engine) {
  return engine->model->config().vocab_size;
}
int ninfer_max_context(ninfer_engine_t* engine) {
  return engine->model->max_context();
}
int ninfer_num_sequences(ninfer_engine_t* engine) {
  return engine->model->num_sequences();
}
int ninfer_device_count(ninfer_engine_t* engine) {
  return engine->model->device_count();
}

long long ninfer_kv_bytes_per_token(ninfer_engine_t* engine) {
  const ModelConfig& c = engine->model->config();
  // K and V, fp16, all layers, all kv heads.
  return 2LL * c.num_hidden_layers * c.num_key_value_heads * c.head_dim * 2;
}

long long ninfer_kv_total_bytes(ninfer_engine_t* engine) {
  return ninfer_kv_bytes_per_token(engine) * engine->model->max_context() *
         engine->model->num_sequences();
}

long long ninfer_kv_seq_stride_elems(ninfer_engine_t* engine) {
  const ModelConfig& c = engine->model->config();
  return static_cast<long long>(engine->model->max_context()) *
         c.num_key_value_heads * c.head_dim;
}

int ninfer_tokenize(ninfer_engine_t* engine, const char* text,
                    long long* out_ids, int capacity) {
  if (!engine || !text) return -1;
  try {
    auto ids = engine->tokenizer->encode(text);
    int n = std::min<int>(capacity, static_cast<int>(ids.size()));
    for (int i = 0; i < n; ++i) out_ids[i] = ids[i];
    return n;
  } catch (const std::exception& e) {
    LOG_ERROR("ninfer_tokenize: %s", e.what());
    return -1;
  }
}

int ninfer_detokenize(ninfer_engine_t* engine, const long long* ids, int count,
                      char* out_text, int capacity) {
  if (!engine || !ids) return -1;
  try {
    std::vector<int64_t> v(ids, ids + count);
    std::string text = engine->tokenizer->decode(v);
    int n = std::min<int>(capacity - 1, static_cast<int>(text.size()));
    std::memcpy(out_text, text.data(), n);
    out_text[n] = '\0';
    return n;
  } catch (const std::exception& e) {
    LOG_ERROR("ninfer_detokenize: %s", e.what());
    return -1;
  }
}

ninfer_session_t* ninfer_session_create(ninfer_engine_t* engine) {
  if (!engine) return nullptr;
  auto* s = new ninfer_session();
  s->engine = engine;
  s->slot = engine->next_slot++ % engine->model->num_sequences();
  s->length = 0;
  engine->model->reset_cache(s->slot);
  return s;
}

void ninfer_session_free(ninfer_session_t* session) { delete session; }

void ninfer_session_reset(ninfer_session_t* session) {
  if (!session) return;
  session->engine->model->reset_cache(session->slot);
  session->length = 0;
}

int ninfer_session_slot(ninfer_session_t* session) {
  return session ? session->slot : -1;
}
int ninfer_session_length(ninfer_session_t* session) {
  return session ? session->length : -1;
}

int ninfer_prefill(ninfer_session_t* session, const long long* tokens,
                   int count, float* out_logits, int capacity) {
  if (!session || !tokens || count <= 0) return -1;
  try {
    Model& model = *session->engine->model;
    const int chunk = model.prefill_chunk();
    int pos = session->length;
    ForwardOutput out;
    for (int start = 0; start < count; start += chunk) {
      int end = std::min(start + chunk, count);
      std::vector<int64_t> part;
      std::vector<int> positions, seq_ids;
      for (int i = start; i < end; ++i) {
        part.push_back(tokens[i]);
        positions.push_back(pos + (i - start));
        seq_ids.push_back(session->slot);
      }
      out = model.forward_batch(part, positions, seq_ids, /*stream=*/0);
    }
    session->length = pos + count;
    if (!out.logits || out.rows <= 0) return -2;
    return copy_logits_row(session->engine, out.logits, out.rows - 1, out_logits,
                           capacity);
  } catch (const std::exception& e) {
    LOG_ERROR("ninfer_prefill: %s", e.what());
    return -1;
  }
}

int ninfer_decode(ninfer_session_t* session, long long token,
                  float* out_logits, int capacity) {
  if (!session) return -1;
  try {
    Model& model = *session->engine->model;
    int pos = session->length;
    std::vector<int64_t> t = {token};
    std::vector<int> positions = {pos};
    std::vector<int> seq_ids = {session->slot};
    ForwardOutput out = model.forward_batch(t, positions, seq_ids, 0);
    session->length = pos + 1;
    if (!out.logits || out.rows < 1) return -2;
    return copy_logits_row(session->engine, out.logits, 0, out_logits, capacity);
  } catch (const std::exception& e) {
    LOG_ERROR("ninfer_decode: %s", e.what());
    return -1;
  }
}

int ninfer_decode_batch(ninfer_engine_t* engine, ninfer_session_t** sessions,
                        const long long* tokens, int n, float* out_logits,
                        int capacity) {
  if (!engine || !sessions || !tokens || n <= 0) return -1;
  try {
    Model& model = *engine->model;
    int V = model.config().vocab_size;
    if (capacity < n * V) return -3;
    std::vector<int64_t> t(n);
    std::vector<int> positions(n), seq_ids(n);
    for (int i = 0; i < n; ++i) {
      t[i] = tokens[i];
      positions[i] = sessions[i]->length;
      seq_ids[i] = sessions[i]->slot;
    }
    ForwardOutput out = model.forward_batch(t, positions, seq_ids, 0);
    if (!out.logits || out.rows != n) return -2;
    for (int i = 0; i < n; ++i) {
      int rc = copy_logits_row(engine, out.logits, i, out_logits + i * V, V);
      if (rc != 0) return rc;
      sessions[i]->length += 1;
    }
    return 0;
  } catch (const std::exception& e) {
    LOG_ERROR("ninfer_decode_batch: %s", e.what());
    return -1;
  }
}

long long ninfer_sample(ninfer_engine_t* engine, const float* logits,
                        float temperature, int top_k, float top_p,
                        float repetition_penalty, unsigned long long seed) {
  if (!engine || !logits) return -1;
  SamplingParams sp;
  sp.temperature = temperature;
  sp.top_k = top_k;
  sp.top_p = top_p;
  sp.repetition_penalty = repetition_penalty;
  std::vector<int64_t> empty;
  return sample_logits(logits, engine->model->config().vocab_size, sp, empty,
                       seed);
}

}  // extern "C"

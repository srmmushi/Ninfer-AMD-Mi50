#include "models/model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "common/json.h"
#include "common/log.h"
#include "common/util.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "ops/ops.h"
#include "runtime/kv_cache.h"

namespace ninfer {

struct Model::Impl {
  ModelOptions opt;
  ModelConfig cfg;
  Blas blas;

  int H, D, Hq, Hkv, V, I, layers;
  float eps, theta;
  float attn_scale;

  struct Layer {
    DeviceBuffer ln1, ln2;
    DeviceBuffer wq, wk, wv, wo;
    DeviceBuffer qn, kn;  // Qwen3 per-head q/k norms
    DeviceBuffer gate, up, down;                    // dense FFN
    DeviceBuffer moe_gate, moe_g, moe_u, moe_down;  // MoE [E, ...] stacked
  };
  std::vector<Layer> lw;

  DeviceBuffer embed, final_norm, lm_head;

  // Activation workspace (sized for prefill_chunk tokens).
  DeviceBuffer hidden, normed, qkv, attn_out, proj, gate_up, act, logits_buf;
  DeviceBuffer ids_dev;
  DeviceBuffer scores_scratch;  // fp32 [chunk, ctx] + fp16 [chunk, ctx]
  // MoE workspace.
  DeviceBuffer moe_logits, moe_perm, moe_eout, moe_gate_up, moe_act;
  DeviceBuffer moe_order_dev, moe_w_dev;

  size_t weight_bytes = 0;
  hipStream_t stream = 0;

  // ------------------------------------------------------------------
  // Weight loading: safetensors (single or sharded) -> FP16 device.
  // ------------------------------------------------------------------
  struct WeightSource {
    std::vector<std::unique_ptr<SafeTensorsFile>> files;
    std::map<std::string, SafeTensorsFile*> by_name;

    bool has(const std::string& n) const { return by_name.count(n) != 0; }
  };

  static WeightSource open_weights(const std::string& dir) {
    WeightSource ws;
    std::string index_path = path_join(dir, "model.safetensors.index.json");
    std::vector<std::string> files;
    if (file_exists(index_path)) {
      Json idx = Json::parse(read_file_text(index_path));
      const Json* wm = idx.find("weight_map");
      if (!wm) throw std::runtime_error("bad safetensors index json");
      for (const auto& [name, file] : wm->members()) {
        if (std::find(files.begin(), files.end(), file.as_string()) ==
            files.end()) {
          files.push_back(file.as_string());
        }
      }
      LOG_INFO("checkpoint: %d shard(s)", static_cast<int>(files.size()));
    } else {
      files.push_back("model.safetensors");
    }
    for (const auto& f : files) {
      ws.files.push_back(
          std::make_unique<SafeTensorsFile>(path_join(dir, f)));
    }
    for (auto& f : ws.files) {
      for (const auto& [name, info] : f->index()) {
        ws.by_name[name] = f.get();
      }
    }
    return ws;
  }

  DeviceBuffer load_fp16(const WeightSource& ws, const std::string& name,
                         bool required = true) const {
    DeviceBuffer buf;
    auto it = ws.by_name.find(name);
    if (it == ws.by_name.end()) {
      if (required) {
        throw std::runtime_error("missing tensor in checkpoint: " + name);
      }
      return buf;
    }
    auto raw = it->second->read_raw(name);
    const StTensorInfo& info = it->second->tensor_info(name);
    auto halfs = convert_to_fp16(raw.data(), info.dtype, info.num_elements());
    buf.alloc(halfs.size() * sizeof(uint16_t));
    buf.copy_from_host(halfs.data(), buf.bytes());
    weight_bytes += buf.bytes();
    return buf;
  }

  // ------------------------------------------------------------------
  void load(const std::string& dir) {
    cfg = ModelConfig::from_file(path_join(dir, "config.json"));
    H = cfg.hidden_size;
    D = cfg.head_dim;
    Hq = cfg.num_attention_heads;
    Hkv = cfg.num_key_value_heads;
    V = cfg.vocab_size;
    I = cfg.ffn_inner();
    layers = cfg.num_hidden_layers;
    eps = cfg.rms_norm_eps;
    theta = cfg.rope_theta;
    attn_scale = 1.0f / std::sqrt(static_cast<float>(D));
    if (D != 64 && D != 128) {
      throw std::runtime_error("only head_dim 64/128 supported on gfx906");
    }
    if (H % Hkv != 0) {
      throw std::runtime_error("num_attention_heads not divisible by "
                               "num_key_value_heads");
    }

    LOG_INFO("loading weights from %s ...", dir.c_str());
    WeightSource ws = open_weights(dir);

    embed = load_fp16(ws, "model.embed_tokens.weight");
    final_norm = load_fp16(ws, "model.norm.weight");
    lm_head = cfg.tie_word_embeddings
                  ? DeviceBuffer()
                  : load_fp16(ws, "lm_head.weight");
    // With tied embeddings the lm_head weight IS the embedding table, so
    // lm_head_ptr() aliases `embed` at forward time.

    lw.resize(layers);
    for (int l = 0; l < layers; ++l) {
      Layer& L = lw[l];
      std::string p = "model.layers." + std::to_string(l) + ".";
      L.ln1 = load_fp16(ws, p + "input_layernorm.weight");
      L.ln2 = load_fp16(ws, p + "post_attention_layernorm.weight");
      L.wq = load_fp16(ws, p + "self_attn.q_proj.weight");
      L.wk = load_fp16(ws, p + "self_attn.k_proj.weight");
      L.wv = load_fp16(ws, p + "self_attn.v_proj.weight");
      L.wo = load_fp16(ws, p + "self_attn.o_proj.weight");
      if (cfg.use_qk_norm) {
        L.qn = load_fp16(ws, p + "self_attn.q_norm.weight");
        L.kn = load_fp16(ws, p + "self_attn.k_norm.weight");
      }
      if (cfg.is_moe) {
        L.moe_gate = load_fp16(ws, p + "mlp.gate.weight");
        L.moe_g = load_fp16(ws, p + "mlp.experts.gate_proj.weight");
        L.moe_u = load_fp16(ws, p + "mlp.experts.up_proj.weight");
        L.moe_down = load_fp16(ws, p + "mlp.experts.down_proj.weight");
      } else {
        L.gate = load_fp16(ws, p + "mlp.gate_proj.weight");
        L.up = load_fp16(ws, p + "mlp.up_proj.weight");
        L.down = load_fp16(ws, p + "mlp.down_proj.weight");
      }
      LOG_DEBUG("layer %d loaded", l);
    }
    HIP_CHECK(hipDeviceSynchronize());
    LOG_INFO("weights resident: %.2f GiB",
             static_cast<double>(weight_bytes) / (1024.0 * 1024.0 * 1024.0));

    alloc_workspace();
  }

  void alloc_workspace() {
    int chunk = opt.prefill_chunk;
    size_t T = chunk;
    hidden.alloc(static_cast<size_t>(T) * H * 2);
    normed.alloc(static_cast<size_t>(T) * H * 2);
    int wqkv = (Hq + 2 * Hkv) * D;
    qkv.alloc(static_cast<size_t>(T) * wqkv * 2);
    attn_out.alloc(static_cast<size_t>(T) * Hq * D * 2);
    proj.alloc(static_cast<size_t>(T) * H * 2);
    gate_up.alloc(static_cast<size_t>(T) * 2 * I * 2);
    act.alloc(static_cast<size_t>(T) * I * 2);
    logits_buf.alloc(static_cast<size_t>(V) * 4);
    ids_dev.alloc(T * sizeof(int64_t));
    // prefill score scratch: fp32 [chunk, ctx] followed by fp16 [chunk, ctx].
    scores_scratch.alloc(static_cast<size_t>(chunk) * opt.max_context * 6);
    if (cfg.is_moe) {
      size_t tsel = static_cast<size_t>(chunk) * cfg.num_experts_per_tok;
      moe_logits.alloc(static_cast<size_t>(T) * cfg.num_experts * 4);
      moe_perm.alloc(tsel * H * 2);
      moe_eout.alloc(tsel * H * 2);
      moe_gate_up.alloc(tsel * 2 * I * 2);
      moe_act.alloc(tsel * I * 2);
      moe_order_dev.alloc(tsel * sizeof(int64_t));
      moe_w_dev.alloc(tsel * sizeof(float));
    }
    HIP_CHECK(hipDeviceSynchronize());
  }

  const __half* lm_head_ptr() const {
    return cfg.tie_word_embeddings
               ? static_cast<const __half*>(embed.data())
               : static_cast<const __half*>(lm_head.data());
  }

  // ------------------------------------------------------------------
  const float* forward(const std::vector<int64_t>& tokens, int pos0,
                       KVCache& kv, hipStream_t st) {
    stream = st;
    int T = static_cast<int>(tokens.size());
    int chunk = opt.prefill_chunk;
    if (T > chunk) throw std::runtime_error("prefill chunk exceeds workspace");
    HIP_CHECK(hipMemcpyAsync(ids_dev.data(), tokens.data(),
                             T * sizeof(int64_t), hipMemcpyHostToDevice,
                             stream));
    embed_gather_fp16(hidden.data(), embed.data(),
                      static_cast<const int64_t*>(ids_dev.data()), T, H,
                      stream);

    int wqkv = (Hq + 2 * Hkv) * D;
    for (int l = 0; l < layers; ++l) {
      Layer& L = lw[l];
      rmsnorm_fp16(normed.data(), hidden.data(), L.ln1.data(), T, H, eps,
                   stream);

      // Fused qkv: three gemms into one [T, wqkv] buffer.
      blas.gemm_view(normed.data(), H, false, L.wq.data(), H, true,
                     qkv.data(), wqkv, Hq * D, T, H, false, stream);
      blas.gemm_view(normed.data(), H, false, L.wk.data(), H, true,
                     static_cast<__half*>(qkv.data()) + Hq * D, wqkv,
                     Hkv * D, T, H, false, stream);
      blas.gemm_view(normed.data(), H, false, L.wv.data(), H, true,
                     static_cast<__half*>(qkv.data()) + (Hq + Hkv) * D, wqkv,
                     Hkv * D, T, H, false, stream);

      __half* qsec = static_cast<__half*>(qkv.data());
      __half* ksec = qsec + Hq * D;
      if (cfg.use_qk_norm) {
        rmsnorm_heads_fp16(qsec, qsec, L.qn.data(), T, Hq, D, wqkv, eps,
                           stream);
        rmsnorm_heads_fp16(ksec, ksec, L.kn.data(), T, Hkv, D, wqkv, eps,
                           stream);
      }
      rope_fp16(qsec, T, Hq, D, wqkv, pos0, theta, stream);
      rope_fp16(ksec, T, Hkv, D, wqkv, pos0, theta, stream);

      kv.append(l, pos0, qsec, qsec, Hq * D, (Hq + Hkv) * D, wqkv, T, stream);

      if (T == 1) {
        attention_decode_fp16(attn_out.data(), qsec,
                              kv.k_data(l), kv.v_data(l), pos0 + 1, Hq, Hkv, D,
                              attn_scale, stream);
      } else {
        for (int h = 0; h < Hq; ++h) {
          attention_prefill_head_fp16(
              static_cast<__half*>(attn_out.data()) + h * D, qsec + h * D,
              kv.k_data(l), kv.v_data(l),
              static_cast<float*>(scores_scratch.data()), T, pos0 + T, pos0, D,
              wqkv, Hq * D, attn_scale, Hkv * D, blas, stream);
        }
      }

      // o_proj + residual.
      blas.gemm_view(attn_out.data(), Hq * D, false, L.wo.data(), H, true,
                     proj.data(), H, H, T, Hq * D, false, stream);
      residual_add_fp16(hidden.data(), proj.data(), T, H, stream);

      // FFN.
      rmsnorm_fp16(normed.data(), hidden.data(), L.ln2.data(), T, H, eps,
                   stream);
      if (!cfg.is_moe) {
        blas.gemm_view(normed.data(), H, false, L.gate.data(), H, true,
                       gate_up.data(), 2 * I, I, T, H, false, stream);
        blas.gemm_view(normed.data(), H, false, L.up.data(), H, true,
                       static_cast<__half*>(gate_up.data()) + I, 2 * I, I, T, H,
                       false, stream);
        silu_and_mul_fp16(act.data(), gate_up.data(), T, I, stream);
        blas.gemm_view(act.data(), I, false, L.down.data(), I, true,
                       proj.data(), H, H, T, I, false, stream);
      } else {
        int E = cfg.num_experts;
        int topk = cfg.num_experts_per_tok;
        blas.gemm_fp32out(normed.data(), L.moe_gate.data(), moe_logits.data(),
                          T, E, H, stream);
        MoERoutingPlan plan =
            moe_route_cpu(static_cast<const float*>(moe_logits.data()), T, E,
                          topk, blas, stream);
        size_t tsel = plan.order.size();
        HIP_CHECK(hipMemcpyAsync(moe_order_dev.data(), plan.order.data(),
                                 tsel * sizeof(int64_t),
                                 hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(moe_w_dev.data(), plan.order_weights.data(),
                                 tsel * sizeof(float), hipMemcpyHostToDevice,
                                 stream));
        gather_rows_fp16(moe_perm.data(), normed.data(),
                         static_cast<const int64_t*>(moe_order_dev.data()),
                         static_cast<int>(tsel), H, stream);
        for (int e = 0; e < E; ++e) {
          int64_t off = plan.expert_offsets[e];
          int64_t n = plan.expert_offsets[e + 1] - off;
          if (n <= 0) continue;
          const __half* wg = static_cast<const __half*>(L.moe_g.data()) +
                             static_cast<int64_t>(e) * I * H;
          const __half* wu = static_cast<const __half*>(L.moe_u.data()) +
                             static_cast<int64_t>(e) * I * H;
          const __half* wd = static_cast<const __half*>(L.moe_down.data()) +
                             static_cast<int64_t>(e) * H * I;
          __half* in = static_cast<__half*>(moe_perm.data()) + off * H;
          __half* gu = static_cast<__half*>(moe_gate_up.data()) + off * 2 * I;
          blas.gemm_view(in, H, false, wg, H, true, gu, 2 * I, I,
                         static_cast<int>(n), H, false, stream);
          blas.gemm_view(in, H, false, wu, H, true, gu + I, 2 * I, I,
                         static_cast<int>(n), H, false, stream);
          silu_and_mul_fp16(static_cast<__half*>(moe_act.data()) + off * I, gu,
                            static_cast<int>(n), I, stream);
          __half* eo = static_cast<__half*>(moe_eout.data()) + off * H;
          blas.gemm_view(static_cast<const __half*>(moe_act.data()) + off * I,
                         I, false, wd, I, true, eo, H, H,
                         static_cast<int>(n), I, false, stream);
        }
        scale_rows_fp16(moe_eout.data(),
                        static_cast<const float*>(moe_w_dev.data()),
                        static_cast<int>(tsel), H, stream);
        scatter_add_rows_fp16(hidden.data(), moe_eout.data(),
                              static_cast<const int64_t*>(moe_order_dev.data()),
                              static_cast<int>(tsel), H, stream);
      }
      residual_add_fp16(hidden.data(), proj.data(), T, H, stream);
    }

    // Final norm + lm_head for the last position only.
    __half* h_last = static_cast<__half*>(hidden.data()) + (T - 1) * H;
    __half* n_last = static_cast<__half*>(normed.data()) + (T - 1) * H;
    rmsnorm_fp16(n_last, h_last, final_norm.data(), 1, H, eps, stream);
    blas.gemm_fp32out(n_last, lm_head_ptr(), logits_buf.data(), 1, V, H,
                      stream);
    return static_cast<const float*>(logits_buf.data());
  }
};

// ---------------------------------------------------------------------
Model::Model(const std::string& model_dir, const ModelOptions& options)
    : prefill_chunk_(options.prefill_chunk), max_context_(options.max_context) {
  impl_ = std::make_unique<Impl>();
  impl_->opt = options;
  impl_->load(model_dir);
}

Model::~Model() = default;

const ModelConfig& Model::config() const { return impl_->cfg; }
size_t Model::weight_bytes() const { return impl_->weight_bytes; }

const float* Model::forward(const std::vector<int64_t>& tokens, int pos0,
                            KVCache& kv, hipStream_t stream) {
  return impl_->forward(tokens, pos0, kv, stream);
}

}  // namespace ninfer

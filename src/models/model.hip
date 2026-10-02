#include "models/model.h"

#include <algorithm>
#include <cmath>
#include <set>
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

  int H, D, Hq, Hkv, V, I, layers;
  float eps, theta, attn_scale;
  int wqkv;

  // -------------------------------------------------------------------------
  // A Shard owns a contiguous layer range on ONE device (pipeline/layer
  // parallelism). Shard 0 owns the embedding; the last shard owns the final
  // norm + LM head. All buffers/caches of a shard live on its device.
  // -------------------------------------------------------------------------
  struct Shard {
    int device = 0;
    int layer_begin = 0, layer_end = 0;
    bool owns_embed = false, owns_head = false;

    std::unique_ptr<Blas> blas;
    std::unique_ptr<KVCache> kv;

    struct Layer {
      DeviceBuffer ln1, ln2;
      DeviceBuffer wqkv_fused;  // [Hq*D + 2*Hkv*D, H] — Q/K/V concatenated at
                                // load time so prefill issues ONE GEMM
      DeviceBuffer qn, kn;      // Qwen3 per-head q/k norms
      DeviceBuffer gu, down;    // dense FFN: gu = [gate; up] fused [2I, H]
      DeviceBuffer moe_gate, moe_gu, moe_down;  // MoE: gu per expert [E, 2I, H]
    };
    std::vector<Layer> lw;

    DeviceBuffer embed, final_norm, lm_head;

    // Activation workspace (sized for prefill_chunk tokens).
    DeviceBuffer hidden, normed, qkv, attn_out, proj, gate_up, act, logits_buf;
    DeviceBuffer ids_dev;
    // MoE workspace.
    DeviceBuffer moe_logits, moe_perm, moe_eout, moe_gate_up, moe_act;
    DeviceBuffer moe_order_dev, moe_w_dev;
    // Precomputed RoPE inverse-frequency table (D/2 fp32).
    DeviceBuffer rope_freq;

    size_t weight_bytes = 0;

    // ------------------------------------------------------------------
    // Weight loading: safetensors (single or sharded) -> FP16 device.
    // ------------------------------------------------------------------
    struct WeightSource {
      std::vector<std::unique_ptr<SafeTensorsFile>> files;
      std::map<std::string, SafeTensorsFile*> by_name;
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
        LOG_INFO("checkpoint: %d shard file(s)", static_cast<int>(files.size()));
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

    // Must be called with this shard's device current.
    DeviceBuffer load_fp16(const WeightSource& ws, const std::string& name,
                           bool account = true) {
      DeviceBuffer buf;
      auto it = ws.by_name.find(name);
      if (it == ws.by_name.end()) {
        throw std::runtime_error("missing tensor in checkpoint: " + name);
      }
      auto raw = it->second->read_raw(name);
      const StTensorInfo& info = it->second->tensor_info(name);
      auto halfs = convert_to_fp16(raw.data(), info.dtype, info.num_elements());
      buf.alloc(halfs.size() * sizeof(uint16_t));
      buf.copy_from_host(halfs.data(), buf.bytes());
      if (account) weight_bytes += buf.bytes();
      return buf;
    }

    void load(const std::string& dir, const ModelConfig& c, int dim_H, int dim_D,
              int dim_Hq, int dim_Hkv, int dim_V, int dim_I, float dim_eps,
              float rope_theta, int chunk, int max_ctx) {
      DeviceGuard guard(device);
      cfg_local = c;
      H = dim_H; D = dim_D; Hq = dim_Hq; Hkv = dim_Hkv; V = dim_V; I = dim_I;
      eps = dim_eps;

      LOG_INFO("shard dev%d: layers [%d, %d)%s%s", device, layer_begin,
               layer_end, owns_embed ? " +embed" : "", owns_head ? " +head" : "");
      WeightSource ws = open_weights(dir);

      if (owns_embed) {
        embed = load_fp16(ws, "model.embed_tokens.weight");
        ids_dev.alloc(static_cast<size_t>(chunk) * sizeof(int64_t));
      }
      if (owns_head) {
        final_norm = load_fp16(ws, "model.norm.weight");
        // Tied embeddings: the last shard may not own the embedding table,
        // so materialize the LM head explicitly in all cases.
        lm_head = cfg_local.tie_word_embeddings
                      ? load_fp16(ws, "model.embed_tokens.weight")
                      : load_fp16(ws, "lm_head.weight");
        logits_buf.alloc(static_cast<size_t>(V) * 4);
      }

      lw.resize(layer_end - layer_begin);
      for (int l = layer_begin; l < layer_end; ++l) {
        Layer& L = lw[l - layer_begin];
        std::string p = "model.layers." + std::to_string(l) + ".";
        L.ln1 = load_fp16(ws, p + "input_layernorm.weight");
        L.ln2 = load_fp16(ws, p + "post_attention_layernorm.weight");

        // Concatenate Q/K/V into one [wqkv, H] weight: three GEMM launches
        // per layer become one (prefill and decode both benefit).
        DeviceBuffer tq = load_fp16(ws, p + "self_attn.q_proj.weight", false);
        DeviceBuffer tk = load_fp16(ws, p + "self_attn.k_proj.weight", false);
        DeviceBuffer tv = load_fp16(ws, p + "self_attn.v_proj.weight", false);
        size_t rows = static_cast<size_t>(Hq + 2 * Hkv) * D;
        size_t bq = static_cast<size_t>(Hq) * D * H * 2;
        size_t bkv = static_cast<size_t>(Hkv) * D * H * 2;
        L.wqkv_fused.alloc(rows * H * 2);
        weight_bytes += L.wqkv_fused.bytes();
        HIP_CHECK(hipMemcpyAsync(L.wqkv_fused.data(), tq.data(), bq,
                                 hipMemcpyDeviceToDevice, 0));
        HIP_CHECK(hipMemcpyAsync(static_cast<char*>(L.wqkv_fused.data()) + bq,
                                 tk.data(), bkv, hipMemcpyDeviceToDevice, 0));
        HIP_CHECK(hipMemcpyAsync(
            static_cast<char*>(L.wqkv_fused.data()) + bq + bkv, tv.data(), bkv,
            hipMemcpyDeviceToDevice, 0));
        HIP_CHECK(hipStreamSynchronize(0));

        L.wo = load_fp16(ws, p + "self_attn.o_proj.weight");
        if (cfg_local.use_qk_norm) {
          L.qn = load_fp16(ws, p + "self_attn.q_norm.weight");
          L.kn = load_fp16(ws, p + "self_attn.k_norm.weight");
        }
        if (cfg_local.is_moe) {
          L.moe_gate = load_fp16(ws, p + "mlp.gate.weight");
          // Per-expert gate/up fusion: [E, I, H] + [E, I, H] -> [E, 2I, H] so
          // both prefill and decode issue ONE projection per expert.
          DeviceBuffer tg = load_fp16(ws, p + "mlp.experts.gate_proj.weight", false);
          DeviceBuffer tu = load_fp16(ws, p + "mlp.experts.up_proj.weight", false);
          int E = cfg_local.num_experts;
          size_t seg = static_cast<size_t>(I) * H * 2;
          L.moe_gu.alloc(static_cast<size_t>(E) * 2 * I * H * 2);
          weight_bytes += L.moe_gu.bytes();
          for (int e = 0; e < E; ++e) {
            HIP_CHECK(hipMemcpyAsync(
                static_cast<char*>(L.moe_gu.data()) + e * 2 * seg,
                static_cast<char*>(tg.data()) + e * seg, seg,
                hipMemcpyDeviceToDevice, 0));
            HIP_CHECK(hipMemcpyAsync(
                static_cast<char*>(L.moe_gu.data()) + e * 2 * seg + seg,
                static_cast<char*>(tu.data()) + e * seg, seg,
                hipMemcpyDeviceToDevice, 0));
          }
          HIP_CHECK(hipStreamSynchronize(0));
          L.moe_down = load_fp16(ws, p + "mlp.experts.down_proj.weight");
        } else {
          // Dense gate/up fusion: [I, H] + [I, H] -> [2I, H].
          DeviceBuffer tg = load_fp16(ws, p + "mlp.gate_proj.weight", false);
          DeviceBuffer tu = load_fp16(ws, p + "mlp.up_proj.weight", false);
          size_t seg = static_cast<size_t>(I) * H * 2;
          L.gu.alloc(seg * 2);
          weight_bytes += L.gu.bytes();
          HIP_CHECK(hipMemcpyAsync(L.gu.data(), tg.data(), seg,
                                   hipMemcpyDeviceToDevice, 0));
          HIP_CHECK(hipMemcpyAsync(static_cast<char*>(L.gu.data()) + seg,
                                   tu.data(), seg, hipMemcpyDeviceToDevice, 0));
          HIP_CHECK(hipStreamSynchronize(0));
          L.down = load_fp16(ws, p + "mlp.down_proj.weight");
        }
      }

      // Activation workspace.
      hidden.alloc(static_cast<size_t>(chunk) * H * 2);
      normed.alloc(static_cast<size_t>(chunk) * H * 2);
      qkv.alloc(static_cast<size_t>(chunk) * (Hq + 2 * Hkv) * D * 2);
      attn_out.alloc(static_cast<size_t>(chunk) * Hq * D * 2);
      proj.alloc(static_cast<size_t>(chunk) * H * 2);
      gate_up.alloc(static_cast<size_t>(chunk) * 2 * I * 2);
      act.alloc(static_cast<size_t>(chunk) * I * 2);
      // RoPE frequency table: inv_freq[i] = theta^(-2i/D), computed once.
      {
        std::vector<float> freqs(D / 2);
        for (int i = 0; i < D / 2; ++i) {
          freqs[i] = std::pow(rope_theta, -2.0f * i / D);
        }
        rope_freq.alloc(freqs.size() * sizeof(float));
        rope_freq.copy_from_host(freqs.data(), rope_freq.bytes());
      }
      if (cfg_local.is_moe) {
        size_t tsel = static_cast<size_t>(chunk) * cfg_local.num_experts_per_tok;
        moe_logits.alloc(static_cast<size_t>(chunk) * cfg_local.num_experts * 4);
        moe_perm.alloc(tsel * H * 2);
        moe_eout.alloc(tsel * H * 2);
        moe_gate_up.alloc(tsel * 2 * I * 2);
        moe_act.alloc(tsel * I * 2);
        moe_order_dev.alloc(tsel * sizeof(int64_t));
        moe_w_dev.alloc(tsel * sizeof(float));
      }
      kv = std::make_unique<KVCache>(layer_end - layer_begin, max_ctx, Hkv, D);
      blas = std::make_unique<Blas>();
      HIP_CHECK(hipDeviceSynchronize());
    }

    // Config mirror (avoid caching full cfg per shard otherwise).
    ModelConfig cfg_local;
    int H = 0, D = 0, Hq = 0, Hkv = 0, V = 0, I = 0;
    float eps = 1e-6f;

    // Runs this shard's layer range. `tokens` is only read when owns_embed.
    // Returns device logits if owns_head, else nullptr. Current device must
    // be `device` (handled by caller via DeviceGuard).
    const float* forward(const std::vector<int64_t>& tokens, int pos0) {
      hipStream_t stream = 0;
      int T = static_cast<int>(tokens.size());
      int chunk = static_cast<int>(hidden.bytes() / (H * 2));
      if (T > chunk) throw std::runtime_error("prefill chunk exceeds workspace");

      if (owns_embed) {
        HIP_CHECK(hipMemcpyAsync(ids_dev.data(), tokens.data(),
                                 T * sizeof(int64_t), hipMemcpyHostToDevice,
                                 stream));
        embed_gather_fp16(hidden.data(), embed.data(),
                          static_cast<const int64_t*>(ids_dev.data()), T, H,
                          stream);
      }

      int nlayers_total = cfg_local.num_hidden_layers;
      for (int li = 0; li < static_cast<int>(lw.size()); ++li) {
        const Layer& L = lw[li];
        int l = layer_begin + li;
        bool model_last = (l == nlayers_total - 1);

        rmsnorm_fp16(normed.data(), hidden.data(), L.ln1.data(), T, H, eps,
                     stream);

        const float attn_scale = 1.0f / std::sqrt(static_cast<float>(D));
        if (T == 1) {
          // ---- Decode fast path (M = 1): custom GEMV + fused post-QKV.
          // qkv GEMV, then ONE kernel for (optional qk-norm) + rope + KV
          // append; attention over the cache.
          gemv_fp16(qkv.data(), L.wqkv_fused.data(), normed.data(),
                    Hq * D + 2 * Hkv * D, H, stream);
          decode_post_qkv_fp16(
              qkv.data(), kv->k_data(li), kv->v_data(li), rope_freq.data(),
              pos0, Hq, Hkv, D, cfg_local.use_qk_norm,
              cfg_local.use_qk_norm ? L.qn.data() : nullptr,
              cfg_local.use_qk_norm ? L.kn.data() : nullptr, eps, stream);
          attention_decode_fp16(attn_out.data(),
                                static_cast<const __half*>(qkv.data()),
                                kv->k_data(li), kv->v_data(li), pos0 + 1, Hq,
                                Hkv, D, attn_scale, stream);
        } else {
          // ---- Prefill path: fused QKV GEMM + per-token kernels.
          blas->gemm_view(normed.data(), H, false, L.wqkv_fused.data(), H, true,
                          qkv.data(), Hq * D + 2 * Hkv * D,
                          Hq * D + 2 * Hkv * D, T, H, false, stream);
          __half* qsec_p = static_cast<__half*>(qkv.data());
          __half* ksec_p = qsec_p + Hq * D;
          if (cfg_local.use_qk_norm) {
            rmsnorm_heads_fp16(qsec_p, qsec_p, L.qn.data(), T, Hq, D,
                               Hq * D + 2 * Hkv * D, eps, stream);
            rmsnorm_heads_fp16(ksec_p, ksec_p, L.kn.data(), T, Hkv, D,
                               Hq * D + 2 * Hkv * D, eps, stream);
          }
          rope_fp16(qsec_p, T, Hq, D, Hq * D + 2 * Hkv * D, pos0,
                    rope_freq.data(), stream);
          rope_fp16(ksec_p, T, Hkv, D, Hq * D + 2 * Hkv * D, pos0,
                    rope_freq.data(), stream);
          kv->append(li, pos0, qsec_p, qsec_p, Hq * D, (Hq + Hkv) * D,
                     Hq * D + 2 * Hkv * D, T, stream);
          // All heads in one fused flash kernel (LDS double-buffered K/V
          // tiles, online softmax, no score scratch).
          attention_prefill_fused_fp16(
              attn_out.data(), qsec_p, kv->k_data(li), kv->v_data(li), T,
              pos0 + T, pos0, Hq, Hkv, D, Hq * D + 2 * Hkv * D, Hq * D,
              attn_scale, stream);
        }

        // o_proj + fused (residual add -> post-attention RMSNorm): the FFN
        // input is produced in the same pass that updates the residual
        // stream, saving one full [T, H] read+write per layer.
        if (T == 1) {
          gemv_fp16(proj.data(), L.wo.data(), attn_out.data(), H, Hq * D,
                    stream);
        } else {
          blas->gemm_view(attn_out.data(), Hq * D, false, L.wo.data(), H, true,
                          proj.data(), H, H, T, Hq * D, false, stream);
        }
        fused_residual_rmsnorm_fp16(hidden.data(), proj.data(), normed.data(),
                                    L.ln2.data(), T, H, eps, stream);

        if (!cfg_local.is_moe) {
          if (T == 1) {
            // Fused gate/up weight: one GEMV, then SwiGLU, then down GEMV.
            gemv_fp16(gate_up.data(), L.gu.data(), normed.data(), 2 * I, H,
                      stream);
            silu_and_mul_fp16(act.data(), gate_up.data(), 1, I, stream);
            gemv_fp16(proj.data(), L.down.data(), act.data(), H, I, stream);
          } else {
            blas->gemm_view(normed.data(), H, false, L.gu.data(), H, true,
                            gate_up.data(), 2 * I, 2 * I, T, H, false, stream);
            silu_and_mul_fp16(act.data(), gate_up.data(), T, I, stream);
            blas->gemm_view(act.data(), I, false, L.down.data(), I, true,
                            proj.data(), H, H, T, I, false, stream);
          }
        } else {
          int E = cfg_local.num_experts;
          int topk = cfg_local.num_experts_per_tok;
          if (T == 1) {
            gemv_fp32out(moe_logits.data(), L.moe_gate.data(), normed.data(), E,
                         H, stream);
          } else {
            blas->gemm_fp32out(normed.data(), L.moe_gate.data(),
                               moe_logits.data(), T, E, H, stream);
          }
          MoERoutingPlan plan = moe_route_cpu(
              static_cast<const float*>(moe_logits.data()), T, E, topk,
              *blas, stream);
          size_t tsel = plan.order.size();
          HIP_CHECK(hipMemcpyAsync(moe_order_dev.data(), plan.order.data(),
                                   tsel * sizeof(int64_t),
                                   hipMemcpyHostToDevice, stream));
          HIP_CHECK(hipMemcpyAsync(moe_w_dev.data(), plan.order_weights.data(),
                                   tsel * sizeof(float),
                                   hipMemcpyHostToDevice, stream));
          gather_rows_fp16(moe_perm.data(), normed.data(),
                           static_cast<const int64_t*>(moe_order_dev.data()),
                           static_cast<int>(tsel), H, stream);
          for (int e = 0; e < E; ++e) {
            int64_t off = plan.expert_offsets[e];
            int64_t n = plan.expert_offsets[e + 1] - off;
            if (n <= 0) continue;
            const __half* wgu = static_cast<const __half*>(L.moe_gu.data()) +
                                static_cast<int64_t>(e) * 2 * I * H;
            const __half* wd = static_cast<const __half*>(L.moe_down.data()) +
                               static_cast<int64_t>(e) * H * I;
            __half* in = static_cast<__half*>(moe_perm.data()) + off * H;
            __half* gu = static_cast<__half*>(moe_gate_up.data()) + off * 2 * I;
            if (T == 1) {
              gemv_fp16(gu, wgu, in, 2 * I, H, stream);
            } else {
              blas->gemm_view(in, H, false, wgu, H, true, gu, 2 * I, 2 * I,
                              static_cast<int>(n), H, false, stream);
            }
            silu_and_mul_fp16(static_cast<__half*>(moe_act.data()) + off * I,
                              gu, static_cast<int>(n), I, stream);
            __half* eo = static_cast<__half*>(moe_eout.data()) + off * H;
            if (T == 1) {
              gemv_fp16(eo, wd,
                        static_cast<const __half*>(moe_act.data()) + off * I, H,
                        I, stream);
            } else {
              blas->gemm_view(
                  static_cast<const __half*>(moe_act.data()) + off * I, I, false,
                  wd, I, true, eo, H, H, static_cast<int>(n), I, false, stream);
            }
          }
          scale_rows_fp16(moe_eout.data(),
                          static_cast<const float*>(moe_w_dev.data()),
                          static_cast<int>(tsel), H, stream);
          scatter_add_rows_fp16(
              hidden.data(), moe_eout.data(),
              static_cast<const int64_t*>(moe_order_dev.data()),
              static_cast<int>(tsel), H, stream);
        }

        if (model_last && owns_head) {
          // FFN residual, then final norm + LM head for the last row only.
          residual_add_fp16(hidden.data(), proj.data(), T, H, stream);
          __half* h_last = static_cast<__half*>(hidden.data()) + (T - 1) * H;
          __half* n_last = static_cast<__half*>(normed.data()) + (T - 1) * H;
          rmsnorm_fp16(n_last, h_last, final_norm.data(), 1, H, eps, stream);
          if (T == 1) {
            gemv_fp32out(logits_buf.data(), lm_head.data(), n_last, V, H,
                         stream);
          } else {
            blas->gemm_fp32out(n_last, lm_head.data(), logits_buf.data(), 1, V,
                               H, stream);
          }
          return static_cast<const float*>(logits_buf.data());
        }
        residual_add_fp16(hidden.data(), proj.data(), T, H, stream);
      }

      if (owns_head) {
        // Degenerate shard with zero layers but head ownership (single-layer
        // split edge case): norm the incoming hidden directly.
        rmsnorm_fp16(normed.data(), hidden.data(), final_norm.data(), 1, H, eps,
                     stream);
        blas->gemm_fp32out(normed.data(), lm_head.data(), logits_buf.data(), 1,
                           V, H, stream);
        return static_cast<const float*>(logits_buf.data());
      }
      return nullptr;
    }
  };

  std::vector<std::unique_ptr<Shard>> shards;

  ~Impl() {
    for (hipEvent_t e : boundary_events) (void)hipEventDestroy(e);
  }

  // -------------------------------------------------------------------------
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

    // Build the shard layout: contiguous layer ranges, one per device.
    std::vector<int> gpus;
    {
      std::set<int> uniq;
      for (int g : opt.gpu_ids) {
        if (uniq.insert(g).second) gpus.push_back(g);
      }
    }
    if (gpus.empty()) throw std::runtime_error("no GPUs requested");
    int n = static_cast<int>(gpus.size());
    if (n > 1 && layers < n) {
      throw std::runtime_error("cannot split " + std::to_string(layers) +
                               " layers across " + std::to_string(n) +
                               " GPUs");
    }
    LOG_INFO("loading weights from %s (%d GPU%s)...", dir.c_str(), n,
             n > 1 ? "s" : "");

    // Every requested device must be gfx906 (init_device only checks dev0).
    for (int g : gpus) {
      hipDeviceProp_t prop{};
      HIP_CHECK(hipGetDeviceProperties(&prop, g));
      if (std::string(prop.gcnArchName).find("gfx906") == std::string::npos) {
        throw std::runtime_error(std::string("device ") + std::to_string(g) +
                                 " ('" + prop.name + "', " + prop.gcnArchName +
                                 ") is not gfx906");
      }
    }

    for (int s = 0; s < n; ++s) {
      auto sh = std::make_unique<Shard>();
      sh->device = gpus[s];
      sh->layer_begin = layers * s / n;
      sh->layer_end = layers * (s + 1) / n;
      sh->owns_embed = (s == 0);
      sh->owns_head = (s == n - 1);
      // Guard during load so hipMalloc/hipMemcpy land on the right device.
      sh->load(dir, cfg, H, D, Hq, Hkv, V, I, eps, theta, opt.prefill_chunk,
               opt.max_context);
      shards.push_back(std::move(sh));
    }

    // Enable PCIe peer access where the platform allows it; hipMemcpyPeerAsync
    // transparently falls back to host staging otherwise.
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        DeviceGuard g(gpus[i]);
        hipError_t e = hipDeviceEnablePeerAccess(gpus[j]);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
          LOG_WARN("P2P dev%d->dev%d unavailable (%s); using staged copies",
                   gpus[i], gpus[j], hipGetErrorString(e));
          (void)hipGetLastError();  // clear the error state
        }
      }
    }

    size_t total = 0;
    for (auto& s : shards) total += s->weight_bytes;
    HIP_CHECK(hipDeviceSynchronize());
    LOG_INFO("weights resident: %.2f GiB across %d device(s)",
             static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0), n);

    // One boundary event per shard pair: recorded on the producer's stream
    // after its last kernel, awaited on the consumer's stream before the
    // cross-device copy — no host-side device drains.
    for (int i = 0; i + 1 < n; ++i) {
      HIP_CHECK(hipEventCreate(&boundary_events.emplace_back()));
    }
  }

  // Cross-shard activation transfer, ordered entirely by GPU events:
  //   producer stream --(event)--> consumer stream wait --> peer copy.
  // This removes the two hipDeviceSynchronize() drains per forward that the
  // naive pipeline needed, which dominated dual-GPU decode latency.
  std::vector<hipEvent_t> boundary_events;

  const float* forward(const std::vector<int64_t>& tokens, int pos0,
                       hipStream_t /*stream*/) {
    int T = static_cast<int>(tokens.size());
    const float* logits = nullptr;
    for (size_t s = 0; s < shards.size(); ++s) {
      Shard& cur = *shards[s];
      if (s > 0) {
        Shard& prev = *shards[s - 1];
        DeviceGuard g(cur.device);
        HIP_CHECK(hipStreamWaitEvent(0, boundary_events[s - 1], 0));
        HIP_CHECK(hipMemcpyPeerAsync(cur.hidden.data(), cur.device,
                                     prev.hidden.data(), prev.device,
                                     static_cast<size_t>(T) * H * 2, 0));
      }
      DeviceGuard g(cur.device);
      logits = cur.forward(tokens, pos0);
      if (s + 1 < shards.size()) {
        HIP_CHECK(hipEventRecord(boundary_events[s], 0));
      }
    }
    return logits;
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
int Model::device_count() const { return static_cast<int>(impl_->shards.size()); }
int Model::output_device() const { return impl_->shards.back()->device; }
void Model::reset_cache() {
  for (auto& s : impl_->shards) {
    DeviceGuard g(s->device);
    s->kv->reset();
  }
}
size_t Model::weight_bytes() const {
  size_t total = 0;
  for (auto& s : impl_->shards) total += s->weight_bytes;
  return total;
}

const float* Model::forward(const std::vector<int64_t>& tokens, int pos0,
                            hipStream_t stream) {
  return impl_->forward(tokens, pos0, stream);
}

}  // namespace ninfer

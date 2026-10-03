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
                                // load time so one GEMM/GEMV covers all three
      DeviceBuffer qn, kn;      // Qwen3 per-head q/k norms
      DeviceBuffer gu, down;    // dense FFN: gu = [gate; up] fused [2I, H]
      DeviceBuffer moe_gate, moe_gu, moe_down;  // MoE: gu per expert [E, 2I, H]
    };
    std::vector<Layer> lw;

    DeviceBuffer embed, final_norm, lm_head;

    // Activation workspace (sized for max(prefill_chunk, num_sequences) rows).
    DeviceBuffer hidden, normed, qkv, attn_out, proj, gate_up, act, logits_buf;
    DeviceBuffer ids_dev, pos_dev, seq_dev;
    // MoE workspace.
    DeviceBuffer moe_logits, moe_perm, moe_eout, moe_gate_up, moe_act;
    DeviceBuffer moe_order_dev, moe_w_dev;
    // Precomputed RoPE inverse-frequency table (D/2 fp32).
    DeviceBuffer rope_freq;

    size_t weight_bytes = 0;

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
              float rope_theta, int chunk, int max_ctx, int num_seq) {
      DeviceGuard guard(device);
      cfg_local = c;
      H = dim_H; D = dim_D; Hq = dim_Hq; Hkv = dim_Hkv; V = dim_V; I = dim_I;
      eps = dim_eps;

      LOG_INFO("shard dev%d: layers [%d, %d)%s%s", device, layer_begin,
               layer_end, owns_embed ? " +embed" : "", owns_head ? " +head" : "");
      WeightSource ws = open_weights(dir);

      int rows_max = std::max(chunk, num_seq);

      if (owns_embed) {
        embed = load_fp16(ws, "model.embed_tokens.weight");
        ids_dev.alloc(static_cast<size_t>(rows_max) * sizeof(int64_t));
      }
      pos_dev.alloc(static_cast<size_t>(rows_max) * sizeof(int));
      seq_dev.alloc(static_cast<size_t>(rows_max) * sizeof(int));
      if (owns_head) {
        final_norm = load_fp16(ws, "model.norm.weight");
        // Tied embeddings: the last shard may not own the embedding table,
        // so materialize the LM head explicitly in all cases.
        lm_head = cfg_local.tie_word_embeddings
                      ? load_fp16(ws, "model.embed_tokens.weight")
                      : load_fp16(ws, "lm_head.weight");
        // One logits row per concurrent sequence.
        logits_buf.alloc(static_cast<size_t>(num_seq) * V * 4);
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
          // Per-expert gate/up fusion: [E, I, H] + [E, I, H] -> [E, 2I, H].
          DeviceBuffer tg =
              load_fp16(ws, p + "mlp.experts.gate_proj.weight", false);
          DeviceBuffer tu =
              load_fp16(ws, p + "mlp.experts.up_proj.weight", false);
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
      hidden.alloc(static_cast<size_t>(rows_max) * H * 2);
      normed.alloc(static_cast<size_t>(rows_max) * H * 2);
      qkv.alloc(static_cast<size_t>(rows_max) * (Hq + 2 * Hkv) * D * 2);
      attn_out.alloc(static_cast<size_t>(rows_max) * Hq * D * 2);
      proj.alloc(static_cast<size_t>(rows_max) * H * 2);
      gate_up.alloc(static_cast<size_t>(rows_max) * 2 * I * 2);
      act.alloc(static_cast<size_t>(rows_max) * I * 2);
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
        size_t tsel = static_cast<size_t>(rows_max) *
                      cfg_local.num_experts_per_tok;
        moe_logits.alloc(static_cast<size_t>(rows_max) * cfg_local.num_experts * 4);
        moe_perm.alloc(tsel * H * 2);
        moe_eout.alloc(tsel * H * 2);
        moe_gate_up.alloc(tsel * 2 * I * 2);
        moe_act.alloc(tsel * I * 2);
        moe_order_dev.alloc(tsel * sizeof(int64_t));
        moe_w_dev.alloc(tsel * sizeof(float));
      }
      kv = std::make_unique<KVCache>(layer_end - layer_begin, max_ctx, Hkv, D,
                                     num_seq);
      blas = std::make_unique<Blas>();
      HIP_CHECK(hipDeviceSynchronize());
    }

    ModelConfig cfg_local;
    int H = 0, D = 0, Hq = 0, Hkv = 0, V = 0, I = 0;
    float eps = 1e-6f;

    // Runs this shard's layer range over `rows` activation rows. Each row
    // belongs to sequence `seq_ids[r]` at absolute position `positions[r]`;
    // rows of one sequence form a group that shares its KV slab.
    ForwardOutput forward_batch(const std::vector<int64_t>& tokens,
                                const std::vector<int>& positions,
                                const std::vector<int>& seq_ids) {
      hipStream_t stream = 0;
      int T = static_cast<int>(tokens.size());
      int rows_max = static_cast<int>(hidden.bytes() / (H * 2));
      if (T > rows_max) throw std::runtime_error("batch exceeds workspace rows");
      if (T != static_cast<int>(positions.size()) ||
          T != static_cast<int>(seq_ids.size())) {
        throw std::runtime_error("forward_batch: mismatched row descriptors");
      }

      if (owns_embed) {
        HIP_CHECK(hipMemcpyAsync(ids_dev.data(), tokens.data(),
                                 T * sizeof(int64_t), hipMemcpyHostToDevice,
                                 stream));
        embed_gather_fp16(hidden.data(), embed.data(),
                          static_cast<const int64_t*>(ids_dev.data()), T, H,
                          stream);
      }
      HIP_CHECK(hipMemcpyAsync(pos_dev.data(), positions.data(),
                               T * sizeof(int), hipMemcpyHostToDevice, stream));
      HIP_CHECK(hipMemcpyAsync(seq_dev.data(), seq_ids.data(),
                               T * sizeof(int), hipMemcpyHostToDevice, stream));

      // Group contiguous rows by sequence (one group == one request).
      struct Group { int seq, start, count, pos0; };
      std::vector<Group> groups;
      for (int i = 0; i < T;) {
        int j = i;
        while (j < T && seq_ids[j] == seq_ids[i]) ++j;
        groups.push_back({seq_ids[i], i, j - i, positions[i]});
        i = j;
      }
      bool all_single = true;
      for (const Group& g : groups) {
        if (g.count != 1) all_single = false;
      }

      const int wqkv = Hq * D + 2 * Hkv * D;
      const float attn_scale_ = 1.0f / std::sqrt(static_cast<float>(D));

      for (int li = 0; li < static_cast<int>(lw.size()); ++li) {
        const Layer& L = lw[li];

        rmsnorm_fp16(normed.data(), hidden.data(), L.ln1.data(), T, H, eps,
                     stream);

        // ---- QKV projection.
        if (T == 1) {
          gemv_fp16(qkv.data(), L.wqkv_fused.data(), normed.data(), wqkv, H,
                    stream);
          // Fused qk-norm + rope + KV append for the single row.
          decode_post_qkv_fp16(
              qkv.data(), kv->k_data(li, groups[0].seq),
              kv->v_data(li, groups[0].seq), rope_freq.data(), groups[0].pos0,
              Hq, Hkv, D, cfg_local.use_qk_norm,
              cfg_local.use_qk_norm ? L.qn.data() : nullptr,
              cfg_local.use_qk_norm ? L.kn.data() : nullptr, eps, stream);
        } else {
          blas->gemm_view(normed.data(), H, false, L.wqkv_fused.data(), H, true,
                          qkv.data(), wqkv, wqkv, T, H, false, stream);
          if (all_single) {
            // Batched decode: one fused kernel for qk-norm + rope + append.
            decode_post_qkv_batch_fp16(
                qkv.data(), kv->k_base(li), kv->v_base(li), pos_dev.data(),
                seq_dev.data(), T, Hq, Hkv, D, wqkv, kv->seq_stride_elems(),
                rope_freq.data(), cfg_local.use_qk_norm,
                cfg_local.use_qk_norm ? L.qn.data() : nullptr,
                cfg_local.use_qk_norm ? L.kn.data() : nullptr, eps, stream);
          } else {
            __half* qs = static_cast<__half*>(qkv.data());
            __half* ks = qs + Hq * D;
            if (cfg_local.use_qk_norm) {
              rmsnorm_heads_fp16(qs, qs, L.qn.data(), T, Hq, D, wqkv, eps,
                                 stream);
              rmsnorm_heads_fp16(ks, ks, L.kn.data(), T, Hkv, D, wqkv, eps,
                                 stream);
            }
            rope_fp16(qs, T, Hq, D, wqkv, pos_dev.data(), rope_freq.data(),
                      stream);
            rope_fp16(ks, T, Hkv, D, wqkv, pos_dev.data(), rope_freq.data(),
                      stream);
            for (const Group& g : groups) {
              kv->append(li, g.seq, g.pos0, qs, qs, g.start, Hq * D,
                         (Hq + Hkv) * D, wqkv, g.count, stream);
            }
          }
        }

        // ---- Attention (per group: each group owns its KV slab).
        for (const Group& g : groups) {
          __half* q_row = static_cast<__half*>(qkv.data()) +
                          static_cast<int64_t>(g.start) * wqkv;
          __half* out_row = static_cast<__half*>(attn_out.data()) +
                            static_cast<int64_t>(g.start) * Hq * D;
          if (g.count == 1) {
            attention_decode_fp16(out_row, q_row, kv->k_data(li, g.seq),
                                  kv->v_data(li, g.seq), g.pos0 + 1, Hq, Hkv, D,
                                  attn_scale_, stream);
          } else {
            attention_prefill_fused_fp16(
                out_row, q_row, kv->k_data(li, g.seq), kv->v_data(li, g.seq),
                g.count, g.pos0 + g.count, g.pos0, Hq, Hkv, D, wqkv, Hq * D,
                attn_scale_, stream);
          }
        }

        // ---- o_proj + fused (residual add -> post-attention RMSNorm).
        if (T == 1) {
          gemv_fp16(proj.data(), L.wo.data(), attn_out.data(), H, Hq * D,
                    stream);
        } else {
          blas->gemm_view(attn_out.data(), Hq * D, false, L.wo.data(), H, true,
                          proj.data(), H, H, T, Hq * D, false, stream);
        }
        fused_residual_rmsnorm_fp16(hidden.data(), proj.data(), normed.data(),
                                    L.ln2.data(), T, H, eps, stream);

        // ---- FFN.
        if (!cfg_local.is_moe) {
          if (T == 1) {
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
              static_cast<const float*>(moe_logits.data()), T, E, topk, *blas,
              stream);
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

        residual_add_fp16(hidden.data(), proj.data(), T, H, stream);
      }

      // ---- Logits: last row of each sequence group, only on the head shard.
      ForwardOutput out;
      out.rows = static_cast<int>(groups.size());
      for (const Group& g : groups) out.group_seq.push_back(g.seq);
      if (owns_head) {
        out.logits = static_cast<const float*>(logits_buf.data());
        for (int gi = 0; gi < out.rows; ++gi) {
          const Group& g = groups[gi];
          int r = g.start + g.count - 1;
          __half* h_last = static_cast<__half*>(hidden.data()) +
                           static_cast<int64_t>(r) * H;
          __half* n_row = static_cast<__half*>(normed.data()) +
                          static_cast<int64_t>(gi) * H;
          rmsnorm_fp16(n_row, h_last, final_norm.data(), 1, H, eps, stream);
          float* logits_row = static_cast<float*>(logits_buf.data()) +
                              static_cast<int64_t>(gi) * V;
          if (T == 1) {
            gemv_fp32out(logits_row, lm_head.data(), n_row, V, H, stream);
          } else {
            blas->gemm_view(n_row, H, false, lm_head.data(), H, true,
                            logits_row, V, V, 1, H, /*c_fp32=*/true, stream);
          }
        }
      }
      return out;
    }
  };

  std::vector<std::unique_ptr<Shard>> shards;

  ~Impl() {
    for (hipEvent_t e : boundary_events) (void)hipEventDestroy(e);
  }

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
                               " layers across " + std::to_string(n) + " GPUs");
    }
    LOG_INFO("loading weights from %s (%d GPU%s, %d sequence slots)...",
             dir.c_str(), n, n > 1 ? "s" : "", opt.num_sequences);

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
      sh->load(dir, cfg, H, D, Hq, Hkv, V, I, eps, theta, opt.prefill_chunk,
               opt.max_context, opt.num_sequences);
      shards.push_back(std::move(sh));
    }

    // PCIe peer access where the platform allows it; hipMemcpyPeerAsync
    // falls back to host staging otherwise.
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        if (i == j) continue;
        DeviceGuard g(gpus[i]);
        hipError_t e = hipDeviceEnablePeerAccess(gpus[j]);
        if (e != hipSuccess && e != hipErrorPeerAccessAlreadyEnabled) {
          LOG_WARN("P2P dev%d->dev%d unavailable (%s); using staged copies",
                   gpus[i], gpus[j], hipGetErrorString(e));
          (void)hipGetLastError();
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
      hipEvent_t ev;
      HIP_CHECK(hipEventCreate(&ev));
      boundary_events.push_back(ev);
    }
  }

  std::vector<hipEvent_t> boundary_events;

  ForwardOutput forward_batch(const std::vector<int64_t>& tokens,
                              const std::vector<int>& positions,
                              const std::vector<int>& seq_ids,
                              hipStream_t /*stream*/) {
    int T = static_cast<int>(tokens.size());
    ForwardOutput out;
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
      out = cur.forward_batch(tokens, positions, seq_ids);
      if (s + 1 < shards.size()) {
        HIP_CHECK(hipEventRecord(boundary_events[s], 0));
      }
    }
    return out;
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
int Model::num_sequences() const { return impl_->opt.num_sequences; }
int Model::output_device() const { return impl_->shards.back()->device; }

void Model::reset_cache() {
  for (auto& s : impl_->shards) {
    DeviceGuard g(s->device);
    s->kv->reset();
  }
}

void Model::reset_cache(int seq) {
  for (auto& s : impl_->shards) {
    DeviceGuard g(s->device);
    for (int l = 0; l < s->kv->layers(); ++l) {
      // Slot reuse: only the logical length is tracked per sequence.
      s->kv->set_length(seq, 0);
    }
  }
}

size_t Model::weight_bytes() const {
  size_t total = 0;
  for (auto& s : impl_->shards) total += s->weight_bytes;
  return total;
}

ForwardOutput Model::forward_batch(const std::vector<int64_t>& tokens,
                                   const std::vector<int>& positions,
                                   const std::vector<int>& seq_ids,
                                   hipStream_t stream) {
  return impl_->forward_batch(tokens, positions, seq_ids, stream);
}

const float* Model::forward(const std::vector<int64_t>& tokens, int pos0,
                            hipStream_t stream) {
  std::vector<int> positions(tokens.size()), seq_ids(tokens.size(), 0);
  for (size_t i = 0; i < tokens.size(); ++i) {
    positions[i] = pos0 + static_cast<int>(i);
  }
  ForwardOutput out = impl_->forward_batch(tokens, positions, seq_ids, stream);
  return out.logits;
}

}  // namespace ninfer

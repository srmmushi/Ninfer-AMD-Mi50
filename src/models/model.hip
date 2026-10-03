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
  float eps, theta;
  bool use_q4 = false;
  bool use_tp = false;

  // One row-group == one request inside a forward_batch call.
  struct Group { int seq, start, count, pos0; };

  // -------------------------------------------------------------------------
  // Weight matrix: fp16 blob, or INT4 packed nibbles + per-group scales.
  // -------------------------------------------------------------------------
  struct Mat {
    DeviceBuffer fp;      // [rows, cols] fp16
    DeviceBuffer qp, qs;  // [rows*cols/2] u8, [rows*cols/32] fp16
    int rows = 0, cols = 0;
    bool is_q4 = false;
    size_t bytes() const { return is_q4 ? qp.bytes() + qs.bytes() : fp.bytes(); }
  };

  // -------------------------------------------------------------------------
  // A Shard owns either a layer range (pp) or a tensor-parallel slice of all
  // layers (tp). All of its buffers live on `device`.
  // -------------------------------------------------------------------------
  struct Shard {
    int device = 0;
    int layer_begin = 0, layer_end = 0;
    bool owns_embed = false, owns_head = false;
    int tp_rank = 0, tp_world = 1;

    std::unique_ptr<Blas> blas;
    std::unique_ptr<KVCache> kv;

    struct Layer {
      Mat ln1, ln2;
      Mat wqkv_fused;                 // pp: [Hq*D + 2*Hkv*D, H]
      Mat wq, wk, wv;                 // tp: head-split slices
      Mat wo;                         // tp: column-split [H, Hq*D/tp]
      Mat qn, kn;
      Mat gu, down;                   // down is column-split in tp
      Mat moe_gate, moe_gu, moe_down;
    };
    std::vector<Layer> lw;

    Mat embed, final_norm, lm_head;

    // Activation workspace.
    DeviceBuffer hidden, normed, qkv, attn_out, proj, gate_up, act, logits_buf;
    DeviceBuffer ids_dev, pos_dev, seq_dev;
    DeviceBuffer moe_logits, moe_perm, moe_eout, moe_gate_up, moe_act;
    DeviceBuffer moe_order_dev, moe_w_dev;
    DeviceBuffer rope_freq;
    // Tensor-parallel scratch: peer partial + one event per rank.
    DeviceBuffer peer_buf;

    size_t weight_bytes = 0;
    int H_local = 0, Hq_local = 0, Hkv_local = 0;  // per-rank head counts (tp)
    int D = 0, Hdim = 0, Iloc = 0;

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

    DeviceBuffer load_fp16(const WeightSource& ws, const std::string& name) {
      auto it = ws.by_name.find(name);
      if (it == ws.by_name.end()) {
        throw std::runtime_error("missing tensor in checkpoint: " + name);
      }
      auto raw = it->second->read_raw(name);
      const StTensorInfo& info = it->second->tensor_info(name);
      auto halfs = convert_to_fp16(raw.data(), info.dtype, info.num_elements());
      DeviceBuffer buf;
      buf.alloc(halfs.size() * sizeof(uint16_t));
      buf.copy_from_host(halfs.data(), buf.bytes());
      return buf;
    }

    // Loads a tensor and keeps only [row_off, row_off+row_count) x
    // [col_off, col_off+col_count) — the tensor-parallel weight split.
    Mat load_mat(const WeightSource& ws, const std::string& name, int full_rows,
                 int full_cols, int row_off, int row_count, int col_off,
                 int col_count, bool use_q4, bool account = true) {
      DeviceBuffer tmp = load_fp16(ws, name);
      DeviceBuffer slice;
      if (col_off == 0 && col_count == full_cols && row_count == full_rows &&
          row_off == 0) {
        slice = std::move(tmp);
      } else if (col_off == 0 && col_count == full_cols) {
        slice.alloc(static_cast<size_t>(row_count) * col_count * 2);
        HIP_CHECK(hipMemcpyAsync(
            slice.data(),
            static_cast<const char*>(tmp.data()) +
                static_cast<int64_t>(row_off) * full_cols * 2,
            static_cast<size_t>(row_count) * col_count * 2,
            hipMemcpyDeviceToDevice, 0));
      } else {
        slice.alloc(static_cast<size_t>(row_count) * col_count * 2);
        slice_cols_fp16(slice.data(),
                        static_cast<const char*>(tmp.data()) +
                            static_cast<int64_t>(row_off) * full_cols * 2,
                        row_count, col_count, full_cols, col_off, 0);
      }
      HIP_CHECK(hipStreamSynchronize(0));

      Mat m;
      m.rows = row_count;
      m.cols = col_count;
      if (use_q4) {
        size_t groups = static_cast<size_t>(row_count) *
                        (col_count / kQ4Group);
        m.qp.alloc(static_cast<size_t>(row_count) * col_count / 2);
        m.qs.alloc(groups * sizeof(uint16_t));
        quantize_q4_fp16(slice.data(), m.qp.data(), m.qs.data(), row_count,
                         col_count, 0);
        HIP_CHECK(hipStreamSynchronize(0));
        m.is_q4 = true;
      } else {
        m.fp = std::move(slice);
      }
      if (account) weight_bytes += m.bytes();
      return m;
    }

    // ---- linear dispatch: fp16 vs q4, GEMV (M=1) vs GEMM (M>1) ----
    void linear(void* out, const Mat& w, const void* x, int M, bool fp32_out) {
      if (w.is_q4) {
        if (M == 1) {
          if (fp32_out) {
            gemv_q4_fp32out(out, w.qp.data(), w.qs.data(), x, w.rows, w.cols, 0);
          } else {
            gemv_q4_fp16(out, w.qp.data(), w.qs.data(), x, w.rows, w.cols, 0);
          }
          return;
        }
        gemm_q4_fp16(out, x, w.qp.data(), w.qs.data(), M, w.rows, w.cols, 0);
        return;
      }
      if (M == 1) {
        if (fp32_out) {
          gemv_fp32out(out, w.fp.data(), x, w.rows, w.cols, 0);
        } else {
          gemv_fp16(out, w.fp.data(), x, w.rows, w.cols, 0);
        }
        return;
      }
      if (fp32_out) {
        blas->gemm_fp32out(x, w.fp.data(), out, M, w.rows, w.cols, 0);
      } else {
        blas->gemm_view(x, w.cols, false, w.fp.data(), w.cols, true, out,
                        w.rows, w.rows, M, w.cols, false, 0);
      }
    }

    // ---- layer stages (split so tp can interleave all-reduce) ----
    void layer_start(int li, const std::vector<Group>& groups, int T,
                     bool all_single, bool tp_mode) {
      hipStream_t stream = 0;
      const Layer& L = lw[li];
      int Hql = Hq_local, Hkvl = Hkv_local;
      int qkv_w = (Hql + 2 * Hkvl) * D;

      rmsnorm_fp16(normed.data(), hidden.data(), L.ln1.fp.data(), T, Hdim, eps,
                   stream);

      if (tp_mode) {
        linear(qkv.data(), L.wq, normed.data(), T, false);
        linear(static_cast<char*>(qkv.data()) + Hql * D * 2, L.wk,
               normed.data(), T, false);
        linear(static_cast<char*>(qkv.data()) + (Hql + Hkvl) * D * 2, L.wv,
               normed.data(), T, false);
      } else if (T == 1) {
        linear(qkv.data(), L.wqkv_fused, normed.data(), 1, false);
        decode_post_qkv_fp16(
            qkv.data(), kv->k_data(li, groups[0].seq),
            kv->v_data(li, groups[0].seq), rope_freq.data(), groups[0].pos0,
            Hql, Hkvl, D, use_qknorm, L.qn.fp.data(), L.kn.fp.data(), eps,
            stream);
      } else {
        linear(qkv.data(), L.wqkv_fused, normed.data(), T, false);
        if (all_single) {
          decode_post_qkv_batch_fp16(
              qkv.data(), kv->k_base(li), kv->v_base(li), pos_dev.data(),
              seq_dev.data(), T, Hql, Hkvl, D, qkv_w,
              kv->seq_stride_elems(), rope_freq.data(), use_qknorm,
              L.qn.fp.data(), L.kn.fp.data(), eps, stream);
        } else {
          __half* qs = static_cast<__half*>(qkv.data());
          __half* ks = qs + Hql * D;
          if (use_qknorm) {
            rmsnorm_heads_fp16(qs, qs, L.qn.fp.data(), T, Hql, D, qkv_w, eps,
                               stream);
            rmsnorm_heads_fp16(ks, ks, L.kn.fp.data(), T, Hkvl, D, qkv_w, eps,
                               stream);
          }
          rope_fp16(qs, T, Hql, D, qkv_w, pos_dev.data(), rope_freq.data(),
                    stream);
          rope_fp16(ks, T, Hkvl, D, qkv_w, pos_dev.data(), rope_freq.data(),
                    stream);
          for (const Group& g : groups) {
            kv->append(li, g.seq, g.pos0, qs, qs, g.start, Hql * D,
                       (Hql + Hkvl) * D, qkv_w, g.count, stream);
          }
        }
      }

      bool tp_decode_single = tp_mode && T == 1;
      if (tp_decode_single) {
        decode_post_qkv_fp16(qkv.data(), kv->k_data(li, groups[0].seq),
                             kv->v_data(li, groups[0].seq), rope_freq.data(),
                             groups[0].pos0, Hql, Hkvl, D, use_qknorm,
                             L.qn.fp.data(), L.kn.fp.data(), eps, stream);
      } else if (tp_mode) {
        if (all_single) {
          decode_post_qkv_batch_fp16(
              qkv.data(), kv->k_base(li), kv->v_base(li), pos_dev.data(),
              seq_dev.data(), T, Hql, Hkvl, D, qkv_w, kv->seq_stride_elems(),
              rope_freq.data(), use_qknorm, L.qn.fp.data(), L.kn.fp.data(),
              eps, stream);
        } else {
          __half* qs = static_cast<__half*>(qkv.data());
          __half* ks = qs + Hql * D;
          rope_fp16(qs, T, Hql, D, qkv_w, pos_dev.data(), rope_freq.data(),
                    stream);
          rope_fp16(ks, T, Hkvl, D, qkv_w, pos_dev.data(), rope_freq.data(),
                    stream);
          for (const Group& g : groups) {
            kv->append(li, g.seq, g.pos0, qs, qs, g.start, Hql * D,
                       (Hql + Hkvl) * D, qkv_w, g.count, stream);
          }
        }
      }

      // Attention, per group (each group owns its KV slab).
      const float scale = 1.0f / std::sqrt(static_cast<float>(D));
      for (const Group& g : groups) {
        __half* q_row = static_cast<__half*>(qkv.data()) +
                        static_cast<int64_t>(g.start) * qkv_w;
        __half* out_row = static_cast<__half*>(attn_out.data()) +
                          static_cast<int64_t>(g.start) * Hql * D;
        if (g.count == 1) {
          attention_decode_fp16(out_row, q_row, kv->k_data(li, g.seq),
                                kv->v_data(li, g.seq), g.pos0 + 1, Hql, Hkvl, D,
                                scale, stream);
        } else {
          attention_prefill_fused_fp16(out_row, q_row, kv->k_data(li, g.seq),
                                       kv->v_data(li, g.seq), g.count,
                                       g.pos0 + g.count, g.pos0, Hql, Hkvl, D,
                                       qkv_w, Hql * D, scale, stream);
        }
      }

      // o_proj -> proj (partial in tp mode: Wo is column-split).
      linear(proj.data(), L.wo, attn_out.data(), T, false);
    }

    void layer_mid(int li, int T) {
      hipStream_t stream = 0;
      const Layer& L = lw[li];
      fused_residual_rmsnorm_fp16(hidden.data(), proj.data(), normed.data(),
                                  L.ln2.fp.data(), T, Hdim, eps, stream);
      if (!is_moe) {
        if (T == 1 && tp_world == 1) {
          // Decode fast path: gate/up GEMV with SwiGLU in the epilogue (one
          // kernel instead of GEMV + silu_and_mul), then down GEMV with the
          // residual add in the epilogue (one kernel instead of GEMV + add).
          if (L.gu.is_q4) {
            gemv_swiglu_q4_fp16(act.data(), L.gu.qp.data(), L.gu.qs.data(),
                                normed.data(), Iloc, Hdim, stream);
          } else {
            gemv_swiglu_fp16(act.data(), L.gu.fp.data(), normed.data(), Iloc,
                             Hdim, stream);
          }
          if (L.down.is_q4) {
            gemv_residual_q4_fp16(hidden.data(), L.down.qp.data(),
                                  L.down.qs.data(), act.data(), Hdim, Iloc,
                                  stream);
          } else {
            gemv_residual_fp16(hidden.data(), L.down.fp.data(), act.data(),
                               Hdim, Iloc, stream);
          }
        } else {
          linear(gate_up.data(), L.gu, normed.data(), T, false);
          silu_and_mul_fp16(act.data(), gate_up.data(), T, Iloc, stream);
          linear(proj.data(), L.down, act.data(), T, false);
        }
      } else {
        int E = num_experts;
        int topk = top_k_experts;
        linear(moe_logits.data(), L.moe_gate, normed.data(), T, true);
        MoERoutingPlan plan = moe_route_cpu(
            static_cast<const float*>(moe_logits.data()), T, E, topk, *blas,
            stream);
        size_t tsel = plan.order.size();
        HIP_CHECK(hipMemcpyAsync(moe_order_dev.data(), plan.order.data(),
                                 tsel * sizeof(int64_t), hipMemcpyHostToDevice,
                                 stream));
        HIP_CHECK(hipMemcpyAsync(moe_w_dev.data(), plan.order_weights.data(),
                                 tsel * sizeof(float), hipMemcpyHostToDevice,
                                 stream));
        gather_rows_fp16(moe_perm.data(), normed.data(),
                         static_cast<const int64_t*>(moe_order_dev.data()),
                         static_cast<int>(tsel), Hdim, stream);
        for (int e = 0; e < E; ++e) {
          int64_t off = plan.expert_offsets[e];
          int64_t n = plan.expert_offsets[e + 1] - off;
          if (n <= 0) continue;
          __half* in = static_cast<__half*>(moe_perm.data()) + off * Hdim;
          __half* gu = static_cast<__half*>(moe_gate_up.data()) + off * 2 * Iloc;
          __half* eo = static_cast<__half*>(moe_eout.data()) + off * Hdim;
          if (w_is_q4) {
            // Per-expert slices: view the stacked [E, rows, cols] blob.
            int64_t gstride = static_cast<int64_t>(2 * Iloc) * Hdim / 2;
            int64_t sstride = static_cast<int64_t>(2 * Iloc) *
                              (Hdim / kQ4Group);
            int64_t dstride = static_cast<int64_t>(Hdim) * Iloc / 2;
            int64_t dss = static_cast<int64_t>(Hdim) * (Iloc / kQ4Group);
            if (T == 1) {
              gemv_q4_fp16(gu,
                           static_cast<const char*>(L.moe_gu.qp.data()) +
                               e * gstride,
                           static_cast<const char*>(L.moe_gu.qs.data()) +
                               e * sstride,
                           in, 2 * Iloc, Hdim, stream);
            } else {
              gemm_q4_fp16(gu, in,
                           static_cast<const char*>(L.moe_gu.qp.data()) +
                               e * gstride,
                           static_cast<const char*>(L.moe_gu.qs.data()) +
                               e * sstride,
                           static_cast<int>(n), 2 * Iloc, Hdim, stream);
            }
            silu_and_mul_fp16(static_cast<__half*>(moe_act.data()) + off * Iloc,
                              gu, static_cast<int>(n), Iloc, stream);
            if (T == 1) {
              gemv_q4_fp16(eo,
                           static_cast<const char*>(L.moe_down.qp.data()) +
                               e * dstride,
                           static_cast<const char*>(L.moe_down.qs.data()) +
                               e * dss,
                           static_cast<const __half*>(moe_act.data()) +
                               off * Iloc,
                           Hdim, Iloc, stream);
            } else {
              gemm_q4_fp16(eo,
                           static_cast<const __half*>(moe_act.data()) +
                               off * Iloc,
                           static_cast<const char*>(L.moe_down.qp.data()) +
                               e * dstride,
                           static_cast<const char*>(L.moe_down.qs.data()) +
                               e * dss,
                           static_cast<int>(n), Hdim, Iloc, stream);
            }
          } else {
            const __half* wgu = static_cast<const __half*>(L.moe_gu.fp.data()) +
                                static_cast<int64_t>(e) * 2 * Iloc * Hdim;
            const __half* wd = static_cast<const __half*>(L.moe_down.fp.data()) +
                               static_cast<int64_t>(e) * Hdim * Iloc;
            if (T == 1) {
              gemv_fp16(gu, wgu, in, 2 * Iloc, Hdim, stream);
            } else {
              blas->gemm_view(in, Hdim, false, wgu, Hdim, true, gu, 2 * Iloc,
                              2 * Iloc, static_cast<int>(n), Hdim, false, stream);
            }
            silu_and_mul_fp16(static_cast<__half*>(moe_act.data()) + off * Iloc,
                              gu, static_cast<int>(n), Iloc, stream);
            const __half* act_row =
                static_cast<const __half*>(moe_act.data()) + off * Iloc;
            if (T == 1) {
              gemv_fp16(eo, wd, act_row, Hdim, Iloc, stream);
            } else {
              blas->gemm_view(act_row, Iloc, false, wd, Iloc, true, eo, Hdim,
                              Hdim, static_cast<int>(n), Iloc, false, stream);
            }
          }
        }
        scale_rows_fp16(moe_eout.data(),
                        static_cast<const float*>(moe_w_dev.data()),
                        static_cast<int>(tsel), Hdim, stream);
        scatter_add_rows_fp16(hidden.data(), moe_eout.data(),
                              static_cast<const int64_t*>(moe_order_dev.data()),
                              static_cast<int>(tsel), Hdim, stream);
      }
    }

    // The FFN residual is fused into the down GEMV only for single-GPU decode
    // of dense models (in tp mode `down` is column-split, so its partial
    // output must be all-reduced before it may touch the residual stream).
    void layer_end(int T) {
      bool fused = (T == 1 && !is_moe && tp_world == 1);
      if (fused) return;
      residual_add_fp16(hidden.data(), proj.data(), T, Hdim, 0);
    }

    ModelConfig cfg_local;
    int Hdim2 = 0;
    float eps = 1e-6f;
    bool use_qknorm = false, is_moe = false, w_is_q4 = false;
    int num_experts = 0, top_k_experts = 0;

    void load(const std::string& dir, const ModelConfig& c, int dim_H, int dim_D,
              int dim_Hq, int dim_Hkv, int dim_V, int dim_I, float dim_eps,
              float rope_theta, int chunk, int max_ctx, int num_seq,
              bool q4_mode, bool tp_mode, int tp_rank_, int tp_world_,
              int layer_begin_, int layer_end_, bool owns_embed_,
              bool owns_head_) {
      DeviceGuard guard(device);
      cfg_local = c;
      D = dim_D; Hdim = dim_H;
      tp_rank = tp_rank_; tp_world = tp_world_;
      layer_begin = layer_begin_; layer_end = layer_end_;
      owns_embed = owns_embed_; owns_head = owns_head_;
      eps = dim_eps;
      use_qknorm = c.use_qk_norm;
      is_moe = c.is_moe;
      w_is_q4 = q4_mode;
      num_experts = c.num_experts;
      top_k_experts = c.num_experts_per_tok;

      // Per-rank slices (tensor parallel) or full widths (pipeline).
      Hq_local = tp_mode ? dim_Hq / tp_world : dim_Hq;
      Hkv_local = tp_mode ? dim_Hkv / tp_world : dim_Hkv;
      int Iloc_full = dim_I;
      Iloc = tp_mode ? dim_I / tp_world : dim_I;

      LOG_INFO("shard dev%d%s: layers [%d, %d)%s%s%s", device,
               tp_mode ? (std::string(" tp") + std::to_string(tp_rank)).c_str()
                       : "",
               layer_begin, layer_end, owns_embed ? " +embed" : "",
               owns_head ? " +head" : "", q4_mode ? " q4" : " fp16");

      WeightSource ws = open_weights(dir);
      int rows_max = std::max(chunk, num_seq);

      if (owns_embed) {
        embed = load_mat(ws, "model.embed_tokens.weight", dim_V, dim_H, 0,
                         dim_V, 0, dim_H, q4_mode);
        ids_dev.alloc(static_cast<size_t>(rows_max) * sizeof(int64_t));
      }
      pos_dev.alloc(static_cast<size_t>(rows_max) * sizeof(int));
      seq_dev.alloc(static_cast<size_t>(rows_max) * sizeof(int));
      if (owns_head) {
        final_norm = load_mat(ws, "model.norm.weight", 1, dim_H, 0, 1, 0, dim_H,
                              false);
        lm_head = load_mat(ws,
                           cfg_local.tie_word_embeddings
                               ? "model.embed_tokens.weight"
                               : "lm_head.weight",
                           dim_V, dim_H, 0, dim_V, 0, dim_H, q4_mode);
        logits_buf.alloc(static_cast<size_t>(num_seq) * dim_V * 4);
      }

      (void)Iloc_full;
      lw.resize(layer_end - layer_begin);
      for (int l = layer_begin; l < layer_end; ++l) {
        Layer& L = lw[l - layer_begin];
        std::string p = "model.layers." + std::to_string(l) + ".";
        L.ln1 = load_mat(ws, p + "input_layernorm.weight", 1, dim_H, 0, 1, 0,
                         dim_H, false);
        L.ln2 = load_mat(ws, p + "post_attention_layernorm.weight", 1, dim_H, 0,
                         1, 0, dim_H, false);

        if (tp_mode) {
          int q_rows = dim_Hq * dim_D / tp_world;
          int kv_rows = dim_Hkv * dim_D / tp_world;
          L.wq = load_mat(ws, p + "self_attn.q_proj.weight", dim_Hq * dim_D,
                          dim_H, tp_rank * q_rows, q_rows, 0, dim_H, q4_mode);
          L.wk = load_mat(ws, p + "self_attn.k_proj.weight", dim_Hkv * dim_D,
                          dim_H, tp_rank * kv_rows, kv_rows, 0, dim_H, q4_mode);
          L.wv = load_mat(ws, p + "self_attn.v_proj.weight", dim_Hkv * dim_D,
                          dim_H, tp_rank * kv_rows, kv_rows, 0, dim_H, q4_mode);
          // o_proj column-split: [H, Hq*D] -> [H, Hq*D/tp]
          int attn_cols = dim_Hq * dim_D;
          L.wo = load_mat(ws, p + "self_attn.o_proj.weight", dim_H, attn_cols, 0,
                          dim_H, tp_rank * (attn_cols / tp_world),
                          attn_cols / tp_world, q4_mode);
        } else {
          // Fuse Q/K/V into one [wqkv, H] weight: 3 projections -> 1.
          DeviceBuffer tq = load_fp16(ws, p + "self_attn.q_proj.weight");
          DeviceBuffer tk = load_fp16(ws, p + "self_attn.k_proj.weight");
          DeviceBuffer tv = load_fp16(ws, p + "self_attn.v_proj.weight");
          size_t bq = static_cast<size_t>(dim_Hq) * dim_D * dim_H * 2;
          size_t bkv = static_cast<size_t>(dim_Hkv) * dim_D * dim_H * 2;
          DeviceBuffer fused;
          fused.alloc(bq + 2 * bkv);
          HIP_CHECK(hipMemcpyAsync(fused.data(), tq.data(), bq,
                                   hipMemcpyDeviceToDevice, 0));
          HIP_CHECK(hipMemcpyAsync(static_cast<char*>(fused.data()) + bq,
                                   tk.data(), bkv, hipMemcpyDeviceToDevice, 0));
          HIP_CHECK(hipMemcpyAsync(static_cast<char*>(fused.data()) + bq + bkv,
                                   tv.data(), bkv, hipMemcpyDeviceToDevice, 0));
          HIP_CHECK(hipStreamSynchronize(0));
          L.wqkv_fused.rows = dim_Hq * dim_D + 2 * dim_Hkv * dim_D;
          L.wqkv_fused.cols = dim_H;
          if (q4_mode) {
            L.wqkv_fused.qp.alloc(L.wqkv_fused.rows * dim_H / 2);
            L.wqkv_fused.qs.alloc(
                static_cast<size_t>(L.wqkv_fused.rows) * (dim_H / kQ4Group) * 2);
            quantize_q4_fp16(fused.data(), L.wqkv_fused.qp.data(),
                             L.wqkv_fused.qs.data(), L.wqkv_fused.rows, dim_H,
                             0);
            HIP_CHECK(hipStreamSynchronize(0));
            L.wqkv_fused.is_q4 = true;
          } else {
            L.wqkv_fused.fp = std::move(fused);
          }
          weight_bytes += L.wqkv_fused.bytes();
          L.wo = load_mat(ws, p + "self_attn.o_proj.weight", dim_H,
                          dim_Hq * dim_D, 0, dim_H, 0, dim_Hq * dim_D, q4_mode);
        }

        if (use_qknorm) {
          L.qn = load_mat(ws, p + "self_attn.q_norm.weight", 1, dim_D, 0, 1, 0,
                          dim_D, false);
          L.kn = load_mat(ws, p + "self_attn.k_norm.weight", 1, dim_D, 0, 1, 0,
                          dim_D, false);
        }

        if (is_moe) {
          if (tp_mode) {
            throw std::runtime_error(
                "tensor parallelism with MoE is not supported yet; use "
                "--parallel pp");
          }
          L.moe_gate = load_mat(ws, p + "mlp.gate.weight", num_experts, dim_H,
                                0, num_experts, 0, dim_H, q4_mode);
          L.moe_gu.rows = num_experts * 2 * dim_I;
          L.moe_gu.cols = dim_H;
          build_moe_gu(ws, p, q4_mode);
          L.moe_down = load_mat(ws, p + "mlp.experts.down_proj.weight",
                                num_experts * dim_H, dim_I, 0,
                                num_experts * dim_H, 0, dim_I, q4_mode);
        } else {
          // gate/up fused [2I, H], row-sliced under tp.
          DeviceBuffer tg = load_fp16(ws, p + "mlp.gate_proj.weight");
          DeviceBuffer tu = load_fp16(ws, p + "mlp.up_proj.weight");
          size_t seg = static_cast<size_t>(dim_I) * dim_H * 2;
          int row_off = tp_mode ? tp_rank * (dim_I / tp_world) : 0;
          int row_cnt = tp_mode ? dim_I / tp_world : dim_I;
          DeviceBuffer slice;
          slice.alloc(static_cast<size_t>(row_cnt) * dim_H * 2 * 2);
          HIP_CHECK(hipMemcpyAsync(
              slice.data(),
              static_cast<const char*>(tg.data()) + row_off * dim_H * 2,
              static_cast<size_t>(row_cnt) * dim_H * 2,
              hipMemcpyDeviceToDevice, 0));
          HIP_CHECK(hipMemcpyAsync(
              static_cast<char*>(slice.data()) + row_cnt * dim_H * 2,
              static_cast<const char*>(tu.data()) + row_off * dim_H * 2,
              static_cast<size_t>(row_cnt) * dim_H * 2,
              hipMemcpyDeviceToDevice, 0));
          HIP_CHECK(hipStreamSynchronize(0));
          L.gu.rows = 2 * row_cnt;
          L.gu.cols = dim_H;
          if (q4_mode) {
            L.gu.qp.alloc(static_cast<size_t>(L.gu.rows) * dim_H / 2);
            L.gu.qs.alloc(static_cast<size_t>(L.gu.rows) *
                          (dim_H / kQ4Group) * 2);
            quantize_q4_fp16(slice.data(), L.gu.qp.data(), L.gu.qs.data(),
                             L.gu.rows, dim_H, 0);
            HIP_CHECK(hipStreamSynchronize(0));
            L.gu.is_q4 = true;
          } else {
            L.gu.fp = std::move(slice);
          }
          weight_bytes += L.gu.bytes();
          // down column-split under tp: [H, I] -> [H, I/tp]
          L.down = load_mat(ws, p + "mlp.down_proj.weight", dim_H, dim_I, 0,
                            dim_H, tp_mode ? tp_rank * (dim_I / tp_world) : 0,
                            tp_mode ? dim_I / tp_world : dim_I, q4_mode);
        }
      }

      // Workspace.
      hidden.alloc(static_cast<size_t>(rows_max) * dim_H * 2);
      normed.alloc(static_cast<size_t>(rows_max) * dim_H * 2);
      qkv.alloc(static_cast<size_t>(rows_max) * (Hq_local + 2 * Hkv_local) *
                dim_D * 2);
      attn_out.alloc(static_cast<size_t>(rows_max) * Hq_local * dim_D * 2);
      proj.alloc(static_cast<size_t>(rows_max) * dim_H * 2);
      gate_up.alloc(static_cast<size_t>(rows_max) * 2 * Iloc * 2);
      act.alloc(static_cast<size_t>(rows_max) * Iloc * 2);
      peer_buf.alloc(static_cast<size_t>(rows_max) * dim_H * 2);
      {
        std::vector<float> freqs(dim_D / 2);
        for (int i = 0; i < dim_D / 2; ++i) {
          freqs[i] = std::pow(rope_theta, -2.0f * i / dim_D);
        }
        rope_freq.alloc(freqs.size() * sizeof(float));
        rope_freq.copy_from_host(freqs.data(), rope_freq.bytes());
      }
      if (is_moe) {
        size_t tsel = static_cast<size_t>(rows_max) * top_k_experts;
        moe_logits.alloc(static_cast<size_t>(rows_max) * num_experts * 4);
        moe_perm.alloc(tsel * dim_H * 2);
        moe_eout.alloc(tsel * dim_H * 2);
        moe_gate_up.alloc(tsel * 2 * Iloc * 2);
        moe_act.alloc(tsel * Iloc * 2);
        moe_order_dev.alloc(tsel * sizeof(int64_t));
        moe_w_dev.alloc(tsel * sizeof(float));
      }
      kv = std::make_unique<KVCache>(layer_end - layer_begin, max_ctx,
                                     Hkv_local, dim_D, num_seq);
      blas = std::make_unique<Blas>();
      HIP_CHECK(hipDeviceSynchronize());
    }

    // Builds the fused per-expert gate/up blob [E, 2I, H] from the two
    // [E, I, H] checkpoint tensors (fp16 or q4).
    void build_moe_gu(const WeightSource& ws, const std::string& p,
                      bool q4_mode) {
      DeviceBuffer tg = load_fp16(ws, p + "mlp.experts.gate_proj.weight");
      DeviceBuffer tu = load_fp16(ws, p + "mlp.experts.up_proj.weight");
      int E = num_experts;
      size_t seg = static_cast<size_t>(Iloc) * Hdim * 2;
      DeviceBuffer fused;
      fused.alloc(static_cast<size_t>(E) * 2 * seg);
      for (int e = 0; e < E; ++e) {
        HIP_CHECK(hipMemcpyAsync(static_cast<char*>(fused.data()) + e * 2 * seg,
                                 static_cast<const char*>(tg.data()) + e * seg,
                                 seg, hipMemcpyDeviceToDevice, 0));
        HIP_CHECK(hipMemcpyAsync(
            static_cast<char*>(fused.data()) + e * 2 * seg + seg,
            static_cast<const char*>(tu.data()) + e * seg, seg,
            hipMemcpyDeviceToDevice, 0));
      }
      HIP_CHECK(hipStreamSynchronize(0));
      // The fused blob keeps the caller's Layer reference; assign through the
      // last loaded layer.
      Layer& L = lw.back();
      if (q4_mode) {
        L.moe_gu.qp.alloc(fused.bytes() / 2);
        L.moe_gu.qs.alloc(static_cast<size_t>(E) * 2 * Iloc *
                          (Hdim / kQ4Group) * 2);
        quantize_q4_fp16(fused.data(), L.moe_gu.qp.data(), L.moe_gu.qs.data(),
                         E * 2 * Iloc, Hdim, 0);
        HIP_CHECK(hipStreamSynchronize(0));
        L.moe_gu.is_q4 = true;
      } else {
        L.moe_gu.fp = std::move(fused);
      }
      weight_bytes += L.moe_gu.bytes();
    }

    // ---- embeddings / rows upload ----
    void begin_rows(const std::vector<int64_t>& tokens,
                    const std::vector<int>& positions,
                    const std::vector<int>& seq_ids) {
      hipStream_t stream = 0;
      int T = static_cast<int>(tokens.size());
      if (owns_embed) {
        HIP_CHECK(hipMemcpyAsync(ids_dev.data(), tokens.data(),
                                 T * sizeof(int64_t), hipMemcpyHostToDevice,
                                 stream));
        embed_gather_fp16(hidden.data(), embed.fp.data(),
                          static_cast<const int64_t*>(ids_dev.data()), T, Hdim,
                          stream);
      }
      HIP_CHECK(hipMemcpyAsync(pos_dev.data(), positions.data(),
                               T * sizeof(int), hipMemcpyHostToDevice, stream));
      HIP_CHECK(hipMemcpyAsync(seq_dev.data(), seq_ids.data(),
                               T * sizeof(int), hipMemcpyHostToDevice, stream));
    }

    // ---- logits for the last row of every group (head shard only) ----
    ForwardOutput finalize(const std::vector<Group>& groups) {
      ForwardOutput out;
      out.rows = static_cast<int>(groups.size());
      for (const Group& g : groups) out.group_seq.push_back(g.seq);
      if (!owns_head) return out;
      out.logits = static_cast<const float*>(logits_buf.data());
      hipStream_t stream = 0;
      int T = static_cast<int>(groups.size());
      for (int gi = 0; gi < out.rows; ++gi) {
        const Group& g = groups[gi];
        int r = g.start + g.count - 1;
        __half* h_last = static_cast<__half*>(hidden.data()) +
                         static_cast<int64_t>(r) * Hdim;
        __half* n_row =
            static_cast<__half*>(normed.data()) + static_cast<int64_t>(gi) * Hdim;
        rmsnorm_fp16(n_row, h_last, final_norm.fp.data(), 1, Hdim, eps, stream);
        float* row = static_cast<float*>(logits_buf.data()) +
                     static_cast<int64_t>(gi) * V_dim;
        if (lm_head.is_q4) {
          gemv_q4_fp32out(row, lm_head.qp.data(), lm_head.qs.data(), n_row,
                          lm_head.rows, lm_head.cols, stream);
        } else {
          gemv_fp32out(row, lm_head.fp.data(), n_row, lm_head.rows,
                       lm_head.cols, stream);
        }
      }
      (void)T;
      return out;
    }

    int V_dim = 0;
  };

  std::vector<std::unique_ptr<Shard>> shards;
  std::vector<hipEvent_t> boundary_events;  // pp: shard boundary
  std::vector<hipEvent_t> tp_events;        // tp: compute done, one per rank
  std::vector<hipEvent_t> tp_done_events;   // tp: all-reduce copy done

  ~Impl() {
    for (hipEvent_t e : boundary_events) (void)hipEventDestroy(e);
    for (hipEvent_t e : tp_events) (void)hipEventDestroy(e);
    for (hipEvent_t e : tp_done_events) (void)hipEventDestroy(e);
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
    use_q4 = (opt.quant == "q4");
    use_tp = (opt.parallel == "tp");
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

    if (use_tp) {
      if (H % n != 0 || Hkv % n != 0 || I % n != 0) {
        throw std::runtime_error("tensor parallelism requires head counts and "
                                 "intermediate_size divisible by GPU count");
      }
      if (cfg.is_moe) {
        throw std::runtime_error("tensor parallelism with MoE is not supported "
                                 "yet; use --parallel pp");
      }
    } else if (n > 1 && layers < n) {
      throw std::runtime_error("cannot split " + std::to_string(layers) +
                               " layers across " + std::to_string(n) + " GPUs");
    }

    LOG_INFO("loading weights from %s (%d GPU%s, %s, %s, %d sequence slots)...",
             dir.c_str(), n, n > 1 ? "s" : "", use_tp ? "tp" : "pp",
             use_q4 ? "q4" : "fp16", opt.num_sequences);

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
      if (use_tp) {
        sh->layer_begin = 0;
        sh->layer_end = layers;
        sh->tp_rank = s;
        sh->tp_world = n;
        sh->owns_embed = (s == 0);
        sh->owns_head = (s == 0);
      } else {
        sh->layer_begin = layers * s / n;
        sh->layer_end = layers * (s + 1) / n;
        sh->owns_embed = (s == 0);
        sh->owns_head = (s == n - 1);
      }
      sh->V_dim = V;
      sh->load(dir, cfg, H, D, Hq, Hkv, V, I, eps, theta, opt.prefill_chunk,
               opt.max_context, opt.num_sequences, use_q4, use_tp, sh->tp_rank,
               sh->tp_world, sh->layer_begin, sh->layer_end, sh->owns_embed,
               sh->owns_head);
      shards.push_back(std::move(sh));
    }

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
    LOG_INFO("weights resident: %.2f GiB across %d device(s) (%s %s)",
             static_cast<double>(total) / (1024.0 * 1024.0 * 1024.0), n,
             use_tp ? "tp" : "pp", use_q4 ? "q4" : "fp16");

    for (int i = 0; i + 1 < n; ++i) {
      hipEvent_t ev;
      HIP_CHECK(hipEventCreate(&ev));
      boundary_events.push_back(ev);
    }
    for (int i = 0; i < n; ++i) {
      hipEvent_t ev;
      HIP_CHECK(hipEventCreate(&ev));
      tp_events.push_back(ev);
      HIP_CHECK(hipEventCreate(&ev));
      tp_done_events.push_back(ev);
    }
  }

  static std::vector<Group> build_groups(const std::vector<int>& seq_ids,
                                         const std::vector<int>& positions,
                                         int T) {
    std::vector<Group> groups;
    for (int i = 0; i < T;) {
      int j = i;
      while (j < T && seq_ids[j] == seq_ids[i]) ++j;
      groups.push_back({seq_ids[i], i, j - i, positions[i]});
      i = j;
    }
    return groups;
  }

  // Tensor-parallel all-reduce of the partial `proj` buffer across ranks.
  // Fully event-ordered (no host device drains, no per-call allocation):
  //   1. wait for the peer's compute event, pull the peer partial over PCIe
  //   2. add it locally, record a completion event
  //   3. every rank waits the others' completion events, so no rank can
  //      overwrite `proj` before all peers have copied it
  void tp_allreduce(int rows) {
    size_t bytes = static_cast<size_t>(rows) * H * 2;
    int n = static_cast<int>(shards.size());
    for (int r = 0; r < n; ++r) {
      Shard& cur = *shards[r];
      Shard& peer = *shards[(r + 1) % n];
      DeviceGuard g(cur.device);
      HIP_CHECK(hipStreamWaitEvent(0, tp_events[(r + 1) % n], 0));
      HIP_CHECK(hipMemcpyPeerAsync(cur.peer_buf.data(), cur.device,
                                   peer.proj.data(), peer.device, bytes, 0));
    }
    for (int r = 0; r < n; ++r) {
      DeviceGuard g(shards[r]->device);
      add_fp16(shards[r]->proj.data(), shards[r]->peer_buf.data(),
               static_cast<int64_t>(rows) * H, 0);
      HIP_CHECK(hipEventRecord(tp_done_events[r], 0));
    }
    for (int r = 0; r < n; ++r) {
      DeviceGuard g(shards[r]->device);
      for (int q = 0; q < n; ++q) {
        if (q != r) HIP_CHECK(hipStreamWaitEvent(0, tp_done_events[q], 0));
      }
    }
  }

  ForwardOutput forward_batch(const std::vector<int64_t>& tokens,
                              const std::vector<int>& positions,
                              const std::vector<int>& seq_ids,
                              hipStream_t /*stream*/) {
    int T = static_cast<int>(tokens.size());
    std::vector<Group> groups = build_groups(seq_ids, positions, T);
    bool all_single = true;
    for (const Group& g : groups) {
      if (g.count != 1) all_single = false;
    }
    ForwardOutput out;

    if (!use_tp) {
      // ---- Pipeline parallelism: shards run in order, activations cross the
      // boundary once per forward.
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
        cur.begin_rows(tokens, positions, seq_ids);
        for (int li = 0; li < static_cast<int>(cur.lw.size()); ++li) {
          cur.layer_start(li, groups, T, all_single, false);
          cur.layer_mid(li, T);
          cur.layer_end(T);
        }
        out = cur.finalize(groups);
        if (s + 1 < shards.size()) {
          HIP_CHECK(hipEventRecord(boundary_events[s], 0));
        }
      }
      return out;
    }

    // ---- Tensor parallelism: all ranks execute the same layers concurrently,
    // with an all-reduce after o_proj and after down_proj.
    for (auto& s : shards) {
      DeviceGuard g(s->device);
      s->begin_rows(tokens, positions, seq_ids);
    }
    int nlayers = static_cast<int>(shards[0]->lw.size());
    for (int li = 0; li < nlayers; ++li) {
      for (auto& s : shards) {
        DeviceGuard g(s->device);
        s->layer_start(li, groups, T, all_single, true);
        HIP_CHECK(hipEventRecord(tp_events[s->tp_rank], 0));
      }
      tp_allreduce(T);  // o_proj partials
      for (auto& s : shards) {
        DeviceGuard g(s->device);
        s->layer_mid(li, T);
        HIP_CHECK(hipEventRecord(tp_events[s->tp_rank], 0));
      }
      tp_allreduce(T);  // down_proj partials
      for (auto& s : shards) {
        DeviceGuard g(s->device);
        s->layer_end(T);
      }
    }
    for (auto& s : shards) {
      DeviceGuard g(s->device);
      ForwardOutput o = s->finalize(groups);
      if (s->owns_head) out = o;
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
    s->kv->set_length(seq, 0);
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

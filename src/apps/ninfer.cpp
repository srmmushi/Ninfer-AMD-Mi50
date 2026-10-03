// NInfer-HIP CLI: one-shot or interactive chat on an AMD MI50 (gfx906).
// Mirrors ninfer's apps/cli: answer content on stdout, diagnostics on stderr.

#include <cstdio>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/log.h"
#include "runtime/engine.h"

using namespace ninfer;

namespace {

struct Options {
  std::string model_dir;
  std::string prompt;
  std::string system;
  bool raw = false;
  bool no_thinking = false;
  int max_context = 32768;
  int prefill_chunk = 512;
  int max_new = 512;
  float temperature = 0.7f;
  int top_k = 50;
  float top_p = 0.95f;
  float repeat_penalty = 1.05f;
  int64_t seed = 0;
  std::vector<int> gpu_ids = {0};
  bool verify = false;  // dual-GPU vs single-GPU golden comparison
  std::string quant = "fp16";    // fp16 | q4 (groupwise INT4)
  std::string parallel = "pp";   // pp (layer split) | tp (tensor parallel)
  bool graph = false;   // opt-in HIP graph decode (single-GPU dense only)
};

std::vector<int> parse_gpu_list(const std::string& csv) {
  std::vector<int> out;
  std::string item;
  for (char c : csv) {
    if (c == ',' || c == ' ') {
      if (!item.empty()) out.push_back(std::stoi(item));
      item.clear();
    } else {
      item.push_back(c);
    }
  }
  if (!item.empty()) out.push_back(std::stoi(item));
  if (out.empty()) throw std::runtime_error("empty --gpus list");
  return out;
}

void usage() {
  std::fprintf(stderr,
               "usage: ninfer <model_dir> [options]\n"
               "  --prompt TEXT      one-shot generation (omit for chat REPL)\n"
               "  --system TEXT      system message\n"
               "  --raw              treat the prompt as raw text (no template)\n"
               "  --no-thinking      disable the Qwen3 thinking marker\n"
               "  --max-context N    KV capacity in tokens (default 32768)\n"
               "  --prefill-chunk N  prefill chunk size (default 512)\n"
               "  --max-new N        new tokens per generation (default 512)\n"
               "  --temperature F    sampling temperature (0 = greedy)\n"
               "  --top-k N          top-k (default 50)\n"
               "  --top-p F          nucleus probability (default 0.95)\n"
               "  --repeat-penalty F repetition penalty (default 1.05)\n"
               "  --seed N           sampling seed (0 = random)\n"
               "  --gpus 0,1         layer-split pipeline across GPUs (default 0)\n"
               "  --quant fp16|q4    weight format (q4 = groupwise INT4, ~4x less traffic)\n"
               "  --parallel pp|tp   pp = layer split (default), tp = tensor parallel\n"
               "  --graph            opt-in HIP graph decode (single-GPU dense only)\n"
               "  --verify           dual-GPU run compared against single-GPU golden\n"
               "  --log-level debug|info|warn|error\n");
}

Options parse(int argc, char** argv) {
  Options o;
  if (argc < 2) {
    usage();
    std::exit(2);
  }
  o.model_dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--prompt") o.prompt = next();
    else if (a == "--system") o.system = next();
    else if (a == "--raw") o.raw = true;
    else if (a == "--no-thinking") o.no_thinking = true;
    else if (a == "--max-context") o.max_context = std::stoi(next());
    else if (a == "--prefill-chunk") o.prefill_chunk = std::stoi(next());
    else if (a == "--max-new") o.max_new = std::stoi(next());
    else if (a == "--temperature") o.temperature = std::stof(next());
    else if (a == "--top-k") o.top_k = std::stoi(next());
    else if (a == "--top-p") o.top_p = std::stof(next());
    else if (a == "--repeat-penalty") o.repeat_penalty = std::stof(next());
    else if (a == "--seed") o.seed = std::stoll(next());
    else if (a == "--gpus") o.gpu_ids = parse_gpu_list(next());
    else if (a == "--verify") o.verify = true;
    else if (a == "--graph") o.graph = true;
    else if (a == "--quant") o.quant = next();
    else if (a == "--parallel") o.parallel = next();
    else if (a == "--log-level") {
      std::string lv = next();
      if (lv == "debug") global_log_level() = LogLevel::Debug;
      else if (lv == "info") global_log_level() = LogLevel::Info;
      else if (lv == "warn") global_log_level() = LogLevel::Warn;
      else if (lv == "error") global_log_level() = LogLevel::Error;
    } else {
      throw std::runtime_error("unknown option: " + a);
    }
  }
  return o;
}

std::string generate_once(Engine& engine,
                          const std::vector<ChatMessage>& messages,
                          const Options& o) {
  SamplingParams sp;
  sp.temperature = o.temperature;
  sp.top_k = o.top_k;
  sp.top_p = o.top_p;
  sp.repetition_penalty = o.repeat_penalty;
  sp.seed = o.seed;
  sp.max_new_tokens = o.max_new;

  std::string out = engine.generate_chat(messages, sp, [](const std::string& d) {
    std::fputs(d.c_str(), stdout);
    std::fflush(stdout);
  });
  std::fputs("\n", stdout);
  return out;
}

// Dual-GPU correctness check: regenerate silently on the dual-GPU engine,
// then rerun the same prompt on a single-GPU golden engine and compare.
// FP16 reduction order differs at the shard boundary, so bit-exact equality
// is not guaranteed — we report exact match or the first divergence point.
void verify_dual(const Options& o, Engine& engine, bool raw,
                 const std::vector<ChatMessage>& messages,
                 const SamplingParams& sp) {
  if (o.gpu_ids.size() < 2) {
    LOG_WARN("--verify requires --gpus listing at least two GPUs");
    return;
  }
  std::string dual_text =
      raw ? engine.generate(o.prompt, sp) : engine.generate_chat(messages, sp);

  ModelOptions mo_single;
  mo_single.max_context = o.max_context;
  mo_single.prefill_chunk = o.prefill_chunk;
  mo_single.gpu_ids = {o.gpu_ids[0]};
  try {
    Engine golden(o.model_dir, mo_single, !o.no_thinking);
    std::string single_text =
        raw ? golden.generate(o.prompt, sp) : golden.generate_chat(messages, sp);
    if (dual_text == single_text) {
      LOG_INFO("verify: PASS — dual-GPU output identical to single-GPU golden "
               "(%zu chars)", dual_text.size());
    } else {
      size_t i = 0;
      while (i < dual_text.size() && i < single_text.size() &&
             dual_text[i] == single_text[i]) {
        ++i;
      }
      LOG_WARN("verify: outputs diverge at char %zu "
               "(dual %zu chars / single %zu chars); fp16 reduction-order "
               "noise is expected at shard boundaries",
               i, dual_text.size(), single_text.size());
    }
  } catch (const std::exception& e) {
    LOG_WARN("verify: single-GPU golden unavailable: %s", e.what());
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    Options o = parse(argc, argv);
    ModelOptions mo;
    mo.max_context = o.max_context;
    mo.prefill_chunk = o.prefill_chunk;
    mo.gpu_ids = o.gpu_ids;
    mo.quant = o.quant;
    mo.parallel = o.parallel;
    mo.use_graph = o.graph;
    Engine engine(o.model_dir, mo, !o.no_thinking);

    std::vector<ChatMessage> messages;
    if (!o.system.empty()) messages.push_back({"system", o.system, false, false});

    if (!o.prompt.empty()) {
      SamplingParams sp;
      sp.temperature = o.temperature;
      sp.top_k = o.top_k;
      sp.top_p = o.top_p;
      sp.repetition_penalty = o.repeat_penalty;
      // Deterministic seed so --verify compares identical sampling paths.
      sp.seed = o.seed != 0 ? o.seed : 1234;
      sp.max_new_tokens = o.max_new;
      if (o.raw) {
        engine.generate(o.prompt, sp, [](const std::string& d) {
          std::fputs(d.c_str(), stdout);
          std::fflush(stdout);
        });
        std::fputs("\n", stdout);
        if (o.verify) verify_dual(o, engine, /*raw=*/true, messages, sp);
      } else {
        messages.push_back({"user", o.prompt, false, false});
        generate_once(engine, messages, o);
        if (o.verify) verify_dual(o, engine, /*raw=*/false, messages, sp);
      }
      return 0;
    }

    // Interactive REPL with multi-turn history.
    std::fprintf(stderr, "NInfer-HIP chat ready. Ctrl-D to exit.\n");
    std::string line;
    while (true) {
      std::fprintf(stderr, "\n> ");
      if (!std::getline(std::cin, line)) break;
      if (line.empty()) continue;
      messages.push_back({"user", line, false, false});
      std::string answer = generate_once(engine, messages, o);
      messages.push_back({"assistant", answer, false, false});
    }
    return 0;
  } catch (const std::exception& e) {
    LOG_ERROR("%s", e.what());
    return 1;
  }
}

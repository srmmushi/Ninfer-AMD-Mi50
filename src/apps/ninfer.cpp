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
};

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

}  // namespace

int main(int argc, char** argv) {
  try {
    Options o = parse(argc, argv);
    ModelOptions mo;
    mo.max_context = o.max_context;
    mo.prefill_chunk = o.prefill_chunk;
    Engine engine(o.model_dir, mo, !o.no_thinking);

    std::vector<ChatMessage> messages;
    if (!o.system.empty()) messages.push_back({"system", o.system, false, false});

    if (!o.prompt.empty()) {
      if (o.raw) {
        SamplingParams sp;
        sp.temperature = o.temperature;
        sp.top_k = o.top_k;
        sp.top_p = o.top_p;
        sp.repetition_penalty = o.repeat_penalty;
        sp.seed = o.seed;
        sp.max_new_tokens = o.max_new;
        engine.generate(o.prompt, sp, [](const std::string& d) {
          std::fputs(d.c_str(), stdout);
          std::fflush(stdout);
        });
        std::fputs("\n", stdout);
      } else {
        messages.push_back({"user", o.prompt, false, false});
        generate_once(engine, messages, o);
        return 0;
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

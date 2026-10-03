// NInfer-HIP server: OpenAI-compatible /v1/chat/completions (streaming SSE
// and JSON), mirroring ninfer's apps/serve surface for the supported subset.

#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "apps/http_server.h"
#include "common/json.h"
#include "common/log.h"
#include "common/util.h"
#include "runtime/engine.h"

using namespace ninfer;

namespace {

struct ServeOptions {
  std::string model_dir;
  uint16_t port = 8080;
  int max_context = 32768;
  int prefill_chunk = 512;
  bool no_thinking = false;
  float temperature = 0.7f;
  int top_k = 50;
  float top_p = 0.95f;
  float repeat_penalty = 1.05f;
  std::vector<int> gpu_ids = {0};
  std::string quant = "fp16";
  std::string parallel = "pp";
  bool graph = false;   // opt-in HIP graph decode (single-GPU dense only)
};

ServeOptions parse(int argc, char** argv) {
  ServeOptions o;
  if (argc < 2) {
    throw std::runtime_error(
        "usage: ninfer-serve <model_dir> [--port N] [--max-context N] "
        "[--prefill-chunk N] [--gpus 0,1] [--no-thinking] [--temperature F] "
        "[--top-k N] [--top-p F] [--repeat-penalty F] [--graph] "
        "[--log-level LEVEL]");
  }
  o.model_dir = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--port") o.port = static_cast<uint16_t>(std::stoi(next()));
    else if (a == "--max-context") o.max_context = std::stoi(next());
    else if (a == "--prefill-chunk") o.prefill_chunk = std::stoi(next());
    else if (a == "--no-thinking") o.no_thinking = true;
    else if (a == "--temperature") o.temperature = std::stof(next());
    else if (a == "--top-k") o.top_k = std::stoi(next());
    else if (a == "--top-p") o.top_p = std::stof(next());
    else if (a == "--repeat-penalty") o.repeat_penalty = std::stof(next());
    else if (a == "--gpus") {
      std::string csv = next();
      std::string item;
      for (char c : csv) {
        if (c == ',' || c == ' ') {
          if (!item.empty()) o.gpu_ids.push_back(std::stoi(item));
          item.clear();
        } else {
          item.push_back(c);
        }
      }
      if (!item.empty()) o.gpu_ids.push_back(std::stoi(item));
      if (o.gpu_ids.empty()) throw std::runtime_error("empty --gpus list");
    }     else if (a == "--quant") o.quant = next();
    else if (a == "--parallel") o.parallel = next();
    else if (a == "--graph") o.graph = true;
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

std::vector<ChatMessage> parse_messages(const Json& body) {
  std::vector<ChatMessage> out;
  const Json* msgs = body.find("messages");
  if (!msgs || !msgs->is_array() || msgs->items().empty()) {
    throw std::runtime_error("'messages' must be a non-empty array");
  }
  for (const auto& m : msgs->items()) {
    ChatMessage cm;
    cm.role = m.at("role").as_string();
    const Json* content = m.find("content");
    if (content && content->is_string()) {
      cm.content = content->as_string();
    } else if (content && content->is_array()) {
      // OpenAI multimodal content parts: keep text parts (vision is out of
      // scope for the gfx906 subset).
      for (const auto& part : content->items()) {
        if (part.as_string().empty() && part.find("text")) {
          cm.content += part.at("text").as_string();
        } else if (part.is_string()) {
          cm.content += part.as_string();
        }
      }
    }
    out.push_back(std::move(cm));
  }
  return out;
}

SamplingParams sampling_from(const Json& body, const ServeOptions& o) {
  SamplingParams sp;
  sp.temperature = body.find("temperature")
                       ? static_cast<float>(body.at("temperature").as_number())
                       : o.temperature;
  sp.top_p = body.find("top_p")
                 ? static_cast<float>(body.at("top_p").as_number())
                 : o.top_p;
  if (body.find("top_k")) sp.top_k = static_cast<int>(body.at("top_k").as_int());
  sp.max_new_tokens = body.find("max_tokens")
                          ? static_cast<int>(body.at("max_tokens").as_int())
                          : 512;
  if (body.find("presence_penalty")) {
    float pp = static_cast<float>(body.at("presence_penalty").as_number());
    sp.repetition_penalty = pp != 0.0f ? 1.0f + pp : o.repeat_penalty;
  }
  if (body.find("seed")) sp.seed = body.at("seed").as_int();
  return sp;
}

std::string sse_data(const std::string& payload) {
  return "data: " + payload + "\n\n";
}

Json make_chunk(const std::string& model, const std::string& delta_content,
                bool finish) {
  Json chunk = Json::object();
  chunk["id"] = Json(std::string("chatcmpl-ninferhip"));
  chunk["object"] = Json(std::string("chat.completion.chunk"));
  chunk["model"] = Json(model);
  Json choices = Json::array();
  Json choice = Json::object();
  choice["index"] = Json(0);
  Json delta = Json::object();
  if (!delta_content.empty()) delta["content"] = Json(delta_content);
  choice["delta"] = delta;
  choice["finish_reason"] = Json(finish ? std::string("stop") : std::string());
  choices.push_back(choice);
  chunk["choices"] = choices;
  return chunk;
}

void handle_chat(const HttpRequest& req, HttpConnection& conn, Engine& engine,
                 std::mutex& gen_mutex, const ServeOptions& opt) {
  Json body = Json::parse(req.body);
  std::vector<ChatMessage> messages = parse_messages(body);
  SamplingParams sp = sampling_from(body, opt);
  bool stream = body.find("stream") ? body.at("stream").as_bool() : false;
  const std::string model = body.find("model") && body.at("model").is_string()
                                ? body.at("model").as_string()
                                : std::string("ninfer-hip");

  std::lock_guard<std::mutex> lock(gen_mutex);  // engine is single-sequence

  if (!stream) {
    std::string text = engine.generate_chat(messages, sp);
    Json resp = Json::object();
    resp["id"] = Json(std::string("chatcmpl-ninferhip"));
    resp["object"] = Json(std::string("chat.completion"));
    resp["model"] = Json(model);
    Json choices = Json::array();
    Json choice = Json::object();
    choice["index"] = Json(0);
    Json message = Json::object();
    message["role"] = Json(std::string("assistant"));
    message["content"] = Json(text);
    choice["message"] = message;
    choice["finish_reason"] = Json(std::string("stop"));
    choices.push_back(choice);
    resp["choices"] = choices;
    Json usage = Json::object();
    usage["prompt_tokens"] = Json(0);
    usage["completion_tokens"] = Json(0);
    resp["usage"] = usage;
    std::string payload = resp.dump();
    conn.write("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
               "Content-Length: " + std::to_string(payload.size()) +
               "\r\nConnection: close\r\n\r\n" + payload);
    return;
  }

  // SSE streaming.
  conn.write("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
             "Cache-Control: no-cache\r\nConnection: close\r\n\r\n");
  engine.generate_chat(messages, sp, [&](const std::string& delta) {
    if (!conn.is_open()) return;
    conn.write(sse_data(make_chunk(model, delta, false).dump()));
  });
  if (conn.is_open()) {
    conn.write(sse_data(make_chunk(model, "", true).dump()));
    conn.write("data: [DONE]\n\n");
  }
}

}  // namespace

int main(int argc, char** argv) {
  try {
    ServeOptions o = parse(argc, argv);
    ModelOptions mo;
    mo.max_context = o.max_context;
    mo.prefill_chunk = o.prefill_chunk;
    mo.gpu_ids = o.gpu_ids;
    mo.quant = o.quant;
    mo.parallel = o.parallel;
    mo.use_graph = o.graph;
    Engine engine(o.model_dir, mo, !o.no_thinking);
    std::mutex gen_mutex;

    serve_http(o.port, [&](const HttpRequest& req, HttpConnection& conn) {
      if (req.method == "GET" && req.path == "/health") {
        std::string body = "{\"ok\":true,\"gpu\":\"gfx906\"}";
        conn.write("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                   "Content-Length: " + std::to_string(body.size()) +
                   "\r\nConnection: close\r\n\r\n" + body);
        return;
      }
      if (req.method == "POST" && req.path == "/v1/chat/completions") {
        handle_chat(req, conn, engine, gen_mutex, o);
        return;
      }
      std::string body = "{\"error\":\"not found\"}";
      conn.write("HTTP/1.1 404 Not Found\r\nContent-Type: application/json\r\n"
                 "Content-Length: " + std::to_string(body.size()) +
                 "\r\nConnection: close\r\n\r\n" + body);
    });
    return 0;
  } catch (const std::exception& e) {
    LOG_ERROR("%s", e.what());
    return 1;
  }
}

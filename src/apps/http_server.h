#pragma once
// Minimal threaded HTTP/1.1 server (POSIX sockets) — replaces the vendored
// cpp-httplib of the original ninfer for the serving endpoint surface.

#include <functional>
#include <map>
#include <string>

namespace ninfer {

struct HttpRequest {
  std::string method;
  std::string path;
  std::map<std::string, std::string> headers;
  std::string body;
};

// The handler writes the full response (status line + headers + body, or SSE
// frames) through the connection writer.
class HttpConnection {
 public:
  virtual ~HttpConnection() = default;
  virtual void write(const std::string& data) = 0;
  virtual bool is_open() const = 0;
};

using HttpHandler =
    std::function<void(const HttpRequest&, HttpConnection&)>;

// Blocking accept loop; one detached thread per connection. Returns only on
// fatal socket errors.
void serve_http(uint16_t port, HttpHandler handler);

}  // namespace ninfer

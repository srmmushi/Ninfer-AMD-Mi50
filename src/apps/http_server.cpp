#include "apps/http_server.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cctype>
#include <cstring>
#include <stdexcept>
#include <thread>

#include "common/log.h"

namespace ninfer {
namespace {

class SocketConnection : public HttpConnection {
 public:
  explicit SocketConnection(int fd) : fd_(fd) {}
  ~SocketConnection() override { close(fd_); }

  void write(const std::string& data) override {
    size_t off = 0;
    while (off < data.size()) {
      ssize_t n = ::send(fd_, data.data() + off, data.size() - off, MSG_NOSIGNAL);
      if (n <= 0) {
        open_ = false;
        return;
      }
      off += static_cast<size_t>(n);
    }
  }

  bool is_open() const override { return open_; }

 private:
  int fd_;
  bool open_ = true;
};

std::string tolower_copy(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

bool read_request(int fd, HttpRequest& req) {
  std::string buf;
  char tmp[4096];
  // Read until end of headers.
  size_t header_end = std::string::npos;
  while (header_end == std::string::npos) {
    ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) return false;
    buf.append(tmp, static_cast<size_t>(n));
    header_end = buf.find("\r\n\r\n");
    if (buf.size() > (1u << 20)) return false;
  }
  std::string head = buf.substr(0, header_end);
  std::string rest = buf.substr(header_end + 4);

  // Request line.
  size_t sp1 = head.find(' ');
  size_t sp2 = head.find(' ', sp1 + 1);
  if (sp1 == std::string::npos || sp2 == std::string::npos) return false;
  req.method = head.substr(0, sp1);
  req.path = head.substr(sp1 + 1, sp2 - sp1 - 1);
  size_t qpos = req.path.find('?');
  if (qpos != std::string::npos) req.path.resize(qpos);

  // Headers.
  size_t line_start = head.find("\r\n");
  while (line_start != std::string::npos && line_start + 2 < head.size()) {
    size_t line_end = head.find("\r\n", line_start + 2);
    if (line_end == std::string::npos) line_end = head.size();
    std::string line = head.substr(line_start + 2, line_end - line_start - 2);
    size_t colon = line.find(':');
    if (colon != std::string::npos) {
      std::string key = tolower_copy(line.substr(0, colon));
      size_t vstart = colon + 1;
      while (vstart < line.size() && line[vstart] == ' ') ++vstart;
      req.headers[key] = line.substr(vstart);
    }
    line_start = line_end;
  }

  // Body.
  size_t content_length = 0;
  auto it = req.headers.find("content-length");
  if (it != req.headers.end()) content_length = std::stoull(it->second);
  if (content_length > (1ull << 30)) return false;
  while (rest.size() < content_length) {
    ssize_t n = ::recv(fd, tmp, sizeof(tmp), 0);
    if (n <= 0) return false;
    rest.append(tmp, static_cast<size_t>(n));
  }
  req.body = rest.substr(0, content_length);
  return true;
}

void handle_client(int fd, const HttpHandler& handler) {
  SocketConnection conn(fd);
  HttpRequest req;
  if (!read_request(fd, req)) return;
  try {
    handler(req, conn);
  } catch (const std::exception& e) {
    LOG_ERROR("http handler: %s", e.what());
    if (conn.is_open()) {
      std::string body = std::string("{\"error\":\"") + e.what() + "\"}";
      conn.write("HTTP/1.1 500 Internal Server Error\r\n"
                 "Content-Type: application/json\r\n"
                 "Connection: close\r\nContent-Length: " +
                 std::to_string(body.size()) + "\r\n\r\n" + body);
    }
  }
}

}  // namespace

void serve_http(uint16_t port, HttpHandler handler) {
  int server = ::socket(AF_INET, SOCK_STREAM, 0);
  if (server < 0) throw std::runtime_error("socket() failed");
  int one = 1;
  setsockopt(server, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  addr.sin_port = htons(port);
  if (::bind(server, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
    throw std::runtime_error("bind() failed on port " + std::to_string(port));
  }
  if (::listen(server, 16) < 0) {
    throw std::runtime_error("listen() failed");
  }
  LOG_INFO("http server listening on 0.0.0.0:%u", port);

  while (true) {
    sockaddr_in peer{};
    socklen_t len = sizeof(peer);
    int fd = ::accept(server, reinterpret_cast<sockaddr*>(&peer), &len);
    if (fd < 0) continue;
    int nd = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &nd, sizeof(nd));
    std::thread(handle_client, fd, handler).detach();
  }
}

}  // namespace ninfer

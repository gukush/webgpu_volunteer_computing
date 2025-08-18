#pragma once
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/ssl.hpp>
#include <functional>
#include <string>
#include <thread>
#include <atomic>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

class WebSocketClient {
private:
    net::io_context ioc;
    ssl::context ctx{ssl::context::tlsv12_client};
    tcp::resolver resolver{ioc};
    websocket::stream<beast::ssl_stream<tcp::socket>> ws{ioc, ctx};

    std::thread ioThread;
    std::atomic<bool> shouldStop{false};

    std::function<void()> onConnected;
    std::function<void()> onDisconnected;
    std::function<void(const std::string&)> onMessage;

public:
    WebSocketClient();
    ~WebSocketClient();

    bool connect(const std::string& host, const std::string& port, const std::string& target);
    void disconnect();
    void send(const std::string& message);

    void setOnConnected(std::function<void()> handler) { onConnected = handler; }
    void setOnDisconnected(std::function<void()> handler) { onDisconnected = handler; }
    void setOnMessage(std::function<void(const std::string&)> handler) { onMessage = handler; }

private:
    void runEventLoop();
};

// websocket_client.hpp - Updated for native WebSocket support
#pragma once
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/beast/ssl.hpp>
#include <boost/asio/ssl.hpp>
#include <nlohmann/json.hpp>
#include <functional>
#include <string>
#include <thread>
#include <atomic>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;
using json = nlohmann::json;

class WebSocketClient {
private:
    net::io_context ioc;
    ssl::context ctx{ssl::context::tlsv12_client};
    tcp::resolver resolver{ioc};
    websocket::stream<beast::ssl_stream<tcp::socket>> ws{ioc, ctx};

    std::thread ioThread;
    std::atomic<bool> shouldStop{false};

    // Event handlers
    std::function<void()> onConnected;
    std::function<void()> onDisconnected;
    std::function<void(const std::string&)> onMessage;
    std::function<void(const json&)> onRegister;
    std::function<void(const json&)> onTaskAssigned;
    std::function<void(const json&)> onWorkloadAssigned;
    std::function<void(const json&)> onChunkAssigned;
    std::function<void(const json&)> onTaskVerified;
    std::function<void(const json&)> onTaskSubmitted;
    std::function<void(const json&)> onWorkloadComplete;

public:
    WebSocketClient();
    ~WebSocketClient();

    // Connection management
    bool connect(const std::string& host, const std::string& port, const std::string& target = "/");
    void disconnect();

    // Message sending
    void send(const std::string& message);
    void sendEvent(const std::string& eventType, const json& data = json::object());

    // High-level API methods
    void joinComputation(const json& capabilities);
    void requestTask();
    void submitTaskResult(const std::string& assignmentId, const std::string& taskId,
                         const json& result, double processingTime, const std::string& checksum = "");
    void submitWorkloadResult(const std::string& workloadId, const std::string& result,
                             double processingTime, const std::string& checksum = "");
    void submitChunkResult(const std::string& parentId, const std::string& chunkId,
                          const json& results, double processingTime,
                          const std::string& strategy = "", const json& metadata = json::object(),
                          const std::string& checksum = "");
    void reportError(const std::string& workloadId, const std::string& message);
    void reportChunkError(const std::string& parentId, const std::string& chunkId,
                         const std::string& message);

    // Event handler setters
    void setOnConnected(std::function<void()> handler) { onConnected = handler; }
    void setOnDisconnected(std::function<void()> handler) { onDisconnected = handler; }
    void setOnMessage(std::function<void(const std::string&)> handler) { onMessage = handler; }
    void setOnRegister(std::function<void(const json&)> handler) { onRegister = handler; }
    void setOnTaskAssigned(std::function<void(const json&)> handler) { onTaskAssigned = handler; }
    void setOnWorkloadAssigned(std::function<void(const json&)> handler) { onWorkloadAssigned = handler; }
    void setOnChunkAssigned(std::function<void(const json&)> handler) { onChunkAssigned = handler; }
    void setOnTaskVerified(std::function<void(const json&)> handler) { onTaskVerified = handler; }
    void setOnTaskSubmitted(std::function<void(const json&)> handler) { onTaskSubmitted = handler; }
    void setOnWorkloadComplete(std::function<void(const json&)> handler) { onWorkloadComplete = handler; }

private:
    void runEventLoop();
};
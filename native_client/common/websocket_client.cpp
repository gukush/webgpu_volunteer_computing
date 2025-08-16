// common/websocket_client.cpp
#include "websocket_client.hpp"
#include <iostream>
#include <boost/beast/core.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/websocket/ssl.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/url.hpp>

namespace beast = boost::beast;
namespace websocket = beast::websocket;
namespace net = boost::asio;
namespace ssl = boost::asio::ssl;
using tcp = boost::asio::ip::tcp;

WebSocketClient::WebSocketClient() {
    // Configure SSL context to accept self-signed certificates
    ctx.set_verify_mode(ssl::verify_none);
    ctx.set_options(ssl::context::default_workarounds |
                   ssl::context::no_sslv2 |
                   ssl::context::no_sslv3 |
                   ssl::context::single_dh_use);
}

WebSocketClient::~WebSocketClient() {
    disconnect();
}

bool WebSocketClient::connect(const std::string& host, const std::string& port, const std::string& target) {
    try {
        // Resolve hostname
        auto const results = resolver.resolve(host, port);
        
        // Get the underlying socket
        auto& socket = beast::get_lowest_layer(ws);
        
        // Make the connection on the IP address we get from a lookup
        auto ep = net::connect(socket, results);
        
        // Update the host string for SNI
        std::string hostWithPort = host + ':' + std::to_string(ep.port());
        
        // Set SNI Hostname (many hosts need this to handshake successfully)
        if (!SSL_set_tlsext_host_name(ws.next_layer().native_handle(), host.c_str())) {
            beast::error_code ec{static_cast<int>(::ERR_get_error()), net::error::get_ssl_category()};
            throw beast::system_error{ec};
        }
        
        // Perform the SSL handshake
        ws.next_layer().handshake(ssl::stream_base::client);
        
        // Set a decorator to change the User-Agent of the handshake
        ws.set_option(websocket::stream_base::decorator(
            [](websocket::request_type& req) {
                req.set(beast::http::field::user_agent, "MultiFramework-Native-Client/1.0");
            }));
        
        // Perform the websocket handshake
        ws.handshake(hostWithPort, target);
        
        // Start the event loop in a separate thread
        shouldStop = false;
        ioThread = std::thread(&WebSocketClient::runEventLoop, this);
        
        if (onConnected) {
            onConnected();
        }
        
        return true;
        
    } catch (std::exception const& e) {
        std::cerr << "WebSocket connection error: " << e.what() << std::endl;
        return false;
    }
}

void WebSocketClient::disconnect() {
    shouldStop = true;
    
    if (ws.is_open()) {
        try {
            ws.close(websocket::close_code::normal);
        } catch (...) {
            // Ignore errors during close
        }
    }
    
    if (ioThread.joinable()) {
        ioThread.join();
    }
    
    if (onDisconnected) {
        onDisconnected();
    }
}

void WebSocketClient::send(const std::string& message) {
    if (!ws.is_open()) {
        std::cerr << "WebSocket not connected, cannot send message" << std::endl;
        return;
    }
    
    try {
        ws.write(net::buffer(message));
    } catch (std::exception const& e) {
        std::cerr << "WebSocket send error: " << e.what() << std::endl;
    }
}

void WebSocketClient::runEventLoop() {
    beast::flat_buffer buffer;
    
    while (!shouldStop && ws.is_open()) {
        try {
            // Read a message
            ws.read(buffer);
            
            // Convert to string
            std::string message = beast::buffers_to_string(buffer.data());
            buffer.clear();
            
            // Call message handler
            if (onMessage) {
                onMessage(message);
            }
            
        } catch (beast::system_error const& se) {
            if (se.code() != websocket::error::closed) {
                std::cerr << "WebSocket read error: " << se.code().message() << std::endl;
            }
            break;
        } catch (std::exception const& e) {
            std::cerr << "WebSocket event loop error: " << e.what() << std::endl;
            break;
        }
    }
}

// ============================================================================
// Framework Client Implementation
// ============================================================================

// common/framework_client.cpp
#include "framework_client.hpp"
#include <iostream>
#include <boost/url.hpp>

FrameworkClient::FrameworkClient(std::unique_ptr<IFrameworkExecutor> exec) 
    : executor(std::move(exec)) {
    wsClient = std::make_unique<WebSocketClient>();
    
    // Setup WebSocket event handlers
    wsClient->setOnConnected([this]() { onConnected(); });
    wsClient->setOnDisconnected([this]() { onDisconnected(); });
    wsClient->setOnMessage([this](const std::string& msg) { onMessage(msg); });
}

FrameworkClient::~FrameworkClient() {
    disconnect();
}

bool FrameworkClient::connect(const std::string& url) {
    serverUrl = url;
    
    try {
        // Parse WebSocket URL
        boost::urls::url parsedUrl(url);
        std::string scheme = parsedUrl.scheme();
        std::string host = parsedUrl.host();
        std::string port = parsedUrl.has_port() ? parsedUrl.port() : (scheme == "wss" ? "443" : "80");
        std::string target = parsedUrl.path();
        
        if (target.empty()) {
            target = "/";
        }
        
        // Add socket.io path if not present
        if (target.find("socket.io") == std::string::npos) {
            target += "socket.io/?EIO=4&transport=websocket";
        }
        
        return wsClient->connect(host, port, target);
        
    } catch (std::exception const& e) {
        std::cerr << "Failed to parse URL: " << e.what() << std::endl;
        return false;
    }
}

void FrameworkClient::disconnect() {
    if (wsClient) {
        wsClient->disconnect();
    }
    connected = false;
}

void FrameworkClient::run() {
    // Keep the main thread alive while connected
    while (connected) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void FrameworkClient::onConnected() {
    connected = true;
    std::cout << "Connected to server" << std::endl;
    
    // Send initial registration
    json registerMsg = {
        {"type", "client:join"},
        {"data", {
            {"gpuInfo", executor->getCapabilities()},
            {"hasWebGPU", false},
            {"supportedFrameworks", {executor->getFrameworkName()}},
            {"clientType", "native"}
        }}
    };
    
    // Socket.io message format: "42" + JSON
    std::string message = "42" + registerMsg.dump();
    wsClient->send(message);
}

void FrameworkClient::onDisconnected() {
    connected = false;
    std::cout << "Disconnected from server" << std::endl;
}

void FrameworkClient::onMessage(const std::string& message) {
    try {
        // Parse Socket.io message format
        if (message.length() < 2 || message.substr(0, 2) != "42") {
            return; // Not a data message
        }
        
        json data = json::parse(message.substr(2));
        
        if (!data.is_array() || data.size() < 2) {
            return;
        }
        
        std::string eventType = data[0];
        json eventData = data[1];
        
        std::cout << "Received event: " << eventType << std::endl;
        
        if (eventType == "register") {
            clientId = eventData["clientId"];
            std::cout << "Registered with client ID: " << clientId << std::endl;
            
        } else if (eventType == "workload:new") {
            handleWorkloadAssignment(eventData);
            
        } else if (eventType == "workload:chunk_assign") {
            handleChunkAssignment(eventData);
            
        } else if (eventType == "task:assign") {
            // Matrix multiplication task - could implement if needed
            std::cout << "Matrix task assignment not implemented for native clients" << std::endl;
        }
        
    } catch (json::exception const& e) {
        std::cerr << "JSON parsing error: " << e.what() << std::endl;
    } catch (std::exception const& e) {
        std::cerr << "Message handling error: " << e.what() << std::endl;
    }
}

void FrameworkClient::handleWorkloadAssignment(const json& data) {
    if (busy) {
        // Send busy response
        json response = {
            {"type", "workload:busy"},
            {"data", {
                {"id", data["id"]},
                {"reason", "busy"}
            }}
        };
        wsClient->send("42" + response.dump());
        return;
    }
    
    try {
        TaskData task;
        task.id = data["id"];
        task.framework = data["framework"];
        task.kernel = data["kernel"];
        task.entry = data["entry"];
        task.workgroupCount = data["workgroupCount"];
        task.bindLayout = data["bindLayout"];
        task.outputSize = data["outputSize"];
        task.compilationOptions = data.value("compilationOptions", json::object());
        
        // Decode input data if present
        if (data.contains("input") && !data["input"].is_null()) {
            std::string inputBase64 = data["input"];
            task.inputData = base64_decode(inputBase64);
        }
        
        // Check framework compatibility
        if (task.framework != executor->getFrameworkName()) {
            json response = {
                {"type", "workload:error"},
                {"data", {
                    {"id", task.id},
                    {"message", "Framework mismatch: expected " + executor->getFrameworkName() + ", got " + task.framework}
                }}
            };
            wsClient->send("42" + response.dump());
            return;
        }
        
        busy = true;
        activeTasks[task.id] = task;
        
        std::cout << "Executing " << task.framework << " workload: " << task.id << std::endl;
        
        // Execute in separate thread to avoid blocking
        std::thread([this, task]() {
            TaskResult result = executor->executeTask(task);
            sendResult(task, result);
            
            busy = false;
            activeTasks.erase(task.id);
        }).detach();
        
    } catch (std::exception const& e) {
        std::cerr << "Workload assignment error: " << e.what() << std::endl;
        
        json response = {
            {"type", "workload:error"},
            {"data", {
                {"id", data["id"]},
                {"message", std::string("Assignment error: ") + e.what()}
            }}
        };
        wsClient->send("42" + response.dump());
    }
}

void FrameworkClient::handleChunkAssignment(const json& data) {
    if (busy) {
        json response = {
            {"type", "workload:chunk_error"},
            {"data", {
                {"parentId", data["parentId"]},
                {"chunkId", data["chunkId"]},
                {"message", "busy"}
            }}
        };
        wsClient->send("42" + response.dump());
        return;
    }
    
    try {
        TaskData task;
        task.id = data["chunkId"];
        task.parentId = data["parentId"];
        task.framework = data["framework"];
        task.kernel = data["wgsl"]; // Legacy field name
        task.entry = data["entry"];
        task.workgroupCount = data["workgroupCount"];
        task.bindLayout = data["bindLayout"];
        task.outputSize = data.value("outputSize", 1024); // Estimate for chunks
        task.chunkUniforms = data["chunkUniforms"];
        task.isChunk = true;
        task.chunkId = data["chunkId"];
        task.chunkOrderIndex = data["chunkOrderIndex"];
        
        // Decode chunk input data
        std::string inputBase64 = data["inputData"];
        task.inputData = base64_decode(inputBase64);
        
        busy = true;
        activeTasks[task.id] = task;
        
        std::cout << "Executing chunk: " << task.chunkId << std::endl;
        
        // Execute in separate thread
        std::thread([this, task]() {
            TaskResult result = executor->executeTask(task);
            
            if (result.success) {
                json response = {
                    {"type", "workload:chunk_done"},
                    {"data", {
                        {"parentId", task.parentId},
                        {"chunkId", task.chunkId},
                        {"chunkOrderIndex", task.chunkOrderIndex},
                        {"result", base64_encode(result.outputData)},
                        {"processingTime", result.processingTime}
                    }}
                };
                wsClient->send("42" + response.dump());
            } else {
                json response = {
                    {"type", "workload:chunk_error"},
                    {"data", {
                        {"parentId", task.parentId},
                        {"chunkId", task.chunkId},
                        {"message", result.errorMessage}
                    }}
                };
                wsClient->send("42" + response.dump());
            }
            
            busy = false;
            activeTasks.erase(task.id);
        }).detach();
        
    } catch (std::exception const& e) {
        std::cerr << "Chunk assignment error: " << e.what() << std::endl;
        
        json response = {
            {"type", "workload:chunk_error"},
            {"data", {
                {"parentId", data["parentId"]},
                {"chunkId", data["chunkId"]},
                {"message", std::string("Assignment error: ") + e.what()}
            }}
        };
        wsClient->send("42" + response.dump());
    }
}

void FrameworkClient::sendResult(const TaskData& task, const TaskResult& result) {
    if (result.success) {
        json response = {
            {"type", "workload:done"},
            {"data", {
                {"id", task.id},
                {"result", base64_encode(result.outputData)},
                {"processingTime", result.processingTime}
            }}
        };
        wsClient->send("42" + response.dump());
        std::cout << "Workload " << task.id << " completed in " 
                  << result.processingTime << "ms" << std::endl;
    } else {
        sendError(task, result.errorMessage);
    }
}

void FrameworkClient::sendError(const TaskData& task, const std::string& error) {
    json response = {
        {"type", "workload:error"},
        {"data", {
            {"id", task.id},
            {"message", error}
        }}
    };
    wsClient->send("42" + response.dump());
    std::cerr << "Workload " << task.id << " failed: " << error << std::endl;
}

// ============================================================================
// Base64 Encoding/Decoding Utilities
// ============================================================================

// common/base64.hpp
#pragma once
#include <string>
#include <vector>

std::string base64_encode(const std::vector<uint8_t>& data);
std::vector<uint8_t> base64_decode(const std::string& encoded);

// common/base64.cpp
#include "base64.hpp"
#include <algorithm>

static const char base64_chars[] = 
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789+/";

static inline bool is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

std::string base64_encode(const std::vector<uint8_t>& data) {
    std::string ret;
    int i = 0;
    int j = 0;
    unsigned char char_array_3[3];
    unsigned char char_array_4[4];
    int len = data.size();
    const unsigned char* bytes_to_encode = data.data();

    while (len--) {
        char_array_3[i++] = *(bytes_to_encode++);
        if (i == 3) {
            char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
            char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
            char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
            char_array_4[3] = char_array_3[2] & 0x3f;

            for(i = 0; (i <4) ; i++)
                ret += base64_chars[char_array_4[i]];
            i = 0;
        }
    }

    if (i) {
        for(j = i; j < 3; j++)
            char_array_3[j] = '\0';

        char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
        char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
        char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
        char_array_4[3] = char_array_3[2] & 0x3f;

        for (j = 0; (j < i + 1); j++)
            ret += base64_chars[char_array_4[j]];

        while((i++ < 3))
            ret += '=';
    }

    return ret;
}

std::vector<uint8_t> base64_decode(const std::string& encoded_string) {
    int len = encoded_string.size();
    int i = 0;
    int j = 0;
    int in = 0;
    unsigned char char_array_4[4], char_array_3[3];
    std::vector<uint8_t> ret;

    while (len-- && ( encoded_string[in] != '=') && is_base64(encoded_string[in])) {
        char_array_4[i++] = encoded_string[in]; in++;
        if (i ==4) {
            for (i = 0; i <4; i++)
                char_array_4[i] = std::find(base64_chars, base64_chars + 64, char_array_4[i]) - base64_chars;

            char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
            char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
            char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

            for (i = 0; (i < 3); i++)
                ret.push_back(char_array_3[i]);
            i = 0;
        }
    }

    if (i) {
        for (j = i; j <4; j++)
            char_array_4[j] = 0;

        for (j = 0; j <4; j++)
            char_array_4[j] = std::find(base64_chars, base64_chars + 64, char_array_4[j]) - base64_chars;

        char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
        char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
        char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];

        for (j = 0; (j < i - 1); j++) ret.push_back(char_array_3[j]);
    }

    return ret;
}

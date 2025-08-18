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
        if (data.contains("workgroupCount") && data["workgroupCount"].is_array()) {
            task.workgroupCount = data["workgroupCount"].get<std::vector<int>>();
        } else {
            task.workgroupCount.clear();
        }
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

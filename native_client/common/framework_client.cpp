#include "framework_client.hpp"
#include <iostream>
#include <thread>
#include <chrono>
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
            if (!target.empty() && target.back() != '/') target.push_back('/');
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
    /*
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

    // Socket.io message format: "42" + JSON - FIXED: Added missing quotes around client:join
    std::string message = ("42[\"client:join\", " + registerMsg.dump() + "]");
    wsClient->send(message);
    */
   joinSent = false;
}

void FrameworkClient::onDisconnected() {
    connected = false;
    std::cout << "Disconnected from server" << std::endl;
}

void FrameworkClient::onMessage(const std::string& message) {
    try {
        if (message.rfind("40", 0) == 0 && !joinSent) {
            json joinPayload = {
                {"gpuInfo", executor->getCapabilities()},
                {"hasWebGPU", false},
                {"supportedFrameworks", {executor->getFrameworkName()}},
                {"clientType", "native"}
            };
            std::string joinMsg = std::string("42[\"client:join\",") + joinPayload.dump() + "]";
            wsClient->send(joinMsg);
            joinSent = true;
            return;
        }
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

std::vector<std::vector<uint8_t>> FrameworkClient::decodeInputs(const json& data) {
    std::vector<std::vector<uint8_t>> inputs;

    // Preferred: inputs as array of { name, data }
    if (data.contains("inputs") && data["inputs"].is_array()) {
        for (const auto& item : data["inputs"]) {
            if (item.is_object() && item.contains("data") && item["data"].is_string()) {
                auto b64 = item["data"].get<std::string>();
                if (!b64.empty()) inputs.push_back(base64_decode(b64));
            } else if (item.is_string()) {
                auto b64 = item.get<std::string>();
                if (!b64.empty()) inputs.push_back(base64_decode(b64));
            }
        }
    }
    // Alternative: chunkInputs object, optionally guided by schema.inputs order
    else if (data.contains("chunkInputs") && data["chunkInputs"].is_object()) {
        if (data.contains("schema") && data["schema"].contains("inputs") && data["schema"]["inputs"].is_array()) {
            for (const auto& def : data["schema"]["inputs"]) {
                if (!def.contains("name")) continue;
                auto name = def["name"].get<std::string>();
                auto it = data["chunkInputs"].find(name);
                if (it != data["chunkInputs"].end() && it->is_string()) {
                    auto b64 = it->get<std::string>();
                    if (!b64.empty()) inputs.push_back(base64_decode(b64));
                }
            }
        } else {
            for (auto it = data["chunkInputs"].begin(); it != data["chunkInputs"].end(); ++it) {
                if (it.value().is_string()) {
                    auto b64 = it.value().get<std::string>();
                    if (!b64.empty()) inputs.push_back(base64_decode(b64));
                }
            }
        }
    }
    // Legacy single-field fallbacks
    else if (data.contains("input") && data["input"].is_string()) {
        auto b64 = data["input"].get<std::string>();
        if (!b64.empty()) inputs.push_back(base64_decode(b64));
    } else if (data.contains("inputData") && data["inputData"].is_string()) {
        auto b64 = data["inputData"].get<std::string>();
        if (!b64.empty()) inputs.push_back(base64_decode(b64));
    }

    return inputs;
}

// FIXED: Added missing opening brace
TaskData FrameworkClient::parseTaskData(const json& data, bool isChunk) {
    TaskData task;

    if (isChunk) {
        task.id = data["chunkId"];
        task.parentId = data["parentId"];
        task.chunkId = data["chunkId"];
        task.chunkOrderIndex = data.value("chunkOrderIndex", -1);
        task.isChunk = true;
    } else {
        task.id = data["id"];
    }

    task.framework = data["framework"];
    task.kernel = data.contains("kernel") ? data["kernel"].get<std::string>() : data["wgsl"].get<std::string>();
    task.entry = data.contains("entry") ? data["entry"].get<std::string>() : (task.framework == "opencl" ? std::string("block_matrix_multiply") : std::string("main"));
    task.bindLayout = data["bindLayout"];
    task.compilationOptions = data.value("compilationOptions", json::object());

    if (isChunk) {
        task.chunkUniforms = data.value("chunkUniforms", json::object());
    }
    // Merge metadata (e.g., block_size, matrix_size) into uniforms for executors
    if (data.contains("metadata") && data["metadata"].is_object()) {
        for (auto it = data["metadata"].begin(); it != data["metadata"].end(); ++it) {
            task.chunkUniforms[it.key()] = it.value();
        }
    }

    // Work sizes: prefer OpenCL-style globalWorkSize, else fallback to workgroupCount
    if (data.contains("globalWorkSize") && data["globalWorkSize"].is_array()) {
        task.workgroupCount = data["globalWorkSize"].get<std::vector<int>>();
    } else if (data.contains("workgroupCount") && data["workgroupCount"].is_array()) {
        task.workgroupCount = data["workgroupCount"].get<std::vector<int>>();
    } else {
        task.workgroupCount = {1, 1, 1};
    }

    // NEW: Parse multi-input data
    task.inputData = decodeInputs(data);

    // NEW: Parse multi-output sizes
    if (data.contains("outputSizes") && data["outputSizes"].is_array()) {
        task.outputSizes = data["outputSizes"].get<std::vector<size_t>>();
    } else if (data.contains("outputSize")) {
        // Legacy single output
        task.outputSizes = {data["outputSize"].get<size_t>()};
    } else {
        // Default output size
        task.outputSizes = {1024};
    }

    // Set legacy fields for backward compatibility
    if (!task.inputData.empty()) {
        task.legacyInputData = task.inputData[0];
    }
    if (!task.outputSizes.empty()) {
        task.legacyOutputSize = task.outputSizes[0];
    }

    return task;
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
        TaskData task = parseTaskData(data, false);

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

        std::cout << "Executing " << task.framework << " workload: " << task.id
                  << " (" << task.getInputCount() << " inputs, " << task.getOutputCount() << " outputs)" << std::endl;

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
        TaskData task = parseTaskData(data, true);

        busy = true;
        activeTasks[task.id] = task;

        std::cout << "Executing chunk: " << task.chunkId
                  << " (" << task.getInputCount() << " inputs, " << task.getOutputCount() << " outputs)" << std::endl;

        // Execute in separate thread
        std::thread([this, task]() {
            TaskResult result = executor->executeTask(task);

            if (result.success) {
                json responseData = {
                    {"parentId", task.parentId},
                    {"chunkId", task.chunkId},
                    {"chunkOrderIndex", task.chunkOrderIndex},
                    {"processingTime", result.processingTime}
                };

                // NEW: Send multi-output results
                if (result.hasMultipleOutputs()) {
                    json resultsArray = json::array();
                    for (const auto& output : result.outputData) {
                        resultsArray.push_back(base64_encode(output));
                    }
                    responseData["results"] = resultsArray;

                    // Also send single result for backward compatibility
                    if (!result.outputData.empty()) {
                        responseData["result"] = base64_encode(result.outputData[0]);
                    }
                } else {
                    // Single output (legacy format)
                    if (!result.outputData.empty()) {
                        responseData["result"] = base64_encode(result.outputData[0]);
                    } else if (!result.legacyOutputData.empty()) {
                        responseData["result"] = base64_encode(result.legacyOutputData);
                    }
                }

                json response = {
                    {"type", "workload:chunk_done"},
                    {"data", responseData}
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
        json responseData = {
            {"id", task.id},
            {"processingTime", result.processingTime}
        };

        // NEW: Send multi-output results
        if (result.hasMultipleOutputs()) {
            json resultsArray = json::array();
            for (const auto& output : result.outputData) {
                resultsArray.push_back(base64_encode(output));
            }
            responseData["results"] = resultsArray;

            // Also send single result for backward compatibility
            if (!result.outputData.empty()) {
                responseData["result"] = base64_encode(result.outputData[0]);
            }
        } else {
            // Single output (legacy format)
            if (!result.outputData.empty()) {
                responseData["result"] = base64_encode(result.outputData[0]);
            } else if (!result.legacyOutputData.empty()) {
                responseData["result"] = base64_encode(result.legacyOutputData);
            }
        }

        json response = {
            {"type", "workload:done"},
            {"data", responseData}
        };
        wsClient->send("42" + response.dump());

        std::cout << "Workload " << task.id << " completed in "
                  << result.processingTime << "ms (" << result.getOutputCount() << " outputs)" << std::endl;
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
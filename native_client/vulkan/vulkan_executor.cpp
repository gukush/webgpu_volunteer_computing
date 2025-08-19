#include "vulkan_executor.hpp"
#include <iostream>
#include <fstream>
#include <chrono>
#include <shaderc/shaderc.hpp>

VulkanExecutor::VulkanExecutor() {}

VulkanExecutor::~VulkanExecutor() {
    cleanup();
}

bool VulkanExecutor::initialize(const json& config) {
    if (initialized) return true;

    if (!createInstance()) {
        std::cerr << "Failed to create Vulkan instance" << std::endl;
        return false;
    }

    if (!selectPhysicalDevice()) {
        std::cerr << "Failed to select Vulkan physical device" << std::endl;
        return false;
    }

    if (!createLogicalDevice()) {
        std::cerr << "Failed to create Vulkan logical device" << std::endl;
        return false;
    }

    // Create command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = computeQueueFamilyIndex;

    if (vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan command pool" << std::endl;
        return false;
    }

    // Create descriptor pool
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 1000; // Max descriptors we might need

    VkDescriptorPoolCreateInfo descriptorPoolInfo{};
    descriptorPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolInfo.poolSizeCount = 1;
    descriptorPoolInfo.pPoolSizes = &poolSize;
    descriptorPoolInfo.maxSets = 1000;
    descriptorPoolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

    if (vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor pool" << std::endl;
        return false;
    }

    initialized = true;
    std::cout << "Vulkan initialized successfully" << std::endl;
    return true;
}

bool VulkanExecutor::createInstance() {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Multi-Framework Compute Client";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;

    return vkCreateInstance(&createInfo, nullptr, &instance) == VK_SUCCESS;
}

bool VulkanExecutor::selectPhysicalDevice() {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);

    if (deviceCount == 0) {
        std::cerr << "No Vulkan physical devices found" << std::endl;
        return false;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());

    // Select the first device with compute queue support
    for (const auto& dev : devices) {
        uint32_t queueFamilyCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, nullptr);

        std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &queueFamilyCount, queueFamilies.data());

        for (uint32_t i = 0; i < queueFamilies.size(); i++) {
            if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                physicalDevice = dev;
                computeQueueFamilyIndex = i;

                // Get device properties for capabilities
                vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
                vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

                return true;
            }
        }
    }

    return false;
}

bool VulkanExecutor::createLogicalDevice() {
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = computeQueueFamilyIndex;
    queueCreateInfo.queueCount = 1;

    float queuePriority = 1.0f;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceFeatures deviceFeatures{};

    VkDeviceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    createInfo.pQueueCreateInfos = &queueCreateInfo;
    createInfo.queueCreateInfoCount = 1;
    createInfo.pEnabledFeatures = &deviceFeatures;

    if (vkCreateDevice(physicalDevice, &createInfo, nullptr, &device) != VK_SUCCESS) {
        return false;
    }

    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);
    return true;
}

uint32_t VulkanExecutor::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) &&
            (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

bool VulkanExecutor::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                 VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        return false;
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties);

    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        vkDestroyBuffer(device, buffer, nullptr);
        return false;
    }

    vkBindBufferMemory(device, buffer, bufferMemory, 0);
    return true;
}

void VulkanExecutor::cleanup() {
    if (!initialized) return;

    // Wait for device to be idle
    if (device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(device);
    }

    // Cleanup cached shaders
    for (auto& [key, shader] : shaderCache) {
        if (shader.computePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(device, shader.computePipeline, nullptr);
        }
        if (shader.pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device, shader.pipelineLayout, nullptr);
        }
        if (shader.descriptorSetLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device, shader.descriptorSetLayout, nullptr);
        }
        if (shader.shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device, shader.shaderModule, nullptr);
        }
    }
    shaderCache.clear();

    if (descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(device, descriptorPool, nullptr);
        descriptorPool = VK_NULL_HANDLE;
    }

    if (commandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, commandPool, nullptr);
        commandPool = VK_NULL_HANDLE;
    }

    if (device != VK_NULL_HANDLE) {
        vkDestroyDevice(device, nullptr);
        device = VK_NULL_HANDLE;
    }

    if (instance != VK_NULL_HANDLE) {
        vkDestroyInstance(instance, nullptr);
        instance = VK_NULL_HANDLE;
    }

    initialized = false;
}

bool VulkanExecutor::compileShader(const std::string& source, const std::string& entryPoint,
                                  const json& compileOpts, CompiledShader& result) {
    // Initialize shaderc compiler
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;

    // Set compilation options
    options.SetOptimizationLevel(shaderc_optimization_level_performance);
    options.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_2);
    options.SetTargetSpirv(shaderc_spirv_version_1_5);

    // Add user-specified options
    if (compileOpts.contains("optimization")) {
        std::string opt = compileOpts["optimization"];
        if (opt == "none") {
            options.SetOptimizationLevel(shaderc_optimization_level_zero);
        } else if (opt == "size") {
            options.SetOptimizationLevel(shaderc_optimization_level_size);
        }
    }

    // Compile GLSL to SPIR-V
    shaderc::SpvCompilationResult module = compiler.CompileGlslToSpv(
        source, shaderc_compute_shader, "shader.comp", entryPoint.c_str(), options);

    if (module.GetCompilationStatus() != shaderc_compilation_status_success) {
        std::cerr << "Vulkan shader compilation failed: " << module.GetErrorMessage() << std::endl;
        return false;
    }

    // Create Vulkan shader module
    std::vector<uint32_t> spirvCode(module.cbegin(), module.cend());

    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = spirvCode.size() * sizeof(uint32_t);
    createInfo.pCode = spirvCode.data();

    if (vkCreateShaderModule(device, &createInfo, nullptr, &result.shaderModule) != VK_SUCCESS) {
        std::cerr << "Failed to create Vulkan shader module" << std::endl;
        return false;
    }

    // Create descriptor set layout (assuming input and output buffers)
    std::vector<VkDescriptorSetLayoutBinding> bindings;

    // Input buffer binding (if needed)
    VkDescriptorSetLayoutBinding inputBinding{};
    inputBinding.binding = 0;
    inputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    inputBinding.descriptorCount = 1;
    inputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(inputBinding);

    // Output buffer binding
    VkDescriptorSetLayoutBinding outputBinding{};
    outputBinding.binding = 1;
    outputBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    outputBinding.descriptorCount = 1;
    outputBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings.push_back(outputBinding);

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &result.descriptorSetLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create descriptor set layout" << std::endl;
        vkDestroyShaderModule(device, result.shaderModule, nullptr);
        return false;
    }

    // Create pipeline layout
    VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &result.descriptorSetLayout;

    if (vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &result.pipelineLayout) != VK_SUCCESS) {
        std::cerr << "Failed to create pipeline layout" << std::endl;
        vkDestroyDescriptorSetLayout(device, result.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, result.shaderModule, nullptr);
        return false;
    }

    // Create compute pipeline
    VkPipelineShaderStageCreateInfo shaderStageInfo{};
    shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shaderStageInfo.module = result.shaderModule;
    shaderStageInfo.pName = entryPoint.c_str();

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage = shaderStageInfo;
    pipelineInfo.layout = result.pipelineLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &result.computePipeline) != VK_SUCCESS) {
        std::cerr << "Failed to create compute pipeline" << std::endl;
        vkDestroyPipelineLayout(device, result.pipelineLayout, nullptr);
        vkDestroyDescriptorSetLayout(device, result.descriptorSetLayout, nullptr);
        vkDestroyShaderModule(device, result.shaderModule, nullptr);
        return false;
    }

    return true;
}

TaskResult VulkanExecutor::executeTask(const TaskData& task) {
    TaskResult result;
    auto startTime = std::chrono::high_resolution_clock::now();

    if (!initialized) {
        result.success = false;
        result.errorMessage = "Vulkan not initialized";
        return result;
    }

    VkBuffer inputBuffer = VK_NULL_HANDLE, outputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory inputBufferMemory = VK_NULL_HANDLE, outputBufferMemory = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;

    try {
        // Compile shader if not cached
        std::string cacheKey = task.kernel + "|" + task.entry;
        auto it = shaderCache.find(cacheKey);
        CompiledShader* shader;

        if (it == shaderCache.end()) {
            CompiledShader newShader;
            if (!compileShader(task.kernel, task.entry, task.compilationOptions, newShader)) {
                result.success = false;
                result.errorMessage = "Shader compilation failed";
                return result;
            }
            shaderCache[cacheKey] = std::move(newShader);
            shader = &shaderCache[cacheKey];
        } else {
            shader = &it->second;
        }

        // Create buffers
        if (!task.inputData.empty()) {
            if (!createBuffer(task.inputData.size(),
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                            inputBuffer, inputBufferMemory)) {
                result.success = false;
                result.errorMessage = "Failed to create input buffer";
                return result;
            }

            // Upload input data
            void* data;
            vkMapMemory(device, inputBufferMemory, 0, task.inputData.size(), 0, &data);
            memcpy(data, task.inputData.data(), task.inputData.size());
            vkUnmapMemory(device, inputBufferMemory);
        }

        if (!createBuffer(task.outputSize,
                        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        outputBuffer, outputBufferMemory)) {
            result.success = false;
            result.errorMessage = "Failed to create output buffer";
            return result;
        }

        // Allocate descriptor set
        VkDescriptorSetAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        allocInfo.descriptorPool = descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &shader->descriptorSetLayout;

        if (vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet) != VK_SUCCESS) {
            result.success = false;
            result.errorMessage = "Failed to allocate descriptor set";
            return result;
        }

        // Update descriptor set
        std::vector<VkWriteDescriptorSet> descriptorWrites;

        if (inputBuffer != VK_NULL_HANDLE) {
            VkDescriptorBufferInfo inputBufferInfo{};
            inputBufferInfo.buffer = inputBuffer;
            inputBufferInfo.offset = 0;
            inputBufferInfo.range = task.inputData.size();

            VkWriteDescriptorSet inputWrite{};
            inputWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            inputWrite.dstSet = descriptorSet;
            inputWrite.dstBinding = 0;
            inputWrite.dstArrayElement = 0;
            inputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            inputWrite.descriptorCount = 1;
            inputWrite.pBufferInfo = &inputBufferInfo;

            descriptorWrites.push_back(inputWrite);
        }

        VkDescriptorBufferInfo outputBufferInfo{};
        outputBufferInfo.buffer = outputBuffer;
        outputBufferInfo.offset = 0;
        outputBufferInfo.range = task.outputSize;

        VkWriteDescriptorSet outputWrite{};
        outputWrite.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        outputWrite.dstSet = descriptorSet;
        outputWrite.dstBinding = 1;
        outputWrite.dstArrayElement = 0;
        outputWrite.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        outputWrite.descriptorCount = 1;
        outputWrite.pBufferInfo = &outputBufferInfo;

        descriptorWrites.push_back(outputWrite);

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(descriptorWrites.size()),
                              descriptorWrites.data(), 0, nullptr);

        // Allocate command buffer
        VkCommandBufferAllocateInfo cmdAllocInfo{};
        cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cmdAllocInfo.commandPool = commandPool;
        cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cmdAllocInfo.commandBufferCount = 1;

        if (vkAllocateCommandBuffers(device, &cmdAllocInfo, &commandBuffer) != VK_SUCCESS) {
            result.success = false;
            result.errorMessage = "Failed to allocate command buffer";
            return result;
        }

        // Record command buffer
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        vkBeginCommandBuffer(commandBuffer, &beginInfo);

        vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, shader->computePipeline);
        vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE,
                               shader->pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        // Dispatch compute shader
        uint32_t groupCountX = task.workgroupCount.size() > 0 ? task.workgroupCount[0] : 1;
        uint32_t groupCountY = task.workgroupCount.size() > 1 ? task.workgroupCount[1] : 1;
        uint32_t groupCountZ = task.workgroupCount.size() > 2 ? task.workgroupCount[2] : 1;

        vkCmdDispatch(commandBuffer, groupCountX, groupCountY, groupCountZ);

        vkEndCommandBuffer(commandBuffer);

        // Submit command buffer
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &commandBuffer;

        if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            result.success = false;
            result.errorMessage = "Failed to submit command buffer";
            return result;
        }

        // Wait for completion
        vkQueueWaitIdle(computeQueue);

        // Read output data
        void* outputData;
        vkMapMemory(device, outputBufferMemory, 0, task.outputSize, 0, &outputData);
        result.outputData.resize(task.outputSize);
        memcpy(result.outputData.data(), outputData, task.outputSize);
        vkUnmapMemory(device, outputBufferMemory);

        auto endTime = std::chrono::high_resolution_clock::now();
        result.processingTime = std::chrono::duration<double, std::milli>(endTime - startTime).count();
        result.success = true;

    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = std::string("Exception: ") + e.what();
    }

    // Cleanup
    if (commandBuffer != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(device, commandPool, 1, &commandBuffer);
    }
    if (descriptorSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
    }
    if (inputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, inputBuffer, nullptr);
    }
    if (inputBufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, inputBufferMemory, nullptr);
    }
    if (outputBuffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, outputBuffer, nullptr);
    }
    if (outputBufferMemory != VK_NULL_HANDLE) {
        vkFreeMemory(device, outputBufferMemory, nullptr);
    }

    return result;
}

json VulkanExecutor::getCapabilities() const {
    json caps;
    caps["framework"] = "vulkan";
    caps["initialized"] = initialized;

    if (initialized && physicalDevice != VK_NULL_HANDLE) {
        caps["device"] = {
            {"name", deviceProperties.deviceName},
            {"apiVersion", deviceProperties.apiVersion},
            {"driverVersion", deviceProperties.driverVersion},
            {"vendorID", deviceProperties.vendorID},
            {"deviceID", deviceProperties.deviceID},
            {"deviceType", deviceProperties.deviceType},
            {"maxComputeWorkGroupCount", {
                deviceProperties.limits.maxComputeWorkGroupCount[0],
                deviceProperties.limits.maxComputeWorkGroupCount[1],
                deviceProperties.limits.maxComputeWorkGroupCount[2]
            }},
            {"maxComputeWorkGroupSize", {
                deviceProperties.limits.maxComputeWorkGroupSize[0],
                deviceProperties.limits.maxComputeWorkGroupSize[1],
                deviceProperties.limits.maxComputeWorkGroupSize[2]
            }},
            {"maxComputeWorkGroupInvocations", deviceProperties.limits.maxComputeWorkGroupInvocations}
        };
    }

    return caps;
}
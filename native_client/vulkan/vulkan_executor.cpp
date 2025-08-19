#include "vulkan_executor.hpp"
#include <iostream>
#include <vector>
#include <cstring>
#include <chrono>
#include <stdexcept>
#include <algorithm>

// Shaderc
#include <shaderc/shaderc.hpp>

// ========== Utility: scoped VkResult check ==========
#define VK_CHECK(x) do { VkResult _ret = (x); if (_ret != VK_SUCCESS) { \
    throw std::runtime_error(std::string("Vulkan error: ") + #x + " -> " + std::to_string(_ret)); } } while(0)

VulkanExecutor::VulkanExecutor() {}
VulkanExecutor::~VulkanExecutor() { cleanup(); }

// -------- Instance & device --------
bool VulkanExecutor::createInstance(bool enableValidation, bool enableDebugUtils) {
    // Query available instance extensions
    uint32_t extCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &extCount, exts.data());

    auto hasExt = [&](const char* name){
        for (auto const& e : exts) if (strcmp(e.extensionName, name) == 0) return true;
        return false;
    };

    std::vector<const char*> enabledExts;
    if (enableDebugUtils && hasExt(VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) {
        enabledExts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    // portability enumeration is harmless if present (mostly for MoltenVK/macOS)
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    if (hasExt(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME)) {
        enabledExts.push_back(VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME);
    }
#endif

    VkApplicationInfo appInfo{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    appInfo.pApplicationName = "Multi-Framework Compute Client (Vulkan)";
    appInfo.applicationVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1,0,0);
    appInfo.apiVersion = VK_API_VERSION_1_1; // reasonable baseline for Linux/Windows in 2025

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &appInfo;
    ci.enabledExtensionCount = static_cast<uint32_t>(enabledExts.size());
    ci.ppEnabledExtensionNames = enabledExts.empty() ? nullptr : enabledExts.data();
#ifdef VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME
    if (std::find(enabledExts.begin(), enabledExts.end(), VK_KHR_PORTABILITY_ENUMERATION_EXTENSION_NAME) != enabledExts.end()) {
        ci.flags |= VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR;
    }
#endif

    VkResult res = vkCreateInstance(&ci, nullptr, &instance);
    if (res != VK_SUCCESS) {
        std::cerr << "vkCreateInstance failed: " << res << std::endl;
        return false;
    }
    return true;
}

bool VulkanExecutor::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance, &count, nullptr);
    if (count == 0) return false;
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance, &count, devs.data());

    // Prefer a device with a dedicated compute queue if possible
    auto pickScore = [&](VkPhysicalDevice pd)->int{
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        uint32_t qfCount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, nullptr);
        std::vector<VkQueueFamilyProperties> qf(qfCount);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qfCount, qf.data());

        bool hasComputeOnly = false;
        bool hasCompute = false;
        for (uint32_t i=0;i<qfCount;i++){
            if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
                hasCompute = true;
                if ((qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) == 0) hasComputeOnly = true;
            }
        }
        int score = 0;
        if (hasCompute) score += 10;
        if (hasComputeOnly) score += 5;
        // prefer discrete
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 3;
        return score;
    };

    int bestScore = -1;
    for (auto pd : devs) {
        int s = pickScore(pd);
        if (s > bestScore) { bestScore = s; physicalDevice = pd; }
    }
    if (physicalDevice == VK_NULL_HANDLE) return false;

    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);
    return true;
}

bool VulkanExecutor::createLogicalDevice() {
    uint32_t qfCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qfCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &qfCount, qf.data());

    bool found = false;
    for (uint32_t i=0;i<qfCount;i++){
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { computeQueueFamilyIndex = i; found = true; break; }
    }
    if (!found) return false;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = computeQueueFamilyIndex;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    VkPhysicalDeviceFeatures feats{}; // minimal
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &feats;

    VK_CHECK(vkCreateDevice(physicalDevice, &dci, nullptr, &device));
    vkGetDeviceQueue(device, computeQueueFamilyIndex, 0, &computeQueue);

    // Pipeline cache (optional)
    VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    VK_CHECK(vkCreatePipelineCache(device, &pci, nullptr, &pipelineCache));
    return true;
}

bool VulkanExecutor::createCommandPool() {
    VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    ci.queueFamilyIndex = computeQueueFamilyIndex;
    ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(device, &ci, nullptr, &commandPool));
    return true;
}

bool VulkanExecutor::createDescriptorPool() {
    // We assume at most 2 STORAGE_BUFFER per set (in/out). Size pool liberally.
    std::array<VkDescriptorPoolSize,1> sizes{};
    sizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    sizes[0].descriptorCount = 2048; // 1000 sets * 2 buffers (rounded up)

    VkDescriptorPoolCreateInfo ci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ci.poolSizeCount = static_cast<uint32_t>(sizes.size());
    ci.pPoolSizes = sizes.data();
    ci.maxSets = 1024;

    VK_CHECK(vkCreateDescriptorPool(device, &ci, nullptr, &descriptorPool));
    return true;
}

void VulkanExecutor::destroyDeviceObjects() {
    if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);
    if (pipelineCache) vkDestroyPipelineCache(device, pipelineCache, nullptr);
    if (device) vkDestroyDevice(device, nullptr);
    descriptorPool = VK_NULL_HANDLE;
    commandPool = VK_NULL_HANDLE;
    pipelineCache = VK_NULL_HANDLE;
    device = VK_NULL_HANDLE;
}

void VulkanExecutor::destroyInstance() {
    if (instance) vkDestroyInstance(instance, nullptr);
    instance = VK_NULL_HANDLE;
}

bool VulkanExecutor::initialize(const json& config) {
    if (initialized) return true;

    bool enableValidation = config.value("enableValidation", false);
    bool enableDebugUtils = config.value("enableDebugUtils", enableValidation);

    try {
        if (!createInstance(enableValidation, enableDebugUtils)) return false;
        if (!pickPhysicalDevice()) throw std::runtime_error("No suitable physical device");
        if (!createLogicalDevice()) throw std::runtime_error("Failed to create logical device");
        if (!createCommandPool()) throw std::runtime_error("Failed to create command pool");
        if (!createDescriptorPool()) throw std::runtime_error("Failed to create descriptor pool");
        initialized = true;
        std::cout << "Vulkan initialized: " << deviceProperties.deviceName << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "[initialize] " << e.what() << std::endl;
        // Clean up partially created resources
        destroyDeviceObjects();
        destroyInstance();
        initialized = false;
        return false;
    }
}

void VulkanExecutor::cleanup() {
    destroyDeviceObjects();
    destroyInstance();
    initialized = false;
}

// -------- Buffers --------
uint32_t VulkanExecutor::findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const {
    for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1u << i)) &&
            (memoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("No suitable memory type");
}

bool VulkanExecutor::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) const {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device, &bi, nullptr, &buffer));

    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(device, buffer, &req);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, properties);

    VK_CHECK(vkAllocateMemory(device, &ai, nullptr, &bufferMemory));
    VK_CHECK(vkBindBufferMemory(device, buffer, bufferMemory, 0));
    return true;
}

// -------- Shader compilation --------
static shaderc_env_version pick_env_from_api(uint32_t apiVersion) {
    uint32_t major = VK_VERSION_MAJOR(apiVersion);
    uint32_t minor = VK_VERSION_MINOR(apiVersion);
    if (major > 1 || (major==1 && minor >= 2)) return shaderc_env_version_vulkan_1_2;
    if (minor >= 1) return shaderc_env_version_vulkan_1_1;
    return shaderc_env_version_vulkan_1_0;
}

bool VulkanExecutor::compileGLSLtoSPIRV(const std::string& glsl, std::vector<uint32_t>& spirv, std::string& error) const {
    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetTargetEnvironment(shaderc_target_env_vulkan, pick_env_from_api(deviceProperties.apiVersion));
    // Let drivers legalize. Optimize for performance if requested later.
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    auto res = compiler.CompileGlslToSpv(glsl, shaderc_compute_shader, "kernel.comp", options);
    if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
        error = res.GetErrorMessage();
        return false;
    }
    spirv.assign(res.cbegin(), res.cend());
    return true;
}

// -------- Execute --------
TaskResult VulkanExecutor::executeTask(const TaskData& task) {
    TaskResult result;
    auto startTime = std::chrono::high_resolution_clock::now();

    if (!initialized) {
        result.success = false;
        result.errorMessage = "Vulkan not initialized";
        return result;
    }

    // Validate workgroup counts
    uint32_t gx = 1, gy = 1, gz = 1;
    if (!task.workgroupCount.empty()) {
        if (task.workgroupCount.size() > 0) gx = std::max(1, task.workgroupCount[0]);
        if (task.workgroupCount.size() > 1) gy = std::max(1, task.workgroupCount[1]);
        if (task.workgroupCount.size() > 2) gz = std::max(1, task.workgroupCount[2]);
    }
    // Clamp to device limits
    gx = std::min<uint32_t>(gx, deviceProperties.limits.maxComputeWorkGroupCount[0]);
    gy = std::min<uint32_t>(gy, deviceProperties.limits.maxComputeWorkGroupCount[1]);
    gz = std::min<uint32_t>(gz, deviceProperties.limits.maxComputeWorkGroupCount[2]);

    // Build descriptor set layout based on bindLayout string (server default: "vulkan-storage-buffer")
    // ABI: set=0, binding 0 -> optional readonly storage buffer (input)
    //      set=0, binding 1 -> storage buffer (output)
    VkDescriptorSetLayoutBinding bindings[2]{};
    uint32_t bindingCount = 0;

    bool hasInput = !task.inputData.empty();
    // Always provide output
    if (hasInput) {
        bindings[bindingCount].binding = 0;
        bindings[bindingCount].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[bindingCount].descriptorCount = 1;
        bindings[bindingCount].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        bindingCount++;
    }
    bindings[bindingCount].binding = hasInput ? 1u : 0u;
    bindings[bindingCount].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[bindingCount].descriptorCount = 1;
    bindings[bindingCount].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindingCount++;

    VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dslci.bindingCount = bindingCount;
    dslci.pBindings = bindings;
    VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    try {
        VK_CHECK(vkCreateDescriptorSetLayout(device, &dslci, nullptr, &descriptorSetLayout));
    } catch (...) {
        result.success = false;
        result.errorMessage = "Failed to create descriptor set layout";
        return result;
    }

    // Pipeline layout
    VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount = 1;
    plci.pSetLayouts = &descriptorSetLayout;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkShaderModule shaderModule = VK_NULL_HANDLE;

    // Resources
    VkBuffer inputBuffer = VK_NULL_HANDLE, outputBuffer = VK_NULL_HANDLE;
    VkDeviceMemory inputMem = VK_NULL_HANDLE, outputMem = VK_NULL_HANDLE;
    VkDescriptorSet descriptorSet = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;

    try {
        VK_CHECK(vkCreatePipelineLayout(device, &plci, nullptr, &pipelineLayout));

        // Compile GLSL (we only support GLSL text for Vulkan client)
        std::vector<uint32_t> spirv;
        std::string compileErr;
        if (!compileGLSLtoSPIRV(task.kernel, spirv, compileErr)) {
            throw std::runtime_error(std::string("GLSL->SPIR-V compile failed: ") + compileErr);
        }

        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spirv.size() * sizeof(uint32_t);
        smci.pCode = spirv.data();
        VK_CHECK(vkCreateShaderModule(device, &smci, nullptr, &shaderModule));

        VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = shaderModule;
        stage.pName = task.entry.empty() ? "main" : task.entry.c_str();

        VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpci.stage = stage;
        cpci.layout = pipelineLayout;
        VK_CHECK(vkCreateComputePipelines(device, pipelineCache, 1, &cpci, nullptr, &pipeline));

        // Buffers
        if (hasInput) {
            VK_CHECK(createBuffer(task.inputData.size(),
                                  VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                  VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                  inputBuffer, inputMem));
            // Map & write input
            void* p = nullptr;
            VkResult mapRes = vkMapMemory(device, inputMem, 0, task.inputData.size(), 0, &p);
            if (mapRes != VK_SUCCESS || !p) throw std::runtime_error("vkMapMemory failed for input buffer");
            std::memcpy(p, task.inputData.data(), task.inputData.size());
            vkUnmapMemory(device, inputMem);
        }

        if (task.outputSize == 0) throw std::runtime_error("outputSize must be > 0");
        VK_CHECK(createBuffer(task.outputSize,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                              VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                              outputBuffer, outputMem));

        // Allocate descriptor set
        VkDescriptorSetLayout layouts[1] = { descriptorSetLayout };
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = descriptorPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = layouts;
        VK_CHECK(vkAllocateDescriptorSets(device, &dsai, &descriptorSet));

        // Update descriptors
        std::vector<VkWriteDescriptorSet> writes;
        VkDescriptorBufferInfo inInfo{}, outInfo{};
        if (hasInput) {
            inInfo.buffer = inputBuffer; inInfo.offset = 0; inInfo.range = task.inputData.size();
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = descriptorSet;
            w.dstBinding = 0;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &inInfo;
            writes.push_back(w);
        }
        outInfo.buffer = outputBuffer; outInfo.offset = 0; outInfo.range = task.outputSize;
        {
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = descriptorSet;
            w.dstBinding = hasInput ? 1u : 0u;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &outInfo;
            writes.push_back(w);
        }
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

        // Command buffer
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = commandPool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(device, &cbai, &cmd));

        VkCommandBufferBeginInfo beginInfo{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        VK_CHECK(vkBeginCommandBuffer(cmd, &beginInfo));

        // Bind pipeline & descriptors
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);

        // Dispatch
        vkCmdDispatch(cmd, gx, gy, gz);

        // Barrier to make shader writes visible to host (explicit even if we wait on fence & coherent memory)
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_HOST_BIT,
                             0,
                             1, &mb,
                             0, nullptr,
                             0, nullptr);

        VK_CHECK(vkEndCommandBuffer(cmd));

        // Fence & submit
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(device, &fci, nullptr, &fence));

        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        VK_CHECK(vkQueueSubmit(computeQueue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_C(30'000'000'000))); // 30s

        // Read back output
        void* mapped = nullptr;
        VkResult mapRes = vkMapMemory(device, outputMem, 0, task.outputSize, 0, &mapped);
        if (mapRes != VK_SUCCESS || !mapped) throw std::runtime_error("vkMapMemory failed for output buffer");
        result.output.assign(reinterpret_cast<uint8_t*>(mapped), reinterpret_cast<uint8_t*>(mapped) + task.outputSize);
        vkUnmapMemory(device, outputMem);

        result.success = true;
        auto end = std::chrono::high_resolution_clock::now();
        result.computeTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - startTime).count();
        result.errorMessage.clear();
    } catch (const std::exception& e) {
        result.success = false;
        result.errorMessage = e.what();
    }

    // Cleanup per-task allocations
    if (fence) vkDestroyFence(device, fence, nullptr);
    if (cmd) vkFreeCommandBuffers(device, commandPool, 1, &cmd);
    if (descriptorSet) vkFreeDescriptorSets(device, descriptorPool, 1, &descriptorSet);
    if (shaderModule) vkDestroyShaderModule(device, shaderModule, nullptr);
    if (pipeline) vkDestroyPipeline(device, pipeline, nullptr);
    if (pipelineLayout) vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    if (descriptorSetLayout) vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
    if (inputBuffer) vkDestroyBuffer(device, inputBuffer, nullptr);
    if (inputMem) vkFreeMemory(device, inputMem, nullptr);
    if (outputBuffer) vkDestroyBuffer(device, outputBuffer, nullptr);
    if (outputMem) vkFreeMemory(device, outputMem, nullptr);

    return result;
}

json VulkanExecutor::getCapabilities() const {
    json caps;
    caps["framework"] = "vulkan";
    caps["initialized"] = initialized;
    if (!initialized || physicalDevice == VK_NULL_HANDLE) return caps;

    caps["device"] = {
        {"name", deviceProperties.deviceName},
        {"apiVersion", deviceProperties.apiVersion},
        {"driverVersion", deviceProperties.driverVersion},
        {"vendorID", deviceProperties.vendorID},
        {"deviceID", deviceProperties.deviceID},
        {"type", deviceProperties.deviceType}
    };
    caps["limits"] = {
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
    return caps;
}

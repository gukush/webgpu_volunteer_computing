#include "vulkan_executor.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <chrono>

#ifndef HAVE_SHADERC
  #error "This build requires shaderc: define HAVE_SHADERC"
#endif
#include <shaderc/shaderc.hpp>

namespace {
// Ensure the source contains a #version directive; default to 450 if absent.
std::string ensure_glsl_version(const std::string& src) {
    if (src.find("#version") != std::string::npos) return src;
    return std::string("#version 450\n") + src;
}

struct MacroDef { std::string name; std::string value; };
static std::vector<MacroDef> parse_macros(const json& opts) {
    std::vector<MacroDef> defs;
    try {
        if (opts.contains("macros") && opts["macros"].is_array()) {
            for (const auto& m : opts["macros"]) {
                if (m.is_string()) {
                    std::string s = m.get<std::string>();
                    auto pos = s.find('=');
                    if (pos == std::string::npos) defs.push_back({s, "1"});
                    else defs.push_back({s.substr(0,pos), s.substr(pos+1)});
                } else if (m.is_object()) {
                    for (auto it = m.begin(); it != m.end(); ++it) {
                        defs.push_back({it.key(), it.value().get<std::string>()});
                    }
                }
            }
        }
    } catch (...) {}
    return defs;
}

std::vector<uint32_t> compile_glsl_shaderc(const std::string& source, const std::string& entry, const std::vector<MacroDef>& macros) {
    shaderc::Compiler compiler;
    shaderc::CompileOptions opts;
    // Target Vulkan environment; change if you need a different Vulkan version
    opts.SetTargetEnvironment(shaderc_target_env_vulkan, shaderc_env_version_vulkan_1_1);
    opts.SetSourceLanguage(shaderc_source_language_glsl);
    opts.SetOptimizationLevel(shaderc_optimization_level_performance);

    for (const auto& d : macros) opts.AddMacroDefinition(d.name, d.value);

    // This code is used for compute shaders in this project. If you want
    // to support other stages add logic here to pick shaderc_shader_kind.
    shaderc_shader_kind kind = shaderc_compute_shader;

    auto res = compiler.CompileGlslToSpv(source, kind, "kernel.comp", entry.c_str(), opts);
    if (res.GetCompilationStatus() != shaderc_compilation_status_success) {
        throw std::runtime_error(std::string("GLSL->SPIR-V (shaderc) error: ") + res.GetErrorMessage());
    }
    return {res.cbegin(), res.cend()};
}

std::vector<uint32_t> compile_glsl_to_spirv(const std::string& source, const std::string& entry, const std::vector<MacroDef>& macros) {
    const std::string src = ensure_glsl_version(source);
    (void)entry; // compile_glsl_shaderc receives entry explicitly
    return compile_glsl_shaderc(src, entry, macros);
}

} // namespace

VulkanExecutor::VulkanExecutor(int preferredDeviceIndex)
    : preferredDeviceIndex_(preferredDeviceIndex) {}

VulkanExecutor::~VulkanExecutor() { cleanup(); }

bool VulkanExecutor::initialize(const json&) {
    if (!createInstance()) return false;
    if (!pickPhysicalDevice()) return false;
    if (!createDevice()) return false;
    return true;
}

void VulkanExecutor::cleanup() { destroyVulkan(); }

json VulkanExecutor::getCapabilities() const {
    json caps;
    caps["framework"] = "vulkan";
    caps["initialized"] = (device_ != VK_NULL_HANDLE);
    caps["supportsMultiInput"] = true;
    caps["supportsMultiOutput"] = true;
    caps["shaderSource"] = "glsl";
    caps["requiresBindings"] = "inputs first, then outputs (set=0, binding=i)";
    if (phys_) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(phys_, &props);
        caps["device"] = {
            {"apiVersion", (uint32_t)props.apiVersion},
            {"driverVersion", (uint32_t)props.driverVersion},
            {"vendorID", (uint32_t)props.vendorID},
            {"deviceID", (uint32_t)props.deviceID},
            {"deviceType", (uint32_t)props.deviceType},
            {"name", std::string(props.deviceName)},
            {"maxComputeWorkGroupCount", {
                props.limits.maxComputeWorkGroupCount[0],
                props.limits.maxComputeWorkGroupCount[1],
                props.limits.maxComputeWorkGroupCount[2]
            }}
        };
    }
    return caps;
}

bool VulkanExecutor::createInstance() {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "native-mimo-client";
    app.applicationVersion = VK_MAKE_VERSION(1,0,0);
    app.pEngineName = "none";
    app.engineVersion = VK_MAKE_VERSION(1,0,0);
    app.apiVersion = VK_API_VERSION_1_1;

    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    return vkCreateInstance(&ci, nullptr, &instance_) == VK_SUCCESS;
}

bool VulkanExecutor::pickPhysicalDevice() {
    uint32_t count = 0;
    vkEnumeratePhysicalDevices(instance_, &count, nullptr);
    if (!count) return false;
    std::vector<VkPhysicalDevice> devs(count);
    vkEnumeratePhysicalDevices(instance_, &count, devs.data());

    auto computeQ = [&](VkPhysicalDevice d)->std::optional<uint32_t>{
        uint32_t n=0; vkGetPhysicalDeviceQueueFamilyProperties(d, &n, nullptr);
        std::vector<VkQueueFamilyProperties> props(n);
        vkGetPhysicalDeviceQueueFamilyProperties(d, &n, props.data());
        for (uint32_t i=0;i<n;++i) if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) return i;
        return std::nullopt;
    };

    uint32_t picked = 0;
    if (preferredDeviceIndex_ >= 0 && preferredDeviceIndex_ < (int)devs.size()) {
        if (computeQ(devs[preferredDeviceIndex_]).has_value()) picked = (uint32_t)preferredDeviceIndex_;
    } else {
        for (uint32_t i=0;i<count;++i) if (computeQ(devs[i]).has_value()) { picked = i; break; }
    }
    phys_ = devs[picked];
    computeQueueFamily_ = computeQ(phys_).value();
    return true;
}

bool VulkanExecutor::createDevice() {
    float prio = 1.0f;
    VkDeviceQueueCreateInfo q{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    q.queueFamilyIndex = computeQueueFamily_;
    q.queueCount = 1;
    q.pQueuePriorities = &prio;

    VkDeviceCreateInfo ci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    ci.queueCreateInfoCount = 1;
    ci.pQueueCreateInfos = &q;

    if (vkCreateDevice(phys_, &ci, nullptr, &device_) != VK_SUCCESS) return false;
    vkGetDeviceQueue(device_, computeQueueFamily_, 0, &queue_);

    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.queueFamilyIndex = computeQueueFamily_;
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(device_, &pi, nullptr, &cmdPool_) != VK_SUCCESS) return false;
    return true;
}

void VulkanExecutor::destroyVulkan() {
    if (device_) {
        if (cmdPool_) { vkDestroyCommandPool(device_, cmdPool_, nullptr); cmdPool_ = VK_NULL_HANDLE; }
        vkDestroyDevice(device_, nullptr); device_ = VK_NULL_HANDLE;
    }
    if (instance_) { vkDestroyInstance(instance_, nullptr); instance_ = VK_NULL_HANDLE; }
}

uint32_t VulkanExecutor::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const {
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(phys_, &mp);
    for (uint32_t i=0;i<mp.memoryTypeCount;++i) {
        if ((typeBits & (1u<<i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    }
    throw std::runtime_error("No compatible memory type");
}

bool VulkanExecutor::createBuffer(VkDeviceSize size,
                                  VkBufferUsageFlags usage,
                                  VkMemoryPropertyFlags props,
                                  VkBuffer& outBuf, VkDeviceMemory& outMem) const {
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device_, &bi, nullptr, &outBuf) != VK_SUCCESS) return false;

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, outBuf, &req);

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = req.size;
    ai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);

    if (vkAllocateMemory(device_, &ai, nullptr, &outMem) != VK_SUCCESS) return false;
    vkBindBufferMemory(device_, outBuf, outMem, 0);
    return true;
}

VulkanExecutor::Buffer VulkanExecutor::makeHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage) const {
    Buffer b;
    b.size = size;
    if (!createBuffer(size, usage, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b.buf, b.mem))
        throw std::runtime_error("createBuffer failed");
    if (vkMapMemory(device_, b.mem, 0, size, 0, &b.mapped) != VK_SUCCESS)
        throw std::runtime_error("vkMapMemory failed");
    return b;
}

void VulkanExecutor::destroyBuffer(Buffer& b) const {
    if (!device_) return;
    if (b.mapped) { vkUnmapMemory(device_, b.mem); b.mapped = nullptr; }
    if (b.buf) { vkDestroyBuffer(device_, b.buf, nullptr); b.buf = VK_NULL_HANDLE; }
    if (b.mem) { vkFreeMemory(device_, b.mem, nullptr); b.mem = VK_NULL_HANDLE; }
}

TaskResult VulkanExecutor::executeTask(const TaskData& task) {
    TaskResult result{};
    auto t0 = std::chrono::high_resolution_clock::now();

    if (device_ == VK_NULL_HANDLE) {
        result.success = false;
        result.errorMessage = "Vulkan not initialized";
        return result;
    }

    VkDescriptorSetLayout dsetLayout = VK_NULL_HANDLE; VkPipelineLayout pipelineLayout = VK_NULL_HANDLE; VkDescriptorPool descPool = VK_NULL_HANDLE; VkShaderModule shaderModule = VK_NULL_HANDLE; VkPipeline pipeline = VK_NULL_HANDLE;
    try {
        // Determine IO counts
        const bool multipleInputs = !task.inputData.empty();
        const bool multipleOutputs = !task.outputSizes.empty();

        const size_t inputCount = multipleInputs ? task.inputData.size() : (task.legacyInputData.empty() ? 0 : 1);
        const size_t outputCount = multipleOutputs ? task.outputSizes.size() : (task.legacyOutputSize ? 1 : 0);

        if (outputCount == 0) {
            result.success = false;
            result.errorMessage = "No outputs requested";
            return result;
        }

        // Create and fill input buffers
        std::vector<Buffer> inputBuffers;
        inputBuffers.reserve(std::max<size_t>(1, inputCount));
        if (multipleInputs) {
            for (const auto& in : task.inputData) {
                auto buf = makeHostBuffer(in.empty() ? 4 : (VkDeviceSize)in.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                if (!in.empty()) std::memcpy(buf.mapped, in.data(), in.size());
                inputBuffers.push_back(buf);
            }
        } else if (!task.legacyInputData.empty()) {
            auto& in = task.legacyInputData;
            auto buf = makeHostBuffer((VkDeviceSize)in.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            std::memcpy(buf.mapped, in.data(), in.size());
            inputBuffers.push_back(buf);
        }

        // Create output buffers
        std::vector<Buffer> outputBuffers;
        outputBuffers.reserve(std::max<size_t>(1, outputCount));
        if (multipleOutputs) {
            for (size_t sz : task.outputSizes) {
                auto buf = makeHostBuffer((VkDeviceSize)std::max<size_t>(sz,4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                outputBuffers.push_back(buf);
            }
        } else {
            auto buf = makeHostBuffer((VkDeviceSize)std::max<size_t>(task.legacyOutputSize,4), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            outputBuffers.push_back(buf);
        }

        const uint32_t totalBindings = (uint32_t)(inputBuffers.size() + outputBuffers.size());

        // Build descriptor set layout: all storage buffers, inputs first then outputs
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        layoutBindings.reserve(totalBindings);
        for (uint32_t i=0; i<totalBindings; ++i) {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = i;
            lb.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            lb.descriptorCount = 1;
            lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            layoutBindings.push_back(lb);
        }

        VkDescriptorSetLayoutCreateInfo dslci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dslci.bindingCount = (uint32_t)layoutBindings.size();
        dslci.pBindings = layoutBindings.data();
        if (vkCreateDescriptorSetLayout(device_, &dslci, nullptr, &dsetLayout) != VK_SUCCESS)
            throw std::runtime_error("vkCreateDescriptorSetLayout failed");

        VkPipelineLayoutCreateInfo plci{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        plci.setLayoutCount = 1;
        plci.pSetLayouts = &dsetLayout;
        if (vkCreatePipelineLayout(device_, &plci, nullptr, &pipelineLayout) != VK_SUCCESS)
            throw std::runtime_error("vkCreatePipelineLayout failed");

        // Compile GLSL to SPIR-V if kernel provided; otherwise fall back to CPU copy.
        bool useGpu = !task.kernel.empty();
        if (useGpu) {
            auto macros = parse_macros(task.compilationOptions);
            std::vector<uint32_t> spirv = compile_glsl_to_spirv(task.kernel, task.entry.empty() ? "main" : task.entry, macros);
            VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            smci.codeSize = spirv.size() * sizeof(uint32_t);
            smci.pCode = spirv.data();
            if (vkCreateShaderModule(device_, &smci, nullptr, &shaderModule) != VK_SUCCESS)
                throw std::runtime_error("vkCreateShaderModule failed");

            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = shaderModule;
            stage.pName = (task.entry.empty() ? "main" : task.entry.c_str());

            VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpci.stage = stage;
            cpci.layout = pipelineLayout;
            if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) != VK_SUCCESS)
                throw std::runtime_error("vkCreateComputePipelines failed");
        }

        // Descriptor pool & set
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = totalBindings;

        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = 1;
        dpci.pPoolSizes = &poolSize;
        if (vkCreateDescriptorPool(device_, &dpci, nullptr, &descPool) != VK_SUCCESS)
            throw std::runtime_error("vkCreateDescriptorPool failed");

        VkDescriptorSet descSet = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = descPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsetLayout;
        if (vkAllocateDescriptorSets(device_, &dsai, &descSet) != VK_SUCCESS)
            throw std::runtime_error("vkAllocateDescriptorSets failed");

        // Update descriptors
        std::vector<VkDescriptorBufferInfo> infos(totalBindings);
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(totalBindings);
        uint32_t bind = 0;
        for (const auto& b : inputBuffers) {
            infos[bind] = { b.buf, 0, b.size };
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = descSet;
            w.dstBinding = bind;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &infos[bind];
            writes.push_back(w);
            ++bind;
        }
        for (const auto& b : outputBuffers) {
            infos[bind] = { b.buf, 0, b.size };
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = descSet;
            w.dstBinding = bind;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &infos[bind];
            writes.push_back(w);
            ++bind;
        }
        vkUpdateDescriptorSets(device_, (uint32_t)writes.size(), writes.data(), 0, nullptr);

        if (useGpu) {
            // Record and submit compute dispatch
            VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cbai.commandPool = cmdPool_;
            cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 1;
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            if (vkAllocateCommandBuffers(device_, &cbai, &cmd) != VK_SUCCESS)
                throw std::runtime_error("vkAllocateCommandBuffers failed");

            VkCommandBufferBeginInfo cbbi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            vkBeginCommandBuffer(cmd, &cbbi);
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);

            uint32_t gx = 1, gy = 1, gz = 1;
            if (!task.workgroupCount.empty()) {
                if (task.workgroupCount.size() >= 1) gx = (uint32_t)std::max(1, task.workgroupCount[0]);
                if (task.workgroupCount.size() >= 2) gy = (uint32_t)std::max(1, task.workgroupCount[1]);
                if (task.workgroupCount.size() >= 3) gz = (uint32_t)std::max(1, task.workgroupCount[2]);
            }
            vkCmdDispatch(cmd, gx, gy, gz);
            vkEndCommandBuffer(cmd);

            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue_);

            vkFreeCommandBuffers(device_, cmdPool_, 1, &cmd);
        } else {
            // CPU fallback: copy inputs to outputs
            const size_t n = std::min(inputBuffers.size(), outputBuffers.size());
            for (size_t i=0;i<n;++i) {
                size_t copyBytes = (size_t)std::min<VkDeviceSize>(inputBuffers[i].size, outputBuffers[i].size);
                std::memcpy(outputBuffers[i].mapped, inputBuffers[i].mapped, copyBytes);
                if (outputBuffers[i].size > copyBytes) {
                    std::memset(static_cast<char*>(outputBuffers[i].mapped)+copyBytes, 0, (size_t)(outputBuffers[i].size - copyBytes));
                }
            }
            for (size_t i=n;i<outputBuffers.size();++i) {
                std::memset(outputBuffers[i].mapped, 0, (size_t)outputBuffers[i].size);
            }
        }

        // Readback
        if (multipleOutputs) {
            result.outputData.resize(outputBuffers.size());
            for (size_t i=0;i<outputBuffers.size();++i) {
                result.outputData[i].resize((size_t)outputBuffers[i].size);
                std::memcpy(result.outputData[i].data(), outputBuffers[i].mapped, (size_t)outputBuffers[i].size);
            }
            if (!result.outputData.empty()) result.legacyOutputData = result.outputData[0];
        } else {
            result.legacyOutputData.resize((size_t)outputBuffers[0].size);
            std::memcpy(result.legacyOutputData.data(), outputBuffers[0].mapped, (size_t)outputBuffers[0].size);
            result.outputData = { result.legacyOutputData };
        }

        // Cleanup resources
        for (auto& b : inputBuffers) destroyBuffer(b);
        for (auto& b : outputBuffers) destroyBuffer(b);
        if (pipeline) vkDestroyPipeline(device_, pipeline, nullptr);
        if (shaderModule) vkDestroyShaderModule(device_, shaderModule, nullptr);
        if (descPool) vkDestroyDescriptorPool(device_, descPool, nullptr);
        if (pipelineLayout) vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
        if (dsetLayout) vkDestroyDescriptorSetLayout(device_, dsetLayout, nullptr);

        auto t1 = std::chrono::high_resolution_clock::now();
        result.processingTime = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.success = true;
        return result;
    } catch (const std::exception& e) {
        if (pipeline) vkDestroyPipeline(device_, pipeline, nullptr);
        if (shaderModule) vkDestroyShaderModule(device_, shaderModule, nullptr);
        if (descPool) vkDestroyDescriptorPool(device_, descPool, nullptr);
        if (pipelineLayout) vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
        if (dsetLayout) vkDestroyDescriptorSetLayout(device_, dsetLayout, nullptr);
        result.success = false;
        result.errorMessage = e.what();
        return result;
    }
}

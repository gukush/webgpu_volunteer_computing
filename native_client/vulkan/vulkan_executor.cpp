#include "vulkan_executor.hpp"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <sstream>

#ifndef HAVE_SHADERC
  #error "This build requires shaderc: define HAVE_SHADERC and link with shaderc"
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
    return compile_glsl_shaderc(src, entry, macros);
}

} // namespace

VulkanExecutor::VulkanExecutor(int preferredDeviceIndex)
    : preferredDeviceIndex_(preferredDeviceIndex) {}

VulkanExecutor::~VulkanExecutor() {
    cleanupShaderCache();
    cleanup();
}

bool VulkanExecutor::initialize(const json&) {
    if (!createInstance()) return false;
    if (!pickPhysicalDevice()) return false;
    if (!createDevice()) return false;
    return true;
}

void VulkanExecutor::cleanup() {
    cleanupShaderCache();
    destroyVulkan();
}

void VulkanExecutor::cleanupShaderCache() {
    if (!device_) return;

    for (auto& [key, shader] : shaderCache_) {
        if (shader.shaderModule != VK_NULL_HANDLE) {
            vkDestroyShaderModule(device_, shader.shaderModule, nullptr);
        }
    }
    shaderCache_.clear();
    std::cout << "Cleaned up Vulkan shader cache" << std::endl;
}

std::string VulkanExecutor::makeCacheKey(const std::string& source, const std::string& entry, const json& compileOpts) const {
    std::hash<std::string> hasher;
    size_t hash = hasher(source);
    hash ^= (hasher(entry) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2));

    // Include compilation options in hash
    std::string optsStr = compileOpts.dump();
    hash ^= (hasher(optsStr) + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2));

    std::ostringstream ss;
    ss << std::hex << hash;
    return ss.str();
}

bool VulkanExecutor::getOrCompileShader(const std::string& source, const std::string& entry,
                                      const json& compileOpts, CompiledShader*& outShader) {
    std::string cacheKey = makeCacheKey(source, entry, compileOpts);

    // Check if shader is already cached
    auto it = shaderCache_.find(cacheKey);
    if (it != shaderCache_.end()) {
        outShader = &it->second;
        std::cout << "Using cached GLSL shader (key: " << cacheKey << ")" << std::endl;
        return true;
    }

    try {
        std::cout << "Compiling GLSL shader to SPIR-V (key: " << cacheKey << ")..." << std::endl;

        // Compile GLSL to SPIR-V
        auto macros = parse_macros(compileOpts);
        std::vector<uint32_t> spirv = compile_glsl_to_spirv(source, entry, macros);

        // Create Vulkan shader module
        VkShaderModuleCreateInfo smci{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        smci.codeSize = spirv.size() * sizeof(uint32_t);
        smci.pCode = spirv.data();

        VkShaderModule shaderModule = VK_NULL_HANDLE;
        if (vkCreateShaderModule(device_, &smci, nullptr, &shaderModule) != VK_SUCCESS) {
            throw std::runtime_error("vkCreateShaderModule failed");
        }

        // Store in cache
        CompiledShader compiledShader;
        compiledShader.spirv = std::move(spirv);
        compiledShader.shaderModule = shaderModule;
        compiledShader.entryPoint = entry;

        auto [insertIt, success] = shaderCache_.emplace(cacheKey, std::move(compiledShader));
        if (success) {
            outShader = &insertIt->second;
            std::cout << "GLSL shader compiled and cached successfully (SPIR-V size: "
                      << outShader->spirv.size() * 4 << " bytes)" << std::endl;
            return true;
        } else {
            // Failed to cache, clean up
            vkDestroyShaderModule(device_, shaderModule, nullptr);
            throw std::runtime_error("Failed to cache compiled shader");
        }

    } catch (const std::exception& e) {
        std::cerr << "Shader compilation failed: " << e.what() << std::endl;
        return false;
    }
}

json VulkanExecutor::getCapabilities() const {
    json caps;
    caps["framework"] = "vulkan";
    caps["initialized"] = (device_ != VK_NULL_HANDLE);
    caps["supportsMultiInput"] = true;
    caps["supportsMultiOutput"] = true;
    caps["shaderSource"] = "glsl";
    caps["requiresBindings"] = "uniforms at binding 0, inputs first, then outputs (set=0, binding=i)";
    caps["shaderCache"] = true;
    caps["shaderCacheSize"] = static_cast<int>(shaderCache_.size());

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

// FIXED: Enhanced executeTask with metadata handling and shader caching
TaskResult VulkanExecutor::executeTask(const TaskData& task) {
    TaskResult result{};
    auto t0 = std::chrono::high_resolution_clock::now();

    if (device_ == VK_NULL_HANDLE) {
        result.success = false;
        result.errorMessage = "Vulkan not initialized";
        return result;
    }

    VkDescriptorSetLayout dsetLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool descPool = VK_NULL_HANDLE;
    VkPipeline pipeline = VK_NULL_HANDLE;

    try {
        // Determine IO counts
        const bool multipleInputs = !task.inputData.empty();
        const bool multipleOutputs = !task.outputSizes.empty();
        const bool hasUniforms = !task.metadata.empty();

        const size_t inputCount = multipleInputs ? task.inputData.size() :
                                 (task.legacyInputData.empty() ? 0 : 1);
        const size_t outputCount = multipleOutputs ? task.outputSizes.size() :
                                  (task.legacyOutputSize ? 1 : 0);

        std::cout << "Task " << task.id << " - Framework: " << task.framework
                  << ", Inputs: " << inputCount << ", Outputs: " << outputCount
                  << ", Has uniforms: " << hasUniforms << std::endl;

        if (outputCount == 0) {
            result.success = false;
            result.errorMessage = "No outputs requested";
            return result;
        }

        // FIXED: Create uniform buffer from metadata
        Buffer uniformBuffer{};
        if (hasUniforms) {
            std::cout << "Creating uniform buffer from metadata..." << std::endl;

            // Extract uniform values in consistent order (matching browser client)
            std::vector<uint32_t> uniformData;
            std::vector<std::string> fieldOrder = {
                "block_size", "matrix_size", "matrix_n", "matrixSize",
                "tile_start_row", "tile_start_col", "tile_rows", "tile_cols",
                "tile_size", "tileSize"
            };

            for (const auto& field : fieldOrder) {
                if (task.metadata.contains(field)) {
                    uint32_t value = task.metadata[field].get<uint32_t>();
                    uniformData.push_back(value);
                    std::cout << "  " << field << " = " << value << std::endl;
                }
            }

            // Add any remaining numeric values not in fieldOrder
            for (auto it = task.metadata.begin(); it != task.metadata.end(); ++it) {
                if (it.value().is_number() &&
                    std::find(fieldOrder.begin(), fieldOrder.end(), it.key()) == fieldOrder.end()) {
                    uniformData.push_back(it.value().get<uint32_t>());
                    std::cout << "  " << it.key() << " = " << it.value().get<uint32_t>() << " (extra)" << std::endl;
                }
            }

            if (!uniformData.empty()) {
                size_t uniformSize = std::max<size_t>(16, uniformData.size() * sizeof(uint32_t));
                uniformBuffer = makeHostBuffer(uniformSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
                std::memcpy(uniformBuffer.mapped, uniformData.data(),
                           uniformData.size() * sizeof(uint32_t));
                std::cout << "Created uniform buffer with " << uniformData.size()
                         << " values (" << uniformSize << " bytes)" << std::endl;
            }
        }

        // Create and fill input buffers
        std::vector<Buffer> inputBuffers;
        inputBuffers.reserve(std::max<size_t>(1, inputCount));
        if (multipleInputs) {
            for (size_t i = 0; i < task.inputData.size(); ++i) {
                const auto& in = task.inputData[i];
                auto buf = makeHostBuffer(in.empty() ? 4 : (VkDeviceSize)in.size(),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                if (!in.empty()) {
                    std::memcpy(buf.mapped, in.data(), in.size());
                    std::cout << "Input buffer " << i << ": " << in.size() << " bytes" << std::endl;
                }
                inputBuffers.push_back(buf);
            }
        } else if (!task.legacyInputData.empty()) {
            auto& in = task.legacyInputData;
            auto buf = makeHostBuffer((VkDeviceSize)in.size(), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            std::memcpy(buf.mapped, in.data(), in.size());
            std::cout << "Legacy input buffer: " << in.size() << " bytes" << std::endl;
            inputBuffers.push_back(buf);
        }

        // Create output buffers
        std::vector<Buffer> outputBuffers;
        outputBuffers.reserve(std::max<size_t>(1, outputCount));
        if (multipleOutputs) {
            for (size_t i = 0; i < task.outputSizes.size(); ++i) {
                size_t sz = task.outputSizes[i];
                auto buf = makeHostBuffer((VkDeviceSize)std::max<size_t>(sz,4),
                                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
                outputBuffers.push_back(buf);
                std::cout << "Output buffer " << i << ": " << sz << " bytes" << std::endl;
            }
        } else {
            auto buf = makeHostBuffer((VkDeviceSize)std::max<size_t>(task.legacyOutputSize,4),
                                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
            outputBuffers.push_back(buf);
            std::cout << "Legacy output buffer: " << task.legacyOutputSize << " bytes" << std::endl;
        }

        // FIXED: Calculate total bindings: uniforms(0 or 1) + inputs + outputs
        const uint32_t uniformBindings = hasUniforms ? 1 : 0;
        const uint32_t totalBindings = uniformBindings + (uint32_t)inputBuffers.size() +
                                      (uint32_t)outputBuffers.size();

        std::cout << "Descriptor layout: " << uniformBindings << " uniforms + "
                  << inputBuffers.size() << " inputs + " << outputBuffers.size()
                  << " outputs = " << totalBindings << " total bindings" << std::endl;

        // Build descriptor set layout
        std::vector<VkDescriptorSetLayoutBinding> layoutBindings;
        layoutBindings.reserve(totalBindings);

        uint32_t binding = 0;

        // FIXED: Add uniform buffer binding at 0 if present
        if (hasUniforms) {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = binding++;
            lb.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            lb.descriptorCount = 1;
            lb.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            layoutBindings.push_back(lb);
            std::cout << "Added uniform buffer at binding " << (binding-1) << std::endl;
        }

        // Add storage buffer bindings for inputs and outputs
        for (uint32_t i = 0; i < (uint32_t)(inputBuffers.size() + outputBuffers.size()); ++i) {
            VkDescriptorSetLayoutBinding lb{};
            lb.binding = binding++;
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

        // NEW: Use cached shader compilation if kernel provided; otherwise fall back to CPU copy.
        bool useGpu = !task.kernel.empty();
        CompiledShader* cachedShader = nullptr;

        if (useGpu) {
            const std::string& entry = task.entry.empty() ? "main" : task.entry;
            if (!getOrCompileShader(task.kernel, entry, task.compilationOptions, cachedShader)) {
                throw std::runtime_error("Shader compilation failed");
            }

            VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
            stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
            stage.module = cachedShader->shaderModule;
            stage.pName = cachedShader->entryPoint.c_str();

            VkComputePipelineCreateInfo cpci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            cpci.stage = stage;
            cpci.layout = pipelineLayout;
            if (vkCreateComputePipelines(device_, VK_NULL_HANDLE, 1, &cpci, nullptr, &pipeline) != VK_SUCCESS)
                throw std::runtime_error("vkCreateComputePipelines failed");
        }

        // Create descriptor pool
        std::vector<VkDescriptorPoolSize> poolSizes;
        if (hasUniforms) {
            poolSizes.push_back({VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1});
        }
        if (inputBuffers.size() + outputBuffers.size() > 0) {
            poolSizes.push_back({VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                               (uint32_t)(inputBuffers.size() + outputBuffers.size())});
        }

        VkDescriptorPoolCreateInfo dpci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpci.maxSets = 1;
        dpci.poolSizeCount = (uint32_t)poolSizes.size();
        dpci.pPoolSizes = poolSizes.data();
        if (vkCreateDescriptorPool(device_, &dpci, nullptr, &descPool) != VK_SUCCESS)
            throw std::runtime_error("vkCreateDescriptorPool failed");

        VkDescriptorSet descSet = VK_NULL_HANDLE;
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = descPool;
        dsai.descriptorSetCount = 1;
        dsai.pSetLayouts = &dsetLayout;
        if (vkAllocateDescriptorSets(device_, &dsai, &descSet) != VK_SUCCESS)
            throw std::runtime_error("vkAllocateDescriptorSets failed");

        // FIXED: Update descriptors with proper binding order
        std::vector<VkDescriptorBufferInfo> infos(totalBindings);
        std::vector<VkWriteDescriptorSet> writes;
        writes.reserve(totalBindings);

        binding = 0;

        // Bind uniform buffer first if present
        if (hasUniforms) {
            infos[binding] = { uniformBuffer.buf, 0, uniformBuffer.size };
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = descSet;
            w.dstBinding = binding;
            w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &infos[binding];
            writes.push_back(w);
            std::cout << "Bound uniform buffer to binding " << binding << std::endl;
            ++binding;
        }

        // Bind input buffers
        for (const auto& b : inputBuffers) {
            infos[binding] = { b.buf, 0, b.size };
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = descSet;
            w.dstBinding = binding;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &infos[binding];
            writes.push_back(w);
            std::cout << "Bound input buffer to binding " << binding << std::endl;
            ++binding;
        }

        // Bind output buffers
        for (const auto& b : outputBuffers) {
            infos[binding] = { b.buf, 0, b.size };
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.dstSet = descSet;
            w.dstBinding = binding;
            w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            w.descriptorCount = 1;
            w.pBufferInfo = &infos[binding];
            writes.push_back(w);
            std::cout << "Bound output buffer to binding " << binding << std::endl;
            ++binding;
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
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout,
                                   0, 1, &descSet, 0, nullptr);

            uint32_t gx = 1, gy = 1, gz = 1;
            if (!task.workgroupCount.empty()) {
                if (task.workgroupCount.size() >= 1) gx = (uint32_t)std::max(1, task.workgroupCount[0]);
                if (task.workgroupCount.size() >= 2) gy = (uint32_t)std::max(1, task.workgroupCount[1]);
                if (task.workgroupCount.size() >= 3) gz = (uint32_t)std::max(1, task.workgroupCount[2]);
            }

            std::cout << "Dispatching compute: " << gx << "x" << gy << "x" << gz << " workgroups" << std::endl;
            vkCmdDispatch(cmd, gx, gy, gz);
            vkEndCommandBuffer(cmd);

            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd;
            vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue_);

            vkFreeCommandBuffers(device_, cmdPool_, 1, &cmd);
        } else {
            std::cout << "Using CPU fallback (no GPU kernel provided)" << std::endl;
            // CPU fallback: copy inputs to outputs
            const size_t n = std::min(inputBuffers.size(), outputBuffers.size());
            for (size_t i = 0; i < n; ++i) {
                size_t copyBytes = (size_t)std::min<VkDeviceSize>(inputBuffers[i].size, outputBuffers[i].size);
                std::memcpy(outputBuffers[i].mapped, inputBuffers[i].mapped, copyBytes);
                if (outputBuffers[i].size > copyBytes) {
                    std::memset(static_cast<char*>(outputBuffers[i].mapped) + copyBytes, 0,
                               (size_t)(outputBuffers[i].size - copyBytes));
                }
            }
            for (size_t i = n; i < outputBuffers.size(); ++i) {
                std::memset(outputBuffers[i].mapped, 0, (size_t)outputBuffers[i].size);
            }
        }

        // Readback results
        if (multipleOutputs) {
            result.outputData.resize(outputBuffers.size());
            for (size_t i = 0; i < outputBuffers.size(); ++i) {
                result.outputData[i].resize((size_t)outputBuffers[i].size);
                std::memcpy(result.outputData[i].data(), outputBuffers[i].mapped, (size_t)outputBuffers[i].size);
                std::cout << "Retrieved output " << i << ": " << result.outputData[i].size() << " bytes" << std::endl;
            }
            if (!result.outputData.empty()) result.legacyOutputData = result.outputData[0];
        } else {
            result.legacyOutputData.resize((size_t)outputBuffers[0].size);
            std::memcpy(result.legacyOutputData.data(), outputBuffers[0].mapped, (size_t)outputBuffers[0].size);
            result.outputData = { result.legacyOutputData };
            std::cout << "Retrieved legacy output: " << result.legacyOutputData.size() << " bytes" << std::endl;
        }

        // Cleanup resources
        for (auto& b : inputBuffers) destroyBuffer(b);
        for (auto& b : outputBuffers) destroyBuffer(b);
        if (hasUniforms) destroyBuffer(uniformBuffer);

        // NOTE: Don't destroy cached shader module - it's managed by the cache
        if (pipeline) vkDestroyPipeline(device_, pipeline, nullptr);
        if (descPool) vkDestroyDescriptorPool(device_, descPool, nullptr);
        if (pipelineLayout) vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
        if (dsetLayout) vkDestroyDescriptorSetLayout(device_, dsetLayout, nullptr);

        auto t1 = std::chrono::high_resolution_clock::now();
        result.processingTime = std::chrono::duration<double, std::milli>(t1 - t0).count();
        result.success = true;

        std::cout << "Task " << task.id << " completed successfully in "
                  << result.processingTime << "ms (shader cache size: " << shaderCache_.size() << ")" << std::endl;
        return result;

    } catch (const std::exception& e) {
        std::cerr << "Task " << task.id << " failed: " << e.what() << std::endl;

        // Cleanup on error (but don't destroy cached shaders)
        if (pipeline) vkDestroyPipeline(device_, pipeline, nullptr);
        if (descPool) vkDestroyDescriptorPool(device_, descPool, nullptr);
        if (pipelineLayout) vkDestroyPipelineLayout(device_, pipelineLayout, nullptr);
        if (dsetLayout) vkDestroyDescriptorSetLayout(device_, dsetLayout, nullptr);

        result.success = false;
        result.errorMessage = e.what();
        return result;
    }
}
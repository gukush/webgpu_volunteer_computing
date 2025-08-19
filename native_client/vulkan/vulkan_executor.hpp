#pragma once
#include "../common/framework_client.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

class VulkanExecutor : public IFrameworkExecutor {
private:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex = 0;
    bool initialized = false;

    // Device properties for capabilities
    VkPhysicalDeviceProperties deviceProperties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};

    struct CompiledShader {
        VkShaderModule shaderModule = VK_NULL_HANDLE;
        VkPipeline computePipeline = VK_NULL_HANDLE;
        VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
        VkDescriptorSetLayout descriptorSetLayout = VK_NULL_HANDLE;
    };

    std::map<std::string, CompiledShader> shaderCache;

    bool createInstance();
    bool selectPhysicalDevice();
    bool createLogicalDevice();
    bool compileShader(const std::string& source, const std::string& entryPoint,
                      const json& compileOpts, CompiledShader& result);

    // Helper methods for buffer management
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties);
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                     VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory);

public:
    VulkanExecutor();
    ~VulkanExecutor() override;

    bool initialize(const json& config = {}) override;
    void cleanup() override;
    TaskResult executeTask(const TaskData& task) override;
    std::string getFrameworkName() const override { return "vulkan"; }
    json getCapabilities() const override;
};
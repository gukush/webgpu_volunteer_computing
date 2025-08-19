#pragma once
#include "../common/framework_client.hpp"
#include <vulkan/vulkan.h>
#include <vector>
#include <memory>

// NOTE: This header replaces the previous include path "../common/framework_client.hpp".
// If your tree uses a different layout, adjust the include accordingly.

class VulkanExecutor : public IFrameworkExecutor {
private:
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue computeQueue = VK_NULL_HANDLE;
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkPipelineCache pipelineCache = VK_NULL_HANDLE;
    uint32_t computeQueueFamilyIndex = 0;
    bool initialized = false;

    // Cached device properties/memory properties
    VkPhysicalDeviceProperties deviceProperties{};
    VkPhysicalDeviceMemoryProperties memoryProperties{};

    // Helpers
    bool createInstance(bool enableValidation, bool enableDebugUtils);
    bool pickPhysicalDevice();
    bool createLogicalDevice();
    bool createCommandPool();
    bool createDescriptorPool();
    void destroyInstance();
    void destroyDeviceObjects();

    // Shader compilation
    bool compileGLSLtoSPIRV(const std::string& glsl, std::vector<uint32_t>& spirv, std::string& error) const;

    // Buffer helpers
    uint32_t findMemoryType(uint32_t typeFilter, VkMemoryPropertyFlags properties) const;
    bool createBuffer(VkDeviceSize size, VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) const;

public:
    VulkanExecutor();
    ~VulkanExecutor() override;

    bool initialize(const json& config = {}) override;
    void cleanup() override;
    TaskResult executeTask(const TaskData& task) override;
    std::string getFrameworkName() const override { return "vulkan"; }
    json getCapabilities() const override;
};

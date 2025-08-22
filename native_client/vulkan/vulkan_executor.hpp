#pragma once
#include "../common/framework_client.hpp"
#ifdef HAVE_VULKAN
#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <optional>
#include <memory>


// Vulkan executor implementing Multi-Input / Multi-Output (MIMO) interface used by the framework.
class VulkanExecutor final : public IFrameworkExecutor {
public:
    explicit VulkanExecutor(int preferredDeviceIndex = 0);
    ~VulkanExecutor() override;

    bool initialize(const json& config = {}) override;
    void cleanup() override;
    TaskResult executeTask(const TaskData& task) override;
    std::string getFrameworkName() const override { return "vulkan"; }
    json getCapabilities() const override;

private:
    int preferredDeviceIndex_{0};

    VkInstance instance_{VK_NULL_HANDLE};
    VkPhysicalDevice phys_{VK_NULL_HANDLE};
    uint32_t computeQueueFamily_{UINT32_MAX};
    VkDevice device_{VK_NULL_HANDLE};
    VkQueue queue_{VK_NULL_HANDLE};
    VkCommandPool cmdPool_{VK_NULL_HANDLE};

    // Helpers
    bool createInstance();
    bool pickPhysicalDevice();
    bool createDevice();
    void destroyVulkan();

    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
    bool createBuffer(VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      VkMemoryPropertyFlags props,
                      VkBuffer& outBuf, VkDeviceMemory& outMem) const;

    struct Buffer {
        VkBuffer buf{VK_NULL_HANDLE};
        VkDeviceMemory mem{VK_NULL_HANDLE};
        VkDeviceSize size{0};
        void* mapped{nullptr};
    };
    Buffer makeHostBuffer(VkDeviceSize size, VkBufferUsageFlags usage) const;
    void destroyBuffer(Buffer& b) const;
};
#endif
#pragma once

#include "common/framework_client.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>

// Forward declarations for kernel types
enum class CPUKernelType {
    MATRIX_MULTIPLY,
    CONVOLUTION,
    ECM_STAGE1,
    SORTING,
    UNKNOWN
};

class CPUExecutor : public IFrameworkExecutor {
private:
    bool initialized = false;
    json deviceInfo;

    // Kernel type detection
    CPUKernelType detectKernelType(const TaskData& task);

    // Kernel execution methods
    TaskResult executeMatrixMultiplyKernel(const TaskData& task);
    TaskResult executeConvolutionKernel(const TaskData& task);
    TaskResult executeECMKernel(const TaskData& task);
    TaskResult executeSortingKernel(const TaskData& task);

    // Data parsing helpers
    template<typename T>
    std::vector<T> parseTypedData(const std::vector<uint8_t>& rawData);

    template<typename T>
    std::vector<uint8_t> serializeTypedData(const std::vector<T>& data);

public:
    CPUExecutor() = default;
    ~CPUExecutor() override = default;

    // IFrameworkExecutor interface
    bool initialize(const json& config = {}) override;
    void cleanup() override;
    TaskResult executeTask(const TaskData& task) override;
    std::string getFrameworkName() const override { return "cpp"; }
    json getCapabilities() const override;
};
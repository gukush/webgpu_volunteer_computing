#pragma once
#include "../common/framework_client.hpp"
#include <cuda_runtime.h>
#include <cuda.h>
#include <nvrtc.h>
#include <vector>
#include <memory>

class CudaExecutor : public IFrameworkExecutor {
private:
    int deviceId = 0;
    cudaDeviceProp deviceProps;
    CUcontext context = nullptr;
    bool initialized = false;

    struct CompiledKernel {
        CUmodule module = nullptr;
        CUfunction function = nullptr;
        std::string ptx;
    };

    std::map<std::string, CompiledKernel> kernelCache;

    // NEW: Enhanced uniform support for task-agnostic operation
    enum class UniformType {
        INT32,
        UINT32,
        FLOAT
    };

    struct UniformValue {
        std::string name;
        UniformType type;
        union {
            int32_t intValue;
            uint32_t uintValue;
            float floatValue;
        };
    };

    bool compileKernel(const std::string& source, const std::string& entryPoint,
                      const json& compileOpts, CompiledKernel& result);

    // NEW: Task-agnostic metadata processing
    bool processMetadataUniforms(const TaskData& task, std::vector<UniformValue>& uniforms);
    void addUniformsToKernelArgs(const std::vector<UniformValue>& uniforms,
                                std::vector<void*>& kernelArgs,
                                std::vector<int32_t>& intStorage,
                                std::vector<uint32_t>& uintStorage,
                                std::vector<float>& floatStorage);

public:
    CudaExecutor(int deviceId = 0);
    ~CudaExecutor() override;

    bool initialize(const json& config = {}) override;
    void cleanup() override;
    TaskResult executeTask(const TaskData& task) override;
    std::string getFrameworkName() const override { return "cuda"; }
    json getCapabilities() const override;
};
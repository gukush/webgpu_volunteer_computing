#pragma once
#include "../common/framework_client.hpp"
#include <cuda_runtime.h>
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
    
    bool compileKernel(const std::string& source, const std::string& entryPoint, 
                      const json& compileOpts, CompiledKernel& result);
    
public:
    CudaExecutor(int deviceId = 0);
    ~CudaExecutor() override;
    
    bool initialize(const json& config = {}) override;
    void cleanup() override;
    TaskResult executeTask(const TaskData& task) override;
    std::string getFrameworkName() const override { return "cuda"; }
    json getCapabilities() const override;
};

// cuda_executor.hpp
#pragma once
#include "../common/framework_client.hpp"
#ifdef HAVE_CUDA
#include <cuda.h>
#include <nvrtc.h>
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <cstdint>

class CudaExecutor : public IFrameworkExecutor {
public:
    CudaExecutor(int deviceId = 0);
    ~CudaExecutor() override;

    bool initialize(const json& config = {}) override;
    void cleanup() override;

    TaskResult executeTask(const TaskData& task) override;

    std::string getFrameworkName() const override { return "cuda"; }
    json getCapabilities() const override;

    // Scalar kinds we support for uniforms (kernel params)
    enum class UniformType {
        INT32,
        UINT32,
        FLOAT32,
        INT64,
        UINT64,
        FLOAT64
    };

    struct UniformValue {
        std::string name;
        UniformType type;
        // store as separate fields
        int32_t   i32  = 0;
        uint32_t  u32  = 0;
        float     f32  = 0.0f;
        int64_t   i64  = 0;
        uint64_t  u64  = 0;
        double    f64  = 0.0;
    };

private:
    int deviceId = 0;
    CUcontext context = nullptr;
    bool initialized = false;

    // Basic device info (Driver API)
    std::string deviceName;
    int computeMajor = 0;
    int computeMinor = 0;
    size_t totalGlobalMem = 0;
    int maxThreadsPerBlock = 0;
    int multiProcessorCount = 0;

    struct CompiledKernel {
        CUmodule module = nullptr;
        CUfunction function = nullptr;
        std::string ptx;
    };

    std::map<std::string, CompiledKernel> kernelCache;

    // ---- Internal helpers ----
    static void logCuError(const char* where, CUresult r);
    static bool cuCheck(CUresult r, const char* where);
    static bool nvrtcCheck(nvrtcResult r, const char* where, const std::string& log = {});

    bool ensureDriverLoaded();
    bool ensureContextCurrent();

    static std::string makeCacheKey(const std::string& source, const std::string& entry);

    bool compileKernel(const std::string& source,
                       const std::string& entryPoint,
                       const json& compileOpts,
                       CompiledKernel& result);

    // Kernel-agnostic uniform collection from metadata/schema
    bool buildUniformList(const TaskData& task, std::vector<UniformValue>& uniforms);

    // Reallocation-safe: reserves storage and then takes addresses
    void addUniformsToKernelArgs(const std::vector<UniformValue>& uniforms,
                                 std::vector<void*>& kernelArgs,
                                 std::vector<int32_t>&  i32Store,
                                 std::vector<uint32_t>& u32Store,
                                 std::vector<float>&    f32Store,
                                 std::vector<int64_t>&  i64Store,
                                 std::vector<uint64_t>& u64Store,
                                 std::vector<double>&   f64Store);
};
#endif
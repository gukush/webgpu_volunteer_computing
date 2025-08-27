// cling_executor.hpp
#pragma once

#include "../common/framework_client.hpp"

#ifdef HAVE_CLING
  // Cling public headers (ship with libcling/clang)
  #include <cling/Interpreter/Interpreter.h>
  #include <cling/Interpreter/Value.h>
#endif

#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <chrono>

// A minimal single-threaded "GPU-style" executor that JIT-compiles C++ kernels
// with Cling and executes them by looping over elements. The contract mirrors
// the existing executors (OpenCL/CUDA/Vulkan) so the FrameworkClient can call it
// the same way.
//
// Expected task fields (aligned with other executors):
//   - task.kernel  : std::string with C++17 source code for the kernel
//   - task.entry   : std::string with the entry function name in task.kernel
//   - task.inputData : std::vector<std::vector<uint8_t>> input buffers
//   - task.outputSizes : std::vector<size_t> sizes (in bytes) for each output buffer
//   - task.metadata / task.chunkUniforms : JSON with numeric uniforms (i32/u32/f32).
//
// Required kernel signature inside task.kernel (C++17):
//   extern "C" void <entry>(size_t idx, void** inputs, void** outputs, const __uniforms_t& u);
//
// Dispatch size (loop bound):
//   - If metadata contains N (or elementCount), that is used.
//   - Else the first outputSizes[0] (in bytes) is used.
//   - Provide N explicitly for clarity.
//
// Notes:
//   - This backend is single-threaded on CPU, intended to emulate GPU kernels deterministically.
//   - Build only when HAVE_CLING is defined and libcling is available.
class ClingExecutor : public IFrameworkExecutor {
public:
    ClingExecutor() = default;
    ~ClingExecutor() override { cleanup(); }

    bool initialize(const json& config = {}) override;
    void cleanup() override;
    TaskResult executeTask(const TaskData& task) override;
    std::string getFrameworkName() const override { return "cling"; }
    json getCapabilities() const override;

private:
    struct UniformValue {
        enum class Type { INT32, UINT32, FLOAT32 };
        std::string name;
        Type type{};
        int32_t  i32{};
        uint32_t u32{};
        float    f32{};
    };

    bool buildUniformList(const TaskData& task, std::vector<UniformValue>& uniforms) const;
    std::string generatePrelude(const std::string& entry,
                                const std::vector<UniformValue>& uniforms) const;
    static std::string sanitizeIdent(const std::string& s);
    size_t resolveDispatchCount(const TaskData& task) const;

#ifdef HAVE_CLING
    std::unique_ptr<cling::Interpreter> interp_;
    std::vector<std::string> extraArgs_;
#endif

    bool initialized_{false};
};

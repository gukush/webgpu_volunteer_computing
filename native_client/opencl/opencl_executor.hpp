#pragma once
#include "../common/framework_client.hpp"
#ifdef HAVE_OPENCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <vector>
#include <memory>

class OpenCLExecutor : public IFrameworkExecutor {
private:
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    bool initialized = false;

    struct CompiledKernel {
        cl_program program = nullptr;
        cl_kernel kernel = nullptr;
    };

    std::map<std::string, CompiledKernel> kernelCache;

    bool compileKernel(const std::string& source, const std::string& entryPoint,
                      const json& compileOpts, CompiledKernel& result);

    struct BufferSet {
        std::vector<cl_mem> inputBuffers;
        std::vector<cl_mem> outputBuffers;

        void clear() {
            for (auto buf : inputBuffers) {
                if (buf) clReleaseMemObject(buf);
            }
            for (auto buf : outputBuffers) {
                if (buf) clReleaseMemObject(buf);
            }
            inputBuffers.clear();
            outputBuffers.clear();
        }

        ~BufferSet() {
            clear();
        }
    };

    bool createInputBuffers(const TaskData& task, BufferSet& buffers);
    bool createOutputBuffers(const TaskData& task, BufferSet& buffers);
    bool setKernelArguments(cl_kernel kernel, const BufferSet& buffers, const TaskData& task);
    bool readOutputBuffers(const BufferSet& buffers, const TaskData& task, TaskResult& result);

public:
    OpenCLExecutor();
    ~OpenCLExecutor() override;

    bool initialize(const json& config = {}) override;
    void cleanup() override;
    TaskResult executeTask(const TaskData& task) override;
    std::string getFrameworkName() const override { return "opencl"; }
    json getCapabilities() const override;
};
#endif
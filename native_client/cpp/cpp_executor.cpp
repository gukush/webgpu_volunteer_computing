#include "cpp_executor.hpp"
#include <chrono>
#include <iostream>
#include <algorithm>
#include <cstring>

// Include the CPU kernels (inline here for simplicity, could be separate files)
#include <cmath>
#include <cstdint>

// ============================================================================
// CPU KERNELS (from previous artifact)
// ============================================================================

typedef unsigned __int128 uint128_t;

// Block Matrix Multiplication Kernel
void block_matrix_multiply(
    int block_size,
    int matrix_size,
    const float* block_a,
    const float* block_b,
    float* partial_result
) {
    for (int row = 0; row < block_size; row++) {
        for (int col = 0; col < block_size; col++) {
            float sum = 0.0f;
            for (int k = 0; k < block_size; k++) {
                float a_val = block_a[row * block_size + k];
                float b_val = block_b[k * block_size + col];
                sum += a_val * b_val;
            }
            int output_idx = row * block_size + col;
            partial_result[output_idx] = sum;
        }
    }
}

// Convolution Kernel
void convolution_kernel(
    int batch_size,
    int input_height,
    int input_width,
    int input_channels,
    int filter_height,
    int filter_width,
    int output_channels,
    int stride,
    int padding,
    const float* input_data,
    const float* filter_data,
    float* output_data
) {
    int output_height = (input_height + 2 * padding - filter_height) / stride + 1;
    int output_width = (input_width + 2 * padding - filter_width) / stride + 1;

    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < output_channels; oc++) {
            for (int oh = 0; oh < output_height; oh++) {
                for (int ow = 0; ow < output_width; ow++) {
                    float sum = 0.0f;

                    for (int ic = 0; ic < input_channels; ic++) {
                        for (int fh = 0; fh < filter_height; fh++) {
                            for (int fw = 0; fw < filter_width; fw++) {
                                int ih = oh * stride - padding + fh;
                                int iw = ow * stride - padding + fw;

                                if (ih >= 0 && ih < input_height && iw >= 0 && iw < input_width) {
                                    int input_idx = b * (input_channels * input_height * input_width) +
                                                  ic * (input_height * input_width) +
                                                  ih * input_width + iw;
                                    int filter_idx = oc * (input_channels * filter_height * filter_width) +
                                                   ic * (filter_height * filter_width) +
                                                   fh * filter_width + fw;
                                    sum += input_data[input_idx] * filter_data[filter_idx];
                                }
                            }
                        }
                    }

                    int output_idx = b * (output_channels * output_height * output_width) +
                                   oc * (output_height * output_width) +
                                   oh * output_width + ow;
                    output_data[output_idx] = sum;
                }
            }
        }
    }
}

// Simple ECM Stage 1 (simplified version for demo)
void ecm_stage1_kernel(
    uint64_t n,
    uint64_t curve_param_a,
    uint64_t curve_param_b,
    uint64_t bound1,
    int num_curves,
    const uint64_t* start_points_x,
    const uint64_t* start_points_y,
    uint64_t* gcd_results
) {
    // Simplified ECM implementation - replace with full version if needed
    for (int i = 0; i < num_curves; i++) {
        // Dummy computation for demonstration
        gcd_results[i] = (start_points_x[i] * start_points_y[i]) % n;
        if (gcd_results[i] == 0) gcd_results[i] = 1;
    }
}

// ============================================================================
// SORTING KERNELS
// ============================================================================

// Merge function for merge sort
template<typename T>
void merge(T* arr, int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<T> leftArr(n1);
    std::vector<T> rightArr(n2);

    for (int i = 0; i < n1; i++)
        leftArr[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        rightArr[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (leftArr[i] <= rightArr[j]) {
            arr[k] = leftArr[i];
            i++;
        } else {
            arr[k] = rightArr[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = leftArr[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = rightArr[j];
        j++;
        k++;
    }
}

// Merge sort implementation
template<typename T>
void merge_sort(T* arr, int left, int right) {
    if (left >= right) return;

    int mid = left + (right - left) / 2;

    merge_sort(arr, left, mid);
    merge_sort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

// Radix sort for integers (32-bit)
void radix_sort_int32(int32_t* arr, int n) {
    // Handle negative numbers by adding offset
    int32_t min_val = *std::min_element(arr, arr + n);
    uint32_t offset = (min_val < 0) ? static_cast<uint32_t>(-min_val) : 0;

    std::vector<uint32_t> temp_arr(n);
    for (int i = 0; i < n; i++) {
        temp_arr[i] = static_cast<uint32_t>(arr[i]) + offset;
    }

    // Find maximum value to determine number of digits
    uint32_t max_val = *std::max_element(temp_arr.begin(), temp_arr.end());

    // Counting sort for each digit
    std::vector<uint32_t> output(n);
    for (uint32_t exp = 1; max_val / exp > 0; exp *= 10) {
        std::vector<int> count(10, 0);

        // Count occurrences of each digit
        for (int i = 0; i < n; i++) {
            count[(temp_arr[i] / exp) % 10]++;
        }

        // Cumulative count
        for (int i = 1; i < 10; i++) {
            count[i] += count[i - 1];
        }

        // Build output array
        for (int i = n - 1; i >= 0; i--) {
            output[count[(temp_arr[i] / exp) % 10] - 1] = temp_arr[i];
            count[(temp_arr[i] / exp) % 10]--;
        }

        temp_arr = output;
    }

    // Convert back to original range
    for (int i = 0; i < n; i++) {
        arr[i] = static_cast<int32_t>(temp_arr[i] - offset);
    }
}

// Quick sort implementation
template<typename T>
int partition(T* arr, int low, int high) {
    T pivot = arr[high];
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            std::swap(arr[i], arr[j]);
        }
    }
    std::swap(arr[i + 1], arr[high]);
    return i + 1;
}

template<typename T>
void quick_sort(T* arr, int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);
        quick_sort(arr, low, pi - 1);
        quick_sort(arr, pi + 1, high);
    }
}

// ============================================================================
// CPU EXECUTOR IMPLEMENTATION
// ============================================================================

bool CPUExecutor::initialize(const json& config) {
    if (initialized) return true;

    try {
        // Setup device info
        deviceInfo = {
            {"name", "CPU Compute Engine"},
            {"type", "CPU"},
            {"vendor", "Native C++"},
            {"isCPU", true},
            {"isGPU", false},
            {"computeUnits", std::thread::hardware_concurrency()},
            {"maxWorkGroupSize", 1024},
            {"localMemorySize", 65536}
        };

        initialized = true;
        std::cout << "CPU executor initialized successfully" << std::endl;
        return true;

    } catch (const std::exception& e) {
        std::cerr << "CPU executor initialization failed: " << e.what() << std::endl;
        return false;
    }
}

void CPUExecutor::cleanup() {
    initialized = false;
}

json CPUExecutor::getCapabilities() const {
    return {
        {"framework", "cpp"},
        {"initialized", initialized},
        {"device", deviceInfo},
        {"supportedFrameworks", {"cpp"}},
        {"supportedKernels", {
            "matrix_multiply",
            "block_matrix_multiply",
            "convolution",
            "ecm_stage1",
            "sorting",
            "merge_sort",
            "radix_sort",
            "quick_sort"
        }}
    };
}

CPUKernelType CPUExecutor::detectKernelType(const TaskData& task) {
    // Check kernel source code or entry point to determine type
    std::string kernel = task.kernel;
    std::string entry = task.entry;

    // Transform to lowercase for easier matching
    std::transform(kernel.begin(), kernel.end(), kernel.begin(), ::tolower);
    std::transform(entry.begin(), entry.end(), entry.begin(), ::tolower);

    // Look for kernel type indicators in the task data
    if (task.chunkUniforms.contains("kernelType")) {
        std::string kernelType = task.chunkUniforms["kernelType"];
        std::transform(kernelType.begin(), kernelType.end(), kernelType.begin(), ::tolower);

        if (kernelType == "matrix_multiply" || kernelType == "block_matrix_multiply") {
            return CPUKernelType::MATRIX_MULTIPLY;
        } else if (kernelType == "convolution") {
            return CPUKernelType::CONVOLUTION;
        } else if (kernelType == "ecm_stage1") {
            return CPUKernelType::ECM_STAGE1;
        } else if (kernelType == "sorting" || kernelType == "merge_sort" ||
                   kernelType == "radix_sort" || kernelType == "quick_sort") {
            return CPUKernelType::SORTING;
        }
    }

    // Fallback: analyze kernel content or entry point
    if (kernel.find("matrix") != std::string::npos ||
        entry.find("matrix") != std::string::npos ||
        entry == "block_matrix_multiply") {
        return CPUKernelType::MATRIX_MULTIPLY;
    } else if (kernel.find("convolution") != std::string::npos ||
               kernel.find("conv") != std::string::npos ||
               entry.find("conv") != std::string::npos) {
        return CPUKernelType::CONVOLUTION;
    } else if (kernel.find("ecm") != std::string::npos ||
               entry.find("ecm") != std::string::npos) {
        return CPUKernelType::ECM_STAGE1;
    } else if (kernel.find("sort") != std::string::npos ||
               entry.find("sort") != std::string::npos ||
               kernel.find("merge") != std::string::npos ||
               kernel.find("radix") != std::string::npos ||
               kernel.find("quick") != std::string::npos) {
        return CPUKernelType::SORTING;
    }

    return CPUKernelType::UNKNOWN;
}

template<typename T>
std::vector<T> CPUExecutor::parseTypedData(const std::vector<uint8_t>& rawData) {
    std::vector<T> result(rawData.size() / sizeof(T));
    std::memcpy(result.data(), rawData.data(), rawData.size());
    return result;
}

template<typename T>
std::vector<uint8_t> CPUExecutor::serializeTypedData(const std::vector<T>& data) {
    std::vector<uint8_t> result(data.size() * sizeof(T));
    std::memcpy(result.data(), data.data(), data.size() * sizeof(T));
    return result;
}

TaskResult CPUExecutor::executeMatrixMultiplyKernel(const TaskData& task) {
    TaskResult result;
    result.success = false;

    try {
        // Extract parameters from task uniforms
        int block_size = task.chunkUniforms.value("block_size", 32);
        int matrix_size = task.chunkUniforms.value("matrix_size", 1024);

        std::cout << "Executing matrix multiply: block_size=" << block_size
                  << ", matrix_size=" << matrix_size << std::endl;

        // Ensure we have at least 2 inputs (block_a, block_b)
        if (task.inputData.size() < 2) {
            result.errorMessage = "Matrix multiply requires at least 2 inputs (block_a, block_b)";
            return result;
        }

        // Parse input data
        auto block_a = parseTypedData<float>(task.inputData[0]);
        auto block_b = parseTypedData<float>(task.inputData[1]);

        // Validate input sizes
        size_t expected_size = block_size * block_size;
        if (block_a.size() < expected_size || block_b.size() < expected_size) {
            result.errorMessage = "Input data size mismatch for matrix multiply";
            return result;
        }

        // Prepare output
        std::vector<float> partial_result(expected_size);

        // Execute kernel
        block_matrix_multiply(
            block_size,
            matrix_size,
            block_a.data(),
            block_b.data(),
            partial_result.data()
        );

        // Serialize result
        result.outputData = {serializeTypedData(partial_result)};
        result.success = true;

    } catch (const std::exception& e) {
        result.errorMessage = std::string("Matrix multiply execution error: ") + e.what();
    }

    return result;
}

TaskResult CPUExecutor::executeConvolutionKernel(const TaskData& task) {
    TaskResult result;
    result.success = false;

    try {
        // Extract parameters
        int batch_size = task.chunkUniforms.value("batch_size", 1);
        int input_height = task.chunkUniforms.value("input_height", 32);
        int input_width = task.chunkUniforms.value("input_width", 32);
        int input_channels = task.chunkUniforms.value("input_channels", 3);
        int filter_height = task.chunkUniforms.value("filter_height", 3);
        int filter_width = task.chunkUniforms.value("filter_width", 3);
        int output_channels = task.chunkUniforms.value("output_channels", 16);
        int stride = task.chunkUniforms.value("stride", 1);
        int padding = task.chunkUniforms.value("padding", 0);

        std::cout << "Executing convolution: " << batch_size << "x" << input_channels
                  << "x" << input_height << "x" << input_width << std::endl;

        // Ensure we have input and filter data
        if (task.inputData.size() < 2) {
            result.errorMessage = "Convolution requires at least 2 inputs (input_data, filter_data)";
            return result;
        }

        // Parse input data
        auto input_data = parseTypedData<float>(task.inputData[0]);
        auto filter_data = parseTypedData<float>(task.inputData[1]);

        // Calculate output dimensions
        int output_height = (input_height + 2 * padding - filter_height) / stride + 1;
        int output_width = (input_width + 2 * padding - filter_width) / stride + 1;
        size_t output_size = batch_size * output_channels * output_height * output_width;

        // Prepare output
        std::vector<float> output_data(output_size);

        // Execute kernel
        convolution_kernel(
            batch_size, input_height, input_width, input_channels,
            filter_height, filter_width, output_channels, stride, padding,
            input_data.data(), filter_data.data(), output_data.data()
        );

        // Serialize result
        result.outputData = {serializeTypedData(output_data)};
        result.success = true;

    } catch (const std::exception& e) {
        result.errorMessage = std::string("Convolution execution error: ") + e.what();
    }

    return result;
}

TaskResult CPUExecutor::executeECMKernel(const TaskData& task) {
    TaskResult result;
    result.success = false;

    try {
        // Extract parameters
        uint64_t n = task.chunkUniforms.value("n", 1024ULL);
        uint64_t curve_param_a = task.chunkUniforms.value("curve_param_a", 1ULL);
        uint64_t curve_param_b = task.chunkUniforms.value("curve_param_b", 1ULL);
        uint64_t bound1 = task.chunkUniforms.value("bound1", 100ULL);
        int num_curves = task.chunkUniforms.value("num_curves", 10);

        std::cout << "Executing ECM Stage 1: n=" << n << ", curves=" << num_curves << std::endl;

        // Ensure we have start points
        if (task.inputData.size() < 2) {
            result.errorMessage = "ECM requires at least 2 inputs (start_points_x, start_points_y)";
            return result;
        }

        // Parse input data
        auto start_points_x = parseTypedData<uint64_t>(task.inputData[0]);
        auto start_points_y = parseTypedData<uint64_t>(task.inputData[1]);

        // Validate input sizes
        if (start_points_x.size() < static_cast<size_t>(num_curves) ||
            start_points_y.size() < static_cast<size_t>(num_curves)) {
            result.errorMessage = "Insufficient start points for ECM";
            return result;
        }

        // Prepare output
        std::vector<uint64_t> gcd_results(num_curves);

        // Execute kernel
        ecm_stage1_kernel(
            n, curve_param_a, curve_param_b, bound1, num_curves,
            start_points_x.data(), start_points_y.data(), gcd_results.data()
        );

        // Serialize result
        result.outputData = {serializeTypedData(gcd_results)};
        result.success = true;

    } catch (const std::exception& e) {
        result.errorMessage = std::string("ECM execution error: ") + e.what();
    }

    return result;
}

TaskResult CPUExecutor::executeSortingKernel(const TaskData& task) {
    TaskResult result;
    result.success = false;

    try {
        // Extract parameters
        std::string sort_algorithm = task.chunkUniforms.value("sort_algorithm", "merge_sort");
        std::string data_type = task.chunkUniforms.value("data_type", "int32");
        bool ascending = task.chunkUniforms.value("ascending", true);

        std::cout << "Executing " << sort_algorithm << " on " << data_type
                  << " data (" << (ascending ? "ascending" : "descending") << ")" << std::endl;

        // Ensure we have input data
        if (task.inputData.empty()) {
            result.errorMessage = "Sorting requires at least 1 input (data array)";
            return result;
        }

        // Handle different data types
        if (data_type == "int32") {
            auto data = parseTypedData<int32_t>(task.inputData[0]);
            int n = static_cast<int>(data.size());

            std::cout << "Sorting " << n << " int32 elements" << std::endl;

            // Choose sorting algorithm
            if (sort_algorithm == "merge_sort" || sort_algorithm == "merge") {
                merge_sort(data.data(), 0, n - 1);
            } else if (sort_algorithm == "radix_sort" || sort_algorithm == "radix") {
                radix_sort_int32(data.data(), n);
            } else if (sort_algorithm == "quick_sort" || sort_algorithm == "quick") {
                quick_sort(data.data(), 0, n - 1);
            } else {
                // Default to merge sort
                merge_sort(data.data(), 0, n - 1);
            }

            // Reverse for descending order
            if (!ascending) {
                std::reverse(data.begin(), data.end());
            }

            result.outputData = {serializeTypedData(data)};

        } else if (data_type == "float32") {
            auto data = parseTypedData<float>(task.inputData[0]);
            int n = static_cast<int>(data.size());

            std::cout << "Sorting " << n << " float32 elements" << std::endl;

            // Float sorting (radix sort not suitable, use merge or quick)
            if (sort_algorithm == "quick_sort" || sort_algorithm == "quick") {
                quick_sort(data.data(), 0, n - 1);
            } else {
                // Default to merge sort for floats
                merge_sort(data.data(), 0, n - 1);
            }

            // Reverse for descending order
            if (!ascending) {
                std::reverse(data.begin(), data.end());
            }

            result.outputData = {serializeTypedData(data)};

        } else if (data_type == "uint32") {
            auto data = parseTypedData<uint32_t>(task.inputData[0]);
            int n = static_cast<int>(data.size());

            std::cout << "Sorting " << n << " uint32 elements" << std::endl;

            // Choose sorting algorithm
            if (sort_algorithm == "merge_sort" || sort_algorithm == "merge") {
                merge_sort(data.data(), 0, n - 1);
            } else if (sort_algorithm == "quick_sort" || sort_algorithm == "quick") {
                quick_sort(data.data(), 0, n - 1);
            } else {
                // Default to merge sort for unsigned integers
                merge_sort(data.data(), 0, n - 1);
            }

            // Reverse for descending order
            if (!ascending) {
                std::reverse(data.begin(), data.end());
            }

            result.outputData = {serializeTypedData(data)};

        } else if (data_type == "uint64") {
            auto data = parseTypedData<uint64_t>(task.inputData[0]);
            int n = static_cast<int>(data.size());

            std::cout << "Sorting " << n << " uint64 elements" << std::endl;

            // Choose sorting algorithm
            if (sort_algorithm == "merge_sort" || sort_algorithm == "merge") {
                merge_sort(data.data(), 0, n - 1);
            } else if (sort_algorithm == "quick_sort" || sort_algorithm == "quick") {
                quick_sort(data.data(), 0, n - 1);
            } else {
                // Default to merge sort
                merge_sort(data.data(), 0, n - 1);
            }

            // Reverse for descending order
            if (!ascending) {
                std::reverse(data.begin(), data.end());
            }

            result.outputData = {serializeTypedData(data)};

        } else {
            result.errorMessage = "Unsupported data type: " + data_type +
                                ". Supported types: int32, float32, uint32, uint64";
            return result;
        }

        result.success = true;

    } catch (const std::exception& e) {
        result.errorMessage = std::string("Sorting execution error: ") + e.what();
    }

    return result;
}

TaskResult CPUExecutor::executeTask(const TaskData& task) {
    if (!initialized) {
        TaskResult result;
        result.success = false;
        result.errorMessage = "CPU executor not initialized";
        return result;
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Detect kernel type
    CPUKernelType kernelType = detectKernelType(task);

    TaskResult result;

    switch (kernelType) {
        case CPUKernelType::MATRIX_MULTIPLY:
            result = executeMatrixMultiplyKernel(task);
            break;

        case CPUKernelType::CONVOLUTION:
            result = executeConvolutionKernel(task);
            break;

        case CPUKernelType::ECM_STAGE1:
            result = executeECMKernel(task);
            break;

        case CPUKernelType::SORTING:
            result = executeSortingKernel(task);
            break;

        case CPUKernelType::UNKNOWN:
        default:
            result.success = false;
            result.errorMessage = "Unknown or unsupported kernel type for CPU execution";
            break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    result.processingTime = duration.count();

    if (result.success) {
        std::cout << "CPU kernel executed in " << result.processingTime << "ms" << std::endl;
    }

    return result;
}
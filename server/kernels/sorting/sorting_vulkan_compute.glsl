#version 450

// Bitonic Sort Vulkan Compute Shader
// Sorts arrays of 32-bit integers using bitonic sorting algorithm
// Supports both ascending and descending order
// Works on power-of-2 sized arrays

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

// Input/Output buffer for the array to be sorted
layout(std430, binding = 0) restrict buffer DataBuffer {
    uint data[];
};

// Push constants for sort parameters
layout(push_constant) uniform PushConstants {
    uint array_size;      // Size of array (must be power of 2)
    uint stage;          // Current bitonic stage (0 to log2(n)-1)
    uint substage;       // Current substage within the stage
    uint ascending;      // 1 for ascending, 0 for descending
};

// Shared memory for local sorting within work group
shared uint local_data[512]; // 2 elements per thread for 256 threads

// Compare and exchange function
void compareExchange(uint i, uint j, bool dir) {
    if (i < array_size && j < array_size) {
        uint a = data[i];
        uint b = data[j];

        if ((a > b) == dir) {
            data[i] = b;
            data[j] = a;
        }
    }
}

// Compare and exchange for shared memory
void compareExchangeLocal(uint i, uint j, bool dir) {
    uint a = local_data[i];
    uint b = local_data[j];

    if ((a > b) == dir) {
        local_data[i] = b;
        local_data[j] = a;
    }
}

// Bitonic merge kernel - merges two bitonic sequences
void bitonicMerge() {
    uint thread_id = gl_GlobalInvocationID.x;
    uint group_size = 1u << substage;
    uint group_id = thread_id / group_size;
    uint local_id = thread_id % group_size;

    // Determine sort direction for this group
    uint stage_size = 1u << stage;
    bool sort_dir = ((group_id / (stage_size / group_size)) % 2) == 0;
    if (ascending == 0u) sort_dir = !sort_dir;

    // Calculate indices to compare
    uint distance = group_size / 2u;
    uint i = group_id * group_size + local_id;
    uint j = i + distance;

    if (local_id < distance) {
        compareExchange(i, j, sort_dir);
    }
}

// Local bitonic sort using shared memory for better performance
void bitonicSortLocal() {
    uint local_id = gl_LocalInvocationID.x;
    uint group_id = gl_WorkGroupID.x;
    uint group_size = gl_WorkGroupSize.x;

    // Load data into shared memory (2 elements per thread)
    uint base_idx = group_id * group_size * 2u;
    if (base_idx + local_id < array_size) {
        local_data[local_id] = data[base_idx + local_id];
    } else {
        local_data[local_id] = 0xFFFFFFFFu; // Max value for padding
    }

    if (base_idx + local_id + group_size < array_size) {
        local_data[local_id + group_size] = data[base_idx + local_id + group_size];
    } else {
        local_data[local_id + group_size] = 0xFFFFFFFFu;
    }

    barrier();

    // Perform bitonic sort on shared memory
    uint n = group_size * 2u;

    // Build bitonic sequence
    for (uint stage_size = 2u; stage_size <= n; stage_size *= 2u) {
        for (uint substage_size = stage_size / 2u; substage_size > 0u; substage_size /= 2u) {
            uint thread_group = local_id / substage_size;
            uint thread_local = local_id % substage_size;

            // Determine direction
            bool dir = ((thread_group / (stage_size / substage_size)) % 2) == 0;
            if (ascending == 0u) dir = !dir;

            uint i = thread_group * substage_size + thread_local;
            uint j = i + substage_size;

            if (thread_local < substage_size / 2u && j < n) {
                compareExchangeLocal(i, j, dir);
            }

            barrier();
        }
    }

    // Write back to global memory
    if (base_idx + local_id < array_size) {
        data[base_idx + local_id] = local_data[local_id];
    }
    if (base_idx + local_id + group_size < array_size) {
        data[base_idx + local_id + group_size] = local_data[local_id + group_size];
    }
}

void main() {
    // For small arrays or initial stages, use local sort
    if (stage == 0xFFFFFFFFu) {
        // Special case: do complete local sort
        bitonicSortLocal();
        return;
    }

    // For larger stages, use global bitonic merge
    bitonicMerge();
}

// Additional shader for initial local sorting
// This would be a separate compute shader file: bitonic_local_sort.comp

/*
#version 450

layout(local_size_x = 256, local_size_y = 1, local_size_z = 1) in;

layout(std430, binding = 0) restrict buffer DataBuffer {
    uint data[];
};

layout(push_constant) uniform PushConstants {
    uint array_size;
    uint ascending;
};

shared uint local_data[512];

void compareExchangeLocal(uint i, uint j, bool dir) {
    uint a = local_data[i];
    uint b = local_data[j];

    if ((a > b) == dir) {
        local_data[i] = b;
        local_data[j] = a;
    }
}

void main() {
    uint local_id = gl_LocalInvocationID.x;
    uint group_id = gl_WorkGroupID.x;
    uint group_size = gl_WorkGroupSize.x;
    uint elements_per_group = group_size * 2u;
    uint base_idx = group_id * elements_per_group;

    // Load data into shared memory
    if (base_idx + local_id < array_size) {
        local_data[local_id] = data[base_idx + local_id];
    } else {
        local_data[local_id] = (ascending == 1u) ? 0xFFFFFFFFu : 0u;
    }

    if (base_idx + local_id + group_size < array_size) {
        local_data[local_id + group_size] = data[base_idx + local_id + group_size];
    } else {
        local_data[local_id + group_size] = (ascending == 1u) ? 0xFFFFFFFFu : 0u;
    }

    barrier();

    // Bitonic sort within shared memory
    for (uint stage = 2u; stage <= elements_per_group; stage *= 2u) {
        for (uint substage = stage / 2u; substage > 0u; substage /= 2u) {
            uint group_idx = local_id / substage;
            uint local_idx = local_id % substage;

            bool dir = ((group_idx / (stage / substage)) % 2) == 0;
            if (ascending == 0u) dir = !dir;

            uint i = group_idx * substage + local_idx;
            uint j = i + substage;

            if (local_idx < substage / 2u && j < elements_per_group) {
                compareExchangeLocal(i, j, dir);
            }

            barrier();
        }
    }

    // Write results back to global memory
    if (base_idx + local_id < array_size) {
        data[base_idx + local_id] = local_data[local_id];
    }
    if (base_idx + local_id + group_size < array_size) {
        data[base_idx + local_id + group_size] = local_data[local_id + group_size];
    }
}
*/
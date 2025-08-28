// Bitonic Sort WebGPU Compute Shader (WGSL)
// Sorts arrays of 32-bit integers using bitonic sorting algorithm
// Supports both ascending and descending order

@group(0) @binding(0) var<storage, read_write> data: array<u32>;

struct PushConstants {
    array_size: u32,
    stage: u32,
    substage: u32,
    ascending: u32,
}

@group(0) @binding(1) var<uniform> params: PushConstants;

// Shared memory for local operations
var<workgroup> local_data: array<u32, 512>;

// Compare and exchange function for global memory
fn compare_exchange_global(i: u32, j: u32, dir: bool) {
    if (i < params.array_size && j < params.array_size) {
        let a = data[i];
        let b = data[j];

        if ((a > b) == dir) {
            data[i] = b;
            data[j] = a;
        }
    }
}

// Compare and exchange function for workgroup memory
fn compare_exchange_local(i: u32, j: u32, dir: bool) {
    let a = local_data[i];
    let b = local_data[j];

    if ((a > b) == dir) {
        local_data[i] = b;
        local_data[j] = a;
    }
}

// Bitonic merge for global operations
fn bitonic_merge_global(global_id: u32) {
    let group_size = 1u << params.substage;
    let group_id = global_id / group_size;
    let local_id = global_id % group_size;

    // Determine sort direction for this group
    let stage_size = 1u << params.stage;
    var sort_dir = ((group_id / (stage_size / group_size)) % 2u) == 0u;
    if (params.ascending == 0u) {
        sort_dir = !sort_dir;
    }

    // Calculate indices to compare
    let distance = group_size / 2u;
    let i = group_id * group_size + local_id;
    let j = i + distance;

    if (local_id < distance) {
        compare_exchange_global(i, j, sort_dir);
    }
}

// Local bitonic sort using workgroup memory
fn bitonic_sort_local(local_id: u32, workgroup_id: u32) {
    let workgroup_size = 256u;
    let elements_per_workgroup = workgroup_size * 2u;
    let base_idx = workgroup_id * elements_per_workgroup;

    // Load data into workgroup memory (2 elements per thread)
    if (base_idx + local_id < params.array_size) {
        local_data[local_id] = data[base_idx + local_id];
    } else {
        local_data[local_id] = select(0u, 0xFFFFFFFFu, params.ascending == 1u);
    }

    if (base_idx + local_id + workgroup_size < params.array_size) {
        local_data[local_id + workgroup_size] = data[base_idx + local_id + workgroup_size];
    } else {
        local_data[local_id + workgroup_size] = select(0u, 0xFFFFFFFFu, params.ascending == 1u);
    }

    workgroupBarrier();

    // Perform bitonic sort on workgroup memory
    let n = elements_per_workgroup;

    // Build bitonic sequence
    var stage_size = 2u;
    while (stage_size <= n) {
        var substage_size = stage_size / 2u;
        while (substage_size > 0u) {
            let thread_group = local_id / substage_size;
            let thread_local = local_id % substage_size;

            // Determine direction
            var dir = ((thread_group / (stage_size / substage_size)) % 2u) == 0u;
            if (params.ascending == 0u) {
                dir = !dir;
            }

            let i = thread_group * substage_size + thread_local;
            let j = i + substage_size;

            if (thread_local < substage_size / 2u && j < n) {
                compare_exchange_local(i, j, dir);
            }

            workgroupBarrier();
            substage_size = substage_size / 2u;
        }
        stage_size = stage_size * 2u;
    }

    // Write back to global memory
    if (base_idx + local_id < params.array_size) {
        data[base_idx + local_id] = local_data[local_id];
    }
    if (base_idx + local_id + workgroup_size < params.array_size) {
        data[base_idx + local_id + workgroup_size] = local_data[local_id + workgroup_size];
    }
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let global_thread_id = global_id.x;
    let local_thread_id = local_id.x;
    let workgroup_thread_id = workgroup_id.x;

    // Check if this is a local sort pass
    if (params.stage == 0xFFFFFFFFu) {
        // Special case: do complete local sort
        bitonic_sort_local(local_thread_id, workgroup_thread_id);
        return;
    }

    // Regular bitonic merge pass
    bitonic_merge_global(global_thread_id);
}

// Alternative shader for initial local sorting only
// This would be used for the first pass to sort small chunks locally

/*
@compute @workgroup_size(256, 1, 1)
fn main_local_sort(@builtin(global_invocation_id) global_id: vec3<u32>,
                   @builtin(local_invocation_id) local_id: vec3<u32>,
                   @builtin(workgroup_id) workgroup_id: vec3<u32>) {

    let local_thread_id = local_id.x;
    let workgroup_thread_id = workgroup_id.x;
    let workgroup_size = 256u;
    let elements_per_workgroup = workgroup_size * 2u;
    let base_idx = workgroup_thread_id * elements_per_workgroup;

    // Load data into workgroup memory
    if (base_idx + local_thread_id < params.array_size) {
        local_data[local_thread_id] = data[base_idx + local_thread_id];
    } else {
        local_data[local_thread_id] = select(0u, 0xFFFFFFFFu, params.ascending == 1u);
    }

    if (base_idx + local_thread_id + workgroup_size < params.array_size) {
        local_data[local_thread_id + workgroup_size] = data[base_idx + local_thread_id + workgroup_size];
    } else {
        local_data[local_thread_id + workgroup_size] = select(0u, 0xFFFFFFFFu, params.ascending == 1u);
    }

    workgroupBarrier();

    // Bitonic sort within workgroup memory
    var stage = 2u;
    while (stage <= elements_per_workgroup) {
        var substage = stage / 2u;
        while (substage > 0u) {
            let group_idx = local_thread_id / substage;
            let local_idx = local_thread_id % substage;

            var dir = ((group_idx / (stage / substage)) % 2u) == 0u;
            if (params.ascending == 0u) {
                dir = !dir;
            }

            let i = group_idx * substage + local_idx;
            let j = i + substage;

            if (local_idx < substage / 2u && j < elements_per_workgroup) {
                compare_exchange_local(i, j, dir);
            }

            workgroupBarrier();
            substage = substage / 2u;
        }
        stage = stage * 2u;
    }

    // Write results back to global memory
    if (base_idx + local_thread_id < params.array_size) {
        data[base_idx + local_thread_id] = local_data[local_thread_id];
    }
    if (base_idx + local_thread_id + workgroup_size < params.array_size) {
        data[base_idx + local_thread_id + workgroup_size] = local_data[local_thread_id + workgroup_size];
    }
}
*/

// Utility functions that can be used in additional passes

// Function to check if array is power of 2
fn is_power_of_two(n: u32) -> bool {
    return (n & (n - 1u)) == 0u && n != 0u;
}

// Function to find next power of 2
fn next_power_of_two(n: u32) -> u32 {
    var result = 1u;
    while (result < n) {
        result = result * 2u;
    }
    return result;
}

// Bitonic sort verification function (for debugging)
fn is_bitonic_sequence(start: u32, length: u32, ascending: bool) -> bool {
    if (length <= 1u) {
        return true;
    }

    let half = length / 2u;
    var increasing = true;
    var decreasing = true;

    // Check first half
    for (var i = start; i < start + half - 1u; i = i + 1u) {
        if (i >= params.array_size || i + 1u >= params.array_size) {
            break;
        }
        if (data[i] > data[i + 1u]) {
            increasing = false;
        }
        if (data[i] < data[i + 1u]) {
            decreasing = false;
        }
    }

    // Check second half
    for (var i = start + half; i < start + length - 1u; i = i + 1u) {
        if (i >= params.array_size || i + 1u >= params.array_size) {
            break;
        }
        if (data[i] < data[i + 1u]) {
            increasing = false;
        }
        if (data[i] > data[i + 1u]) {
            decreasing = false;
        }
    }

    return (ascending && increasing) || (!ascending && decreasing);
}
// DistributedSortChunkingStrategy.js - Multi-phase distributed sorting strategy
import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';
import { v4 as uuidv4 } from 'uuid';

export default class DistributedSortChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('distributed_sort');
  }

  defineInputSchema() {
    return {
      inputs: [
        {
          name: 'unsorted_data',
          type: 'storage_buffer',
          binding: 1,
          elementType: 'f32', // or u32 for integer keys
          chunking: 'parallel'
        }
      ],
      outputs: [
        {
          name: 'sorted_data',
          type: 'storage_buffer',
          binding: 2,
          elementType: 'f32'
        }
      ],
      uniforms: [
        {
          name: 'sort_params',
          type: 'uniform_buffer',
          binding: 0,
          fields: [
            { name: 'data_size', type: 'u32' },
            { name: 'sort_phase', type: 'u32' },
            { name: 'merge_level', type: 'u32' },
            { name: 'is_merge_operation', type: 'u32' }
          ]
        }
      ]
    };
  }

  planExecution(workload) {
    const { 
      dataSize, 
      chunkSize = 1024 * 1024, // 1MB chunks
      elementSize = 4,
      sortAlgorithm = 'radix'
    } = workload.metadata;

    if (!dataSize) {
      throw new Error('dataSize required in metadata');
    }

    const elementsPerChunk = Math.floor(chunkSize / elementSize);
    const totalChunks = Math.ceil(dataSize / elementsPerChunk);

    return {
      strategy: this.name,
      executionModel: 'iterative_refinement',
      totalPhases: this.calculateMergePhases(totalChunks),
      totalChunks: totalChunks, // Initial sorting chunks
      schema: this.defineInputSchema(),
      metadata: {
        ...workload.metadata,
        elementsPerChunk,
        totalChunks,
        sortAlgorithm,
        elementSize,
        dataSize
      },
      assemblyStrategy: 'distributed_sort_assembly',
      phases: this.planPhases(totalChunks)
    };
  }

  calculateMergePhases(initialChunks) {
    // Phase 0: Local sorting
    // Phase 1+: Binary tree merging
    return Math.ceil(Math.log2(initialChunks)) + 1;
  }

  planPhases(totalChunks) {
    const phases = [];
    
    // Phase 0: Local sorting phase
    phases.push({
      phaseId: 'local_sort',
      phaseType: 'sort',
      expectedChunks: totalChunks,
      description: 'GPU-accelerated local sorting of data shards'
    });

    // Phase 1+: Merge phases (binary tree)
    let remainingChunks = totalChunks;
    let phaseNum = 1;
    
    while (remainingChunks > 1) {
      const mergeChunks = Math.ceil(remainingChunks / 2);
      phases.push({
        phaseId: `merge_level_${phaseNum}`,
        phaseType: 'merge',
        expectedChunks: mergeChunks,
        mergeLevel: phaseNum,
        inputChunks: remainingChunks,
        description: `Merge level ${phaseNum}: ${remainingChunks} → ${mergeChunks} chunks`
      });
      
      remainingChunks = mergeChunks;
      phaseNum++;
    }

    return phases;
  }

  createPhaseChunkDescriptors(plan, phase, globalState) {
    switch (phase.phaseType) {
      case 'sort':
        return this.createSortChunkDescriptors(plan, phase, globalState);
      case 'merge':
        return this.createMergeChunkDescriptors(plan, phase, globalState);
      default:
        throw new Error(`Unknown phase type: ${phase.phaseType}`);
    }
  }

  createSortChunkDescriptors(plan, phase, globalState) {
    const { elementsPerChunk, elementSize, sortAlgorithm } = plan.metadata;
    const schema = plan.schema;
    const parsedInputs = this.parseMultipleInputs(plan.metadata.inputData, schema);
    
    const descriptors = [];
    const firstInputKey = Object.keys(parsedInputs)[0];
    const inputBuffer = firstInputKey ? Buffer.from(parsedInputs[firstInputKey], 'base64') : Buffer.alloc(0);

    for (let chunkIndex = 0; chunkIndex < phase.expectedChunks; chunkIndex++) {
      const startByte = chunkIndex * elementsPerChunk * elementSize;
      const endByte = Math.min((chunkIndex + 1) * elementsPerChunk * elementSize, inputBuffer.length);
      const chunkData = inputBuffer.slice(startByte, endByte);
      const actualElements = chunkData.length / elementSize;

      descriptors.push({
        chunkId: `sort-${chunkIndex}`,
        chunkIndex,
        parentId: plan.parentId,
        
        framework: 'webgpu',
        kernel: this.getRadixSortShader(),
        entry: 'radix_sort_main',
        workgroupCount: [Math.ceil(actualElements / 256), 1, 1],

        inputs: [
          { name: 'unsorted_data', data: chunkData.toString('base64') }
        ],
        outputSizes: [chunkData.length], // Same size, but sorted

        uniforms: {
          data_size: actualElements,
          sort_phase: 0,
          merge_level: 0,
          is_merge_operation: 0
        },

        assemblyMetadata: {
          chunkIndex,
          phaseType: 'sort',
          dataSize: chunkData.length,
          elementCount: actualElements
        }
      });
    }

    return descriptors;
  }

  createMergeChunkDescriptors(plan, phase, globalState) {
    const descriptors = [];
    const sortedRuns = globalState.currentArray; // Array of sorted data chunks from previous phase
    
    // Create merge operations pairing up sorted runs
    let chunkIndex = 0;
    for (let i = 0; i < sortedRuns.length; i += 2) {
      const run1 = sortedRuns[i];
      const run2 = sortedRuns[i + 1] || null; // Handle odd number of runs

      if (!run2) {
        // Odd run out - just pass through
        descriptors.push({
          chunkId: `merge-${phase.mergeLevel}-${chunkIndex}`,
          chunkIndex,
          parentId: plan.parentId,
          
          framework: 'webgpu',
          kernel: this.getPassThroughShader(),
          entry: 'pass_through_main',
          workgroupCount: [Math.ceil(run1.length / 1024), 1, 1],

          inputs: [
            { name: 'input_data', data: Buffer.from(run1.buffer).toString('base64') }
          ],
          outputSizes: [run1.length * 4],

          uniforms: {
            data_size: run1.length,
            sort_phase: phase.mergeLevel,
            merge_level: phase.mergeLevel,
            is_merge_operation: 0
          },

          assemblyMetadata: {
            chunkIndex,
            phaseType: 'merge',
            mergeLevel: phase.mergeLevel,
            inputCount: 1
          }
        });
      } else {
        // Two runs to merge
        const outputSize = (run1.length + run2.length) * 4;
        
        descriptors.push({
          chunkId: `merge-${phase.mergeLevel}-${chunkIndex}`,
          chunkIndex,
          parentId: plan.parentId,
          
          framework: 'webgpu',
          kernel: this.getParallelMergeShader(),
          entry: 'parallel_merge_main',
          workgroupCount: [Math.ceil((run1.length + run2.length) / 256), 1, 1],

          inputs: [
            { name: 'sorted_run_a', data: Buffer.from(run1.buffer).toString('base64') },
            { name: 'sorted_run_b', data: Buffer.from(run2.buffer).toString('base64') }
          ],
          outputSizes: [outputSize],

          uniforms: {
            run_a_size: run1.length,
            run_b_size: run2.length,
            sort_phase: phase.mergeLevel,
            merge_level: phase.mergeLevel,
            is_merge_operation: 1
          },

          assemblyMetadata: {
            chunkIndex,
            phaseType: 'merge',
            mergeLevel: phase.mergeLevel,
            inputCount: 2,
            outputSize
          }
        });
      }

      chunkIndex++;
    }

    return descriptors;
  }

  updateGlobalState(globalState, phaseResults, phase) {
    const newState = { ...globalState };
    
    // Convert results back to Float32Arrays for next phase
    const processedResults = phaseResults.map(result => {
      const buffer = Buffer.from(result.result, 'base64');
      return new Float32Array(buffer.buffer, buffer.byteOffset, buffer.byteLength / 4);
    });

    newState.currentArray = processedResults;
    newState.currentPhase = phase;
    newState.completedPhases++;

    console.log(`DistributedSort: Phase ${phase.phaseId} complete. ${processedResults.length} chunks remaining.`);

    return newState;
  }

  shouldContinue(globalState, phase, plan) {
    // Continue until we have a single sorted result
    return globalState.currentArray.length > 1;
  }

  assembleFinalResult(globalState, plan) {
    if (globalState.currentArray.length !== 1) {
      throw new Error(`Expected single final result, got ${globalState.currentArray.length} chunks`);
    }

    const finalSortedData = globalState.currentArray[0];
    const resultBuffer = Buffer.from(finalSortedData.buffer);

    return {
      success: true,
      data: resultBuffer.toString('base64'),
      metadata: {
        strategy: this.name,
        sortedElements: finalSortedData.length,
        totalPhases: globalState.completedPhases,
        algorithm: 'distributed_merge_tree'
      }
    };
  }

  // GPU Shader Generation Methods

  getRadixSortShader() {
    return `
      struct SortParams {
        data_size: u32,
        sort_phase: u32,
        merge_level: u32,
        is_merge_operation: u32,
      }

      @group(0) @binding(0) var<uniform> params: SortParams;
      @group(0) @binding(1) var<storage, read_write> data: array<f32>;

      // Shared memory for local sorting
      var<workgroup> shared_data: array<f32, 256>;
      var<workgroup> shared_keys: array<u32, 256>;

      @compute @workgroup_size(256, 1, 1)
      fn radix_sort_main(@builtin(global_invocation_id) gid: vec3<u32>,
                         @builtin(local_invocation_id) lid: vec3<u32>,
                         @builtin(workgroup_id) wid: vec3<u32>) {
        let global_idx = gid.x;
        let local_idx = lid.x;
        let workgroup_id = wid.x;
        
        // Load data into shared memory
        if (global_idx < params.data_size) {
          shared_data[local_idx] = data[global_idx];
          shared_keys[local_idx] = bitcast<u32>(data[global_idx]);
        } else {
          shared_data[local_idx] = f32(0x7F800000); // +infinity for padding
          shared_keys[local_idx] = 0xFFFFFFFF;
        }
        
        workgroupBarrier();

        // Radix sort within workgroup (simplified - normally would do multiple passes)
        // This is a simplified bitonic sort for demonstration
        for (var stage: u32 = 2u; stage <= 256u; stage = stage << 1u) {
          for (var step: u32 = stage >> 1u; step > 0u; step = step >> 1u) {
            let partner = local_idx ^ step;
            
            if (partner > local_idx) {
              let should_swap = ((local_idx & stage) == 0u && shared_keys[local_idx] > shared_keys[partner]) ||
                               ((local_idx & stage) != 0u && shared_keys[local_idx] < shared_keys[partner]);
              
              if (should_swap) {
                // Swap
                let temp_data = shared_data[local_idx];
                let temp_key = shared_keys[local_idx];
                shared_data[local_idx] = shared_data[partner];
                shared_keys[local_idx] = shared_keys[partner];
                shared_data[partner] = temp_data;
                shared_keys[partner] = temp_key;
              }
            }
            
            workgroupBarrier();
          }
        }

        // Write back sorted data
        if (global_idx < params.data_size) {
          data[global_idx] = shared_data[local_idx];
        }
      }
    `;
  }

  getParallelMergeShader() {
    return `
      struct MergeParams {
        run_a_size: u32,
        run_b_size: u32,
        sort_phase: u32,
        merge_level: u32,
      }

      @group(0) @binding(0) var<uniform> params: MergeParams;
      @group(0) @binding(1) var<storage, read> run_a: array<f32>;
      @group(0) @binding(2) var<storage, read> run_b: array<f32>;
      @group(0) @binding(3) var<storage, write> merged_output: array<f32>;

      @compute @workgroup_size(256, 1, 1)
      fn parallel_merge_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let output_idx = gid.x;
        let total_size = params.run_a_size + params.run_b_size;
        
        if (output_idx >= total_size) {
          return;
        }

        // Binary search to find position in merged array
        // This is a simplified parallel merge - production would use more sophisticated algorithm
        
        // Find how many elements from run_a should come before output_idx
        var left = 0u;
        var right = min(params.run_a_size, output_idx + 1u);
        
        while (left < right) {
          let mid = (left + right) / 2u;
          let b_count = output_idx - mid;
          
          let a_val = if (mid < params.run_a_size) { run_a[mid] } else { f32(0x7F800000) };
          let b_val = if (b_count < params.run_b_size) { run_b[b_count] } else { f32(0x7F800000) };
          
          if (a_val <= b_val) {
            left = mid + 1u;
          } else {
            right = mid;
          }
        }
        
        let a_count = left;
        let b_count = output_idx - a_count;
        
        // Determine which array to take from
        let use_a = (a_count < params.run_a_size) && 
                   (b_count >= params.run_b_size || run_a[a_count] <= run_b[b_count]);
        
        if (use_a) {
          merged_output[output_idx] = run_a[a_count];
        } else {
          merged_output[output_idx] = run_b[b_count];
        }
      }
    `;
  }

  getPassThroughShader() {
    return `
      struct PassParams {
        data_size: u32,
        sort_phase: u32,
        merge_level: u32,
        is_merge_operation: u32,
      }

      @group(0) @binding(0) var<uniform> params: PassParams;
      @group(0) @binding(1) var<storage, read> input_data: array<f32>;
      @group(0) @binding(2) var<storage, write> output_data: array<f32>;

      @compute @workgroup_size(256, 1, 1)
      fn pass_through_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let idx = gid.x;
        if (idx >= params.data_size) {
          return;
        }
        output_data[idx] = input_data[idx];
      }
    `;
  }
}

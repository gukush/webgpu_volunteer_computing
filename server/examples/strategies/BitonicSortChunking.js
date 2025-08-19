export default class BitonicSortChunking extends BaseChunkingStrategy {
  constructor() {
    super('bitonic_sort');
  }

  validateWorkload(workload) {
    const { arraySize, chunkSize } = workload.metadata;

    // Bitonic sort requires power-of-2 array size
    if (!arraySize || (arraySize & (arraySize - 1)) !== 0) {
      return {
        valid: false,
        error: 'Array size must be a power of 2 for bitonic sort'
      };
    }

    if (!chunkSize || chunkSize <= 0) {
      return {
        valid: false,
        error: 'Chunk size must be specified and > 0'
      };
    }

    return { valid: true };
  }

  planExecution(workload) {
    const { arraySize, chunkSize } = workload.metadata;
    const numChunks = Math.ceil(arraySize / chunkSize);
    const stages = Math.log2(arraySize);

    // Generate all phases of bitonic sort
    const phases = this.generateBitonicPhases(arraySize, chunkSize);

    return {
      strategy: this.name,
      executionModel: 'iterative_refinement', // Multiple phases with sync
      totalPhases: phases.length,
      phases: phases,
      metadata: {
        arraySize,
        chunkSize,
        numChunks,
        stages,
        inputBuffer: Buffer.from(workload.input, 'base64')
      },
      assemblyStrategy: 'bitonic_sort_assembly'
    };
  }

  generateBitonicPhases(arraySize, chunkSize) {
    const phases = [];
    const numStages = Math.log2(arraySize);

    // Bitonic sort has numStages stages
    for (let stage = 0; stage < numStages; stage++) {
      // Each stage has (stage + 1) steps
      for (let step = 0; step <= stage; step++) {
        phases.push({
          phaseId: `stage-${stage}-step-${step}`,
          stage: stage,
          step: step,
          stageTotal: stage + 1,
          operation: 'bitonic_compare_exchange',

          // Phase parameters for the kernel
          phaseParams: {
            stage: stage,
            step: step,
            arraySize: arraySize,
            chunkSize: chunkSize,

            // Bitonic sort parameters
            compareDistance: 1 << (stage - step),
            blockSize: 1 << (stage + 1),
            ascending: true // We'll sort in ascending order
          },

          // Synchronization: wait for previous step in same stage
          waitForPrevious: step > 0,
          dependencies: step > 0 ? [`stage-${stage}-step-${step-1}`] : []
        });
      }
    }

    return phases;
  }

  createChunkDescriptors(plan) {
    // This will be called once per phase during execution
    throw new Error('Use createPhaseChunkDescriptors for iterative workloads');
  }

  createPhaseChunkDescriptors(plan, phase, globalState) {
    const { arraySize, chunkSize, numChunks } = plan.metadata;
    const descriptors = [];

    // Create one chunk descriptor for each data chunk
    for (let chunkIndex = 0; chunkIndex < numChunks; chunkIndex++) {
      const startElement = chunkIndex * chunkSize;
      const endElement = Math.min((chunkIndex + 1) * chunkSize, arraySize);
      const actualChunkSize = endElement - startElement;

      descriptors.push({
        chunkId: `${phase.phaseId}-chunk-${chunkIndex}`,
        chunkIndex,
        phase: phase.phaseId,
        parentId: plan.parentId,

        framework: 'webgpu',
        kernel: this.getBitonicSortShader(),
        entry: 'main',
        workgroupCount: [Math.ceil(actualChunkSize / 64), 1, 1],

        // Input is current state of the array (from globalState)
        inputData: this.extractChunkData(globalState.currentArray, startElement, actualChunkSize),
        outputSize: actualChunkSize * 4, // float32 array

        uniforms: {
          chunk_start: startElement,
          chunk_size: actualChunkSize,
          array_size: arraySize,

          // Bitonic sort phase parameters
          stage: phase.phaseParams.stage,
          step: phase.phaseParams.step,
          compare_distance: phase.phaseParams.compareDistance,
          block_size: phase.phaseParams.blockSize,
          ascending: phase.phaseParams.ascending ? 1 : 0
        },

        assemblyMetadata: {
          chunkIndex,
          startElement,
          chunkSize: actualChunkSize,
          phase: phase.phaseId
        }
      });
    }

    return descriptors;
  }

  extractChunkData(fullArray, startElement, chunkSize) {
    const chunkBuffer = Buffer.alloc(chunkSize * 4); // float32
    const chunkView = new Float32Array(chunkBuffer.buffer);

    for (let i = 0; i < chunkSize; i++) {
      chunkView[i] = fullArray[startElement + i] || 0;
    }

    return chunkBuffer.toString('base64');
  }

  getBitonicSortShader() {
    return `
      struct BitonicParams {
          chunk_start: u32,
          chunk_size: u32,
          array_size: u32,
          stage: u32,
          step: u32,
          compare_distance: u32,
          block_size: u32,
          ascending: u32,
      };

      @group(0) @binding(0) var<uniform> params: BitonicParams;
      @group(0) @binding(1) var<storage, read> input_chunk: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output_chunk: array<f32>;

      @compute @workgroup_size(64, 1, 1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let local_index = global_id.x;
          if (local_index >= params.chunk_size) {
              return;
          }

          let global_index = params.chunk_start + local_index;
          if (global_index >= params.array_size) {
              return;
          }

          // Bitonic sort logic
          let partner_index = global_index ^ params.compare_distance;

          // Check if partner is within our chunk
          if (partner_index >= params.chunk_start &&
              partner_index < params.chunk_start + params.chunk_size) {

              let my_value = input_chunk[local_index];
              let partner_local_index = partner_index - params.chunk_start;
              let partner_value = input_chunk[partner_local_index];

              // Determine sort direction for this block
              let block_id = global_index / params.block_size;
              let ascending = (block_id % 2u == 0u) == (params.ascending != 0u);

              // Compare and exchange
              let should_swap = (ascending && my_value > partner_value) ||
                               (!ascending && my_value < partner_value);

              if (should_swap) {
                  output_chunk[local_index] = partner_value;
              } else {
                  output_chunk[local_index] = my_value;
              }
          } else {
              // Partner is in different chunk, just copy value
              // The comparison will happen when the partner chunk processes this element
              output_chunk[local_index] = input_chunk[local_index];
          }
      }
    `;
  }

  // Update global state after each phase
  updateGlobalState(globalState, phaseResults, phase) {
    // Reconstruct the full array from chunk results
    const { arraySize } = phaseResults[0].metadata;
    const newArray = new Float32Array(arraySize);

    // Merge results from all chunks
    phaseResults.forEach(chunkResult => {
      const { startElement, chunkSize } = chunkResult.assemblyMetadata;
      const chunkData = new Float32Array(
        Buffer.from(chunkResult.result, 'base64').buffer
      );

      for (let i = 0; i < chunkSize; i++) {
        if (startElement + i < arraySize) {
          newArray[startElement + i] = chunkData[i];
        }
      }
    });

    return {
      ...globalState,
      currentArray: newArray,
      currentPhase: phase.phaseId,
      completedPhases: (globalState.completedPhases || 0) + 1
    };
  }

  shouldContinue(globalState, phase, plan) {
    return globalState.completedPhases < plan.totalPhases;
  }

  assembleFinalResult(globalState, plan) {
    return {
      success: true,
      data: Buffer.from(globalState.currentArray.buffer).toString('base64'),
      metadata: {
        type: 'sorted_array',
        algorithm: 'bitonic_sort',
        arraySize: plan.metadata.arraySize,
        totalPhases: globalState.completedPhases,
        format: 'float32'
      }
    };
  }
}


class IterativeChunkingManager extends EnhancedChunkingManager {
  async processIterativeWorkload(workload) {
    const strategy = this.registry.getStrategy(workload.chunkingStrategy);
    const plan = strategy.planExecution(workload);

    if (plan.executionModel !== 'iterative_refinement') {
      // Fall back to single-phase processing
      return await super.processChunkedWorkload(workload);
    }

    console.log(`Starting iterative workload ${workload.id} with ${plan.totalPhases} phases`);

    // Initialize global state
    let globalState = {
      currentArray: this.parseInitialArray(workload.input, plan.metadata),
      currentPhase: null,
      completedPhases: 0
    };

    // Execute each phase sequentially
    for (let phaseIndex = 0; phaseIndex < plan.phases.length; phaseIndex++) {
      const phase = plan.phases[phaseIndex];

      console.log(`Executing phase ${phaseIndex + 1}/${plan.phases.length}: ${phase.phaseId}`);

      // Create chunk descriptors for this phase
      const phaseChunks = strategy.createPhaseChunkDescriptors(plan, phase, globalState);

      // Execute all chunks in this phase in parallel
      const phaseResults = await this.executePhaseChunks(phaseChunks, workload.id);

      // Update global state with phase results
      globalState = strategy.updateGlobalState(globalState, phaseResults, phase);

      // Check if we should continue
      if (!strategy.shouldContinue(globalState, phase, plan)) {
        console.log(`Early termination after phase ${phase.phaseId}`);
        break;
      }

      // Broadcast phase completion to all clients (for monitoring)
      this.io.emit('phase:complete', {
        workloadId: workload.id,
        phaseId: phase.phaseId,
        progress: (phaseIndex + 1) / plan.phases.length
      });
    }

    // Assemble final result
    const finalResult = strategy.assembleFinalResult(globalState, plan);

    console.log(`✅ Iterative workload ${workload.id} completed after ${globalState.completedPhases} phases`);

    return {
      success: true,
      finalResult,
      stats: {
        totalPhases: globalState.completedPhases,
        algorithm: strategy.name,
        executionModel: 'iterative_refinement'
      }
    };
  }

  async executePhaseChunks(chunkDescriptors, workloadId) {
    return new Promise((resolve, reject) => {
      const results = new Map();
      const totalChunks = chunkDescriptors.length;
      let completedChunks = 0;

      // Set up completion handler for this phase
      const phaseCompletionHandler = (chunkId, result, processingTime) => {
        const chunkDesc = chunkDescriptors.find(desc => desc.chunkId === chunkId);
        if (!chunkDesc) return;

        results.set(chunkId, {
          chunkId,
          result,
          processingTime,
          assemblyMetadata: chunkDesc.assemblyMetadata,
          metadata: { arraySize: chunkDesc.uniforms.array_size }
        });

        completedChunks++;

        if (completedChunks === totalChunks) {
          // All chunks in this phase are complete
          const sortedResults = Array.from(results.values())
            .sort((a, b) => a.assemblyMetadata.chunkIndex - b.assemblyMetadata.chunkIndex);
          resolve(sortedResults);
        }
      };

      // Register phase completion handler
      this.phaseCompletionHandlers.set(workloadId, phaseCompletionHandler);

      // Dispatch all chunks for this phase
      chunkDescriptors.forEach(chunk => {
        this.dispatchChunkToAvailableClient(chunk);
      });

      // Set timeout for phase completion
      setTimeout(() => {
        if (completedChunks < totalChunks) {
          reject(new Error(`Phase timeout: only ${completedChunks}/${totalChunks} chunks completed`));
        }
      }, 5 * 60 * 1000); // 5 minute timeout per phase
    });
  }

  parseInitialArray(inputData, metadata) {
    const buffer = Buffer.from(inputData, 'base64');
    return new Float32Array(buffer.buffer);
  }

  // Enhanced chunk completion handler for iterative workloads
  handleIterativeChunkCompletion(workloadId, chunkId, result, processingTime) {
    const handler = this.phaseCompletionHandlers.get(workloadId);
    if (handler) {
      handler(chunkId, result, processingTime);
    }
  }
}

// ===== Usage Example =====

// Submit bitonic sort job
async function submitBitonicSort() {
  // Prepare test data: random float32 array (power of 2 size)
  const arraySize = 1024; // Must be power of 2
  const testArray = new Float32Array(arraySize);
  for (let i = 0; i < arraySize; i++) {
    testArray[i] = Math.random() * 1000;
  }

  const inputBase64 = Buffer.from(testArray.buffer).toString('base64');

  // Submit iterative bitonic sort
  const response = await fetch('/api/workloads/iterative', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      label: 'Bitonic Sort Test',
      chunkingStrategy: 'bitonic_sort',
      framework: 'webgpu',
      input: inputBase64,
      metadata: {
        arraySize: arraySize,
        chunkSize: 64, // Elements per chunk
        algorithm: 'bitonic_sort'
      }
    })
  });

  console.log('Bitonic sort submitted:', await response.json());
}
/*
node submit-task.mjs iterative-sort \
  --algorithm bitonic \
  --array-size 1024 \
  --chunk-size 64 \
  --input ./random_data.bin \
  --label "Bitonic Sort 1024 elements"
*/
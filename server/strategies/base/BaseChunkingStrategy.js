// COMPLETE: strategies/base/BaseChunkingStrategy.js - Enhanced base class with multi-input/output support

export class BaseChunkingStrategy {
  constructor(name) {
    this.name = name;
  }

  /**
   * Validate that the workload is compatible with this strategy
   * @param {Object} workload - The workload definition
   * @returns {Object} - { valid: boolean, error?: string }
   */
  validateWorkload(workload) {
    // NEW: Validate multi-input/output constraints
    const schema = this.defineInputSchema();

    if (workload.input) {
      const parsedInputs = this.parseMultipleInputs(workload.input, schema);
      const inputKeys = Object.keys(parsedInputs);

      // Check max inputs
      if (inputKeys.length > 4) {
        return {
          valid: false,
          error: `Maximum 4 inputs supported, got ${inputKeys.length}`
        };
      }

      // Check input names match schema
      const schemaInputNames = schema.inputs.map(inp => inp.name);
      for (const inputName of inputKeys) {
        if (!schemaInputNames.includes(inputName)) {
          return {
            valid: false,
            error: `Input '${inputName}' not defined in schema. Expected: ${schemaInputNames.join(', ')}`
          };
        }
      }
    }

    // Check output sizes if provided
    if (workload.outputSizes) {
      if (workload.outputSizes.length > 3) {
        return {
          valid: false,
          error: `Maximum 3 outputs supported, got ${workload.outputSizes.length}`
        };
      }

      if (workload.outputSizes.length !== schema.outputs.length) {
        return {
          valid: false,
          error: `Output sizes length (${workload.outputSizes.length}) doesn't match schema outputs (${schema.outputs.length})`
        };
      }
    }

    return { valid: true };
  }

  /**
   * Define the input schema this strategy expects
   * @returns {Object} - Schema defining inputs and outputs
   */
  defineInputSchema() {
    return {
      inputs: [
        {
          name: 'input',
          type: 'storage_buffer',
          binding: 0,
          elementType: 'f32',
          chunking: 'parallel' // 'parallel' for chunked, 'replicate' for full copy
        }
      ],
      outputs: [
        {
          name: 'output',
          type: 'storage_buffer',
          binding: 1,
          elementType: 'f32'
        }
      ]
    };
  }

  /**
   * Plan the execution of the workload
   * @param {Object} workload - The workload to process
   * @returns {Object} - Execution plan
   */
  planExecution(workload) {
    const schema = this.defineInputSchema();
    return {
      strategy: this.name,
      totalChunks: 1,
      schema: schema,
      metadata: {
        inputData: workload.input,
        outputSizes: workload.outputSizes || [workload.outputSize]
      },
      assemblyStrategy: this.name.replace('_chunking', '_assembly').replace('chunking', 'assembly')
    };
  }

  /**
   * Create chunk descriptors for standard single-phase execution
   * @param {Object} plan - The execution plan
   * @returns {Array} - Array of chunk descriptors
   */
  createChunkDescriptors(plan) {
    const schema = plan.schema;
    const parsedInputs = this.parseMultipleInputs(plan.metadata.inputData, schema);

    const descriptors = [];
    for (let chunkIndex = 0; chunkIndex < plan.totalChunks; chunkIndex++) {
      // Create inputs array for this chunk
      const inputChunks = this.chunkInputs(schema, parsedInputs, chunkIndex, plan.totalChunks, plan.metadata);

      // Compute output sizes for this chunk
      const outputSizes = this.computeChunkOutputSizes(schema, plan.metadata, chunkIndex, plan.totalChunks);

      descriptors.push({
        chunkId: `${this.name}-${chunkIndex}`,
        chunkIndex,
        parentId: plan.parentId,

        framework: 'webgpu',
        kernel: plan.metadata.customShader || this.getDefaultShader(),
        entry: 'main',
        workgroupCount: this.computeWorkgroupCount(chunkIndex, plan),

        inputs: inputChunks, // NEW: Array of base64 strings
        outputSizes: outputSizes, // NEW: Array of output sizes

        // Backward compatibility
        inputData: inputChunks[0] || '',
        outputSize: outputSizes[0] || 0,

        uniforms: this.computeChunkUniforms(chunkIndex, plan),

        assemblyMetadata: {
          chunkIndex,
          strategy: this.name
        }
      });
    }

    return descriptors;
  }

  /**
   * Create chunk descriptors for a specific phase (multi-phase algorithms)
   * @param {Object} plan - The execution plan
   * @param {Object} phase - The current phase
   * @param {Object} globalState - Current global state
   * @returns {Array} - Array of chunk descriptors for this phase
   */
  createPhaseChunkDescriptors(plan, phase, globalState) {
    // Default: fall back to single-phase behavior
    return this.createChunkDescriptors(plan);
  }

  /**
   * Update global state after phase completion (multi-phase algorithms)
   * @param {Object} globalState - Current global state
   * @param {Array} phaseResults - Results from completed phase
   * @param {Object} phase - The completed phase
   * @returns {Object} - Updated global state
   */
  updateGlobalState(globalState, phaseResults, phase) {
    return globalState;
  }

  /**
   * Check if execution should continue (multi-phase algorithms)
   * @param {Object} globalState - Current global state
   * @param {Object} phase - The completed phase
   * @param {Object} plan - The execution plan
   * @returns {boolean} - Whether to continue execution
   */
  shouldContinue(globalState, phase, plan) {
    return false;
  }

  /**
   * Assemble final result (multi-phase algorithms)
   * @param {Object} globalState - Final global state
   * @param {Object} plan - The execution plan
   * @returns {Object} - Final result
   */
  assembleFinalResult(globalState, plan) {
    return {
      success: true,
      data: globalState.finalData,
      metadata: { strategy: this.name }
    };
  }

  /**
   * Parse multiple inputs from workload input data
   * @param {string} inputData - Raw input data (base64 or JSON)
   * @param {Object} schema - Input schema
   * @returns {Object} - Parsed inputs by name
   */
  parseMultipleInputs(inputData, schema) {
    if (!inputData) return {};

    // Handle JSON format: {"input_a": "base64...", "input_b": "base64..."}
    if (typeof inputData === 'string' && inputData.startsWith('{')) {
      try {
        return JSON.parse(inputData);
      } catch (e) {
        throw new Error(`Invalid JSON input format: ${e.message}`);
      }
    }

    // Handle single input case - map to first input name
    if (schema.inputs.length === 1) {
      return { [schema.inputs[0].name]: inputData };
    }

    // Handle binary format with header (if needed in future)
    throw new Error('Binary multi-input format not yet implemented');
  }

  /**
   * Create input chunks for all inputs based on chunking strategy
   * @param {Object} schema - Input schema
   * @param {Object} parsedInputs - Parsed inputs by name
   * @param {number} chunkIndex - Index of this chunk
   * @param {number} totalChunks - Total number of chunks
   * @param {Object} chunkingParams - Strategy-specific parameters
   * @returns {Array} - Array of base64 strings (one per input)
   */
  chunkInputs(schema, parsedInputs, chunkIndex, totalChunks, chunkingParams = {}) {
    const inputChunks = [];

    for (const inputDef of schema.inputs) {
      const inputData = parsedInputs[inputDef.name];

      if (!inputData) {
        // Input not provided, add empty string
        inputChunks.push('');
        continue;
      }

      if (inputDef.chunking === 'parallel') {
        // Chunk this input
        const chunkedData = this.chunkInput(inputData, chunkIndex, totalChunks, chunkingParams);
        inputChunks.push(chunkedData);
      } else if (inputDef.chunking === 'replicate' || inputDef.chunking === 'none') {
        // Replicate full input to all chunks
        inputChunks.push(inputData);
      } else {
        // Default to parallel chunking
        const chunkedData = this.chunkInput(inputData, chunkIndex, totalChunks, chunkingParams);
        inputChunks.push(chunkedData);
      }
    }

    return inputChunks;
  }

  /**
   * Extract a chunk of data from an input based on chunking strategy
   * @param {string|Buffer} inputData - The input data
   * @param {number} chunkIndex - Index of this chunk
   * @param {number} totalChunks - Total number of chunks
   * @param {Object} chunkingParams - Strategy-specific parameters
   * @returns {string} - Base64 encoded chunk data
   */
  chunkInput(inputData, chunkIndex, totalChunks, chunkingParams = {}) {
    // Default: linear chunking
    const buffer = Buffer.from(inputData, 'base64');
    const chunkSize = Math.ceil(buffer.length / totalChunks);
    const start = chunkIndex * chunkSize;
    const end = Math.min(start + chunkSize, buffer.length);

    return buffer.slice(start, end).toString('base64');
  }

  /**
   * Compute output sizes for a specific chunk
   * @param {Object} schema - Output schema
   * @param {Object} metadata - Plan metadata
   * @param {number} chunkIndex - Index of this chunk
   * @param {number} totalChunks - Total number of chunks
   * @returns {Array} - Array of output sizes for this chunk
   */
  computeChunkOutputSizes(schema, metadata, chunkIndex, totalChunks) {
    // Default: use provided chunk output sizes or divide total by chunks
    const outputSizes = metadata.chunkOutputSizes || metadata.outputSizes;

    if (outputSizes) {
      return outputSizes;
    }

    // Fallback: divide total output size by number of chunks
    const totalOutputSizes = metadata.outputSizes || [metadata.outputSize || 0];
    return totalOutputSizes.map(size => Math.ceil(size / totalChunks));
  }

  /**
   * Compute workgroup count for a specific chunk
   * @param {number} chunkIndex - Index of this chunk
   * @param {Object} plan - Execution plan
   * @returns {Array} - [x, y, z] workgroup count
   */
  computeWorkgroupCount(chunkIndex, plan) {
    // Default workgroup count
    return [1, 1, 1];
  }

  /**
   * Compute uniforms for a specific chunk
   * @param {number} chunkIndex - Index of this chunk
   * @param {Object} plan - Execution plan
   * @returns {Object} - Uniform values
   */
  computeChunkUniforms(chunkIndex, plan) {
    return {
      chunkIndex: chunkIndex,
      totalChunks: plan.totalChunks
    };
  }

  /**
   * Get default shader for this strategy
   * @returns {string} - Shader code
   */
  getDefaultShader() {
    return `
      @compute @workgroup_size(64, 1, 1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        // Default shader - implement in subclass
      }
    `;
  }
}
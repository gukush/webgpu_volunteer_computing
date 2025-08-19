// strategies/base/BaseChunkingStrategy.js
// Base class for all chunking strategies

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
          chunking: 'parallel'
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
    throw new Error(`${this.name}: planExecution must be implemented by subclass`);
  }

  /**
   * Create chunk descriptors for standard single-phase execution
   * @param {Object} plan - The execution plan
   * @returns {Array} - Array of chunk descriptors
   */
  createChunkDescriptors(plan) {
    throw new Error(`${this.name}: createChunkDescriptors must be implemented by subclass`);
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
}
// strategies/EnhancedChunkingManager.js
// Main manager for enhanced chunking with pluggable strategies

import { ChunkingStrategyRegistry } from './ChunkingStrategyRegistry.js';
import { v4 as uuidv4 } from 'uuid';

export class EnhancedChunkingManager {
  constructor() {
    this.registry = new ChunkingStrategyRegistry();
    this.activeWorkloads = new Map();
    this.phaseCompletionHandlers = new Map();
  }

  /**
   * Process a chunked workload using pluggable strategies
   * @param {Object} workload - Enhanced workload definition
   * @returns {Object} - Processing result
   */
  async processChunkedWorkload(workload) {
    try {
      // Get the chunking strategy
      const chunkingStrategy = this.registry.getChunkingStrategy(workload.chunkingStrategy);
      if (!chunkingStrategy) {
        return {
          success: false,
          error: `Unknown chunking strategy: ${workload.chunkingStrategy}`
        };
      }

      // Validate the workload
      const validation = chunkingStrategy.validateWorkload(workload);
      if (!validation.valid) {
        return {
          success: false,
          error: validation.error
        };
      }

      // Plan the execution
      const plan = chunkingStrategy.planExecution(workload);
      plan.parentId = workload.id;

      // Check for multi-phase execution
      if (plan.executionModel === 'iterative_refinement') {
        return await this.processIterativeWorkload(workload, plan, chunkingStrategy);
      }

      // Create chunk descriptors for single-phase execution
      const chunkDescriptors = chunkingStrategy.createChunkDescriptors(plan);

      return {
        success: true,
        plan,
        chunkDescriptors,
        totalChunks: plan.totalChunks
      };

    } catch (error) {
      return {
        success: false,
        error: `Chunking failed: ${error.message}`
      };
    }
  }

  /**
   * Process an iterative workload with multiple phases
   * @param {Object} workload - Original workload
   * @param {Object} plan - Execution plan
   * @param {Object} strategy - Chunking strategy instance
   * @returns {Object} - Final result
   */
  async processIterativeWorkload(workload, plan, strategy) {
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

  /**
   * Execute all chunks in a phase and wait for completion
   * @param {Array} chunkDescriptors - Chunks to execute
   * @param {string} workloadId - Parent workload ID
   * @returns {Promise<Array>} - Phase results
   */
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
          metadata: { arraySize: chunkDesc.uniforms?.array_size }
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

      // NOTE: Actual chunk dispatching would be handled by the main server
      // This is a placeholder for the execution coordination

      // Set timeout for phase completion
      setTimeout(() => {
        if (completedChunks < totalChunks) {
          reject(new Error(`Phase timeout: only ${completedChunks}/${totalChunks} chunks completed`));
        }
      }, 5 * 60 * 1000); // 5 minute timeout per phase
    });
  }

  /**
   * Handle completion of a chunk (called by server)
   * @param {string} parentId - Parent workload ID
   * @param {string} chunkId - Chunk ID
   * @param {string} result - Chunk result
   * @param {number} processingTime - Processing time
   * @returns {Object} - Assembly result if complete
   */
  handleChunkCompletion(parentId, chunkId, result, processingTime) {
    // Handle iterative workload chunk completion
    const handler = this.phaseCompletionHandlers.get(parentId);
    if (handler) {
      handler(chunkId, result, processingTime);
      return { success: true, status: 'phase_in_progress' };
    }

    // Handle regular chunked workload completion
    return this.handleRegularChunkCompletion(parentId, chunkId, result, processingTime);
  }

  /**
   * Handle completion of a regular (non-iterative) chunk
   * @param {string} parentId - Parent workload ID
   * @param {string} chunkId - Chunk ID
   * @param {string} result - Chunk result
   * @param {number} processingTime - Processing time
   * @returns {Object} - Assembly result
   */
  handleRegularChunkCompletion(parentId, chunkId, result, processingTime) {
    // This would integrate with the existing chunk completion logic
    // For now, return a placeholder
    return { success: true, status: 'chunk_completed' };
  }

  /**
   * Register a custom strategy from code
   * @param {string} strategyCode - JavaScript strategy code
   * @param {string} type - 'chunking' or 'assembly'
   * @param {string} name - Strategy name
   * @returns {Object} - Registration result
   */
  registerCustomStrategy(strategyCode, type, name = null) {
    return this.registry.loadCustomStrategy(strategyCode, type, name);
  }

  /**
   * Parse initial array from input data
   * @param {string} inputData - Base64 input data
   * @param {Object} metadata - Workload metadata
   * @returns {Float32Array} - Parsed array
   */
  parseInitialArray(inputData, metadata) {
    const buffer = Buffer.from(inputData, 'base64');
    return new Float32Array(buffer.buffer);
  }

  /**
   * Get available strategies
   * @returns {Object} - Available strategies by type
   */
  getAvailableStrategies() {
    return this.registry.listStrategies();
  }
}
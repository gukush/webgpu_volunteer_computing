// strategies/EnhancedChunkingManager.js
// UPDATED: Main manager for enhanced chunking with pluggable strategies, multi-input/output support, and two-step processing

import { ChunkingStrategyRegistry } from './ChunkingStrategyRegistry.js';
import { v4 as uuidv4 } from 'uuid';

export class EnhancedChunkingManager {
  constructor() {
    this.registry = new ChunkingStrategyRegistry();
    this.activeWorkloads = new Map();
    this.phaseCompletionHandlers = new Map();
  }

  /**
   * PHASE 1: Validate workload and create execution plan (WITHOUT processing files)
   * This is called during workload creation before files are uploaded
   * @param {Object} workload - Enhanced workload definition
   * @returns {Object} - Validation and planning result
   */
  async validateAndPlanWorkload(workload) {
    try {
      // Get the chunking strategy
      const chunkingStrategy = this.registry.getChunkingStrategy(workload.chunkingStrategy);
      if (!chunkingStrategy) {
        return {
          success: false,
          error: `Unknown chunking strategy: ${workload.chunkingStrategy}`
        };
      }

      // Validate the workload (now includes multi-input/output validation)
      const validation = chunkingStrategy.validateWorkload(workload);
      if (!validation.valid) {
        return {
          success: false,
          error: validation.error
        };
      }

      // Plan the execution (should NOT read files, just validate metadata)
      const plan = chunkingStrategy.planExecution(workload);
      plan.parentId = workload.id;

      // NEW: Validate schema constraints
      const schemaValidation = this.validateSchema(plan.schema, workload);
      if (!schemaValidation.valid) {
        return {
          success: false,
          error: schemaValidation.error
        };
      }

      return {
        success: true,
        plan,
        requiresFileUpload: this.strategyRequiresFileUpload(chunkingStrategy, workload),
        message: 'Workload validated and planned successfully'
      };

    } catch (error) {
      return {
        success: false,
        error: `Workload validation failed: ${error.message}`
      };
    }
  }

  /**
   * PHASE 2: Process a chunked workload and create chunk descriptors (AFTER files uploaded)
   * This is called after files have been uploaded and we're ready to start computation
   * @param {Object} workload - Enhanced workload definition (with inputRefs populated)
   * @returns {Object} - Processing result with chunk descriptors
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

      // For strategies that need files, validate files are uploaded
      if (this.strategyRequiresFileUpload(chunkingStrategy, workload)) {
        if (!workload.inputRefs || workload.inputRefs.length === 0) {
          return {
            success: false,
            error: `Strategy '${workload.chunkingStrategy}' requires input files. Upload files first via POST /api/workloads/:id/inputs`
          };
        }
      }

      // Re-validate the workload with uploaded files
      const validation = chunkingStrategy.validateWorkload(workload);
      if (!validation.valid) {
        return {
          success: false,
          error: validation.error
        };
      }

      // Plan the execution (now with file access if needed)
      const plan = chunkingStrategy.planExecution(workload);
      plan.parentId = workload.id;

      // Check for multi-phase execution
      if (plan.executionModel === 'iterative_refinement') {
        return await this.processIterativeWorkload(workload, plan, chunkingStrategy);
      }

      // Create chunk descriptors - NOW with file access for two-step strategies
      let chunkDescriptors;

      if (typeof chunkingStrategy.createChunkDescriptors === 'function') {
        // Two-step strategy: use createChunkDescriptors with full plan including inputRefs
        const fullPlan = {
          ...plan,
          inputRefs: workload.inputRefs,
          metadata: { ...plan.metadata, ...workload.metadata }
        };
        chunkDescriptors = await chunkingStrategy.createChunkDescriptors(fullPlan);
      } else {
        // Legacy single-step strategy: plan execution includes chunk creation
        chunkDescriptors = plan.chunkDescriptors || [];
      }

      // NEW: Validate chunk descriptors for multi-input/output compatibility
      const descriptorValidation = this.validateChunkDescriptors(chunkDescriptors, plan.schema);
      if (!descriptorValidation.valid) {
        return {
          success: false,
          error: descriptorValidation.error
        };
      }

      // Register the workload as active for chunk completion tracking
      this.registerActiveWorkload(workload.id, plan, chunkDescriptors);

      return {
        success: true,
        plan,
        chunkDescriptors,
        totalChunks: chunkDescriptors.length
      };

    } catch (error) {
      return {
        success: false,
        error: `Chunking failed: ${error.message}`,
        stack: error.stack
      };
    }
  }

  /**
   * NEW: Check if a strategy requires file upload before processing
   * @param {Object} strategy - Chunking strategy instance
   * @param {Object} workload - Workload definition
   * @returns {boolean} - True if strategy needs files uploaded first
   */
  strategyRequiresFileUpload(strategy, workload) {
    // Check if strategy has two-step support
    if (typeof strategy.createChunkDescriptors === 'function') {
      // Two-step strategy - check if it needs file input
      if (strategy.name === 'block_matrix' ||
          strategy.name === 'image_tiled' ||
          strategy.name === 'file_processing') {
        return true;
      }
    }

    // Legacy strategies that process inline input don't need separate file upload
    if (workload.input && typeof workload.input === 'string') {
      return false;
    }

    // If no inline input provided, assume file upload is needed
    return !workload.input;
  }

  /**
   * NEW: Register an active workload for tracking
   * @param {string} workloadId - Workload ID
   * @param {Object} plan - Execution plan
   * @param {Array} chunkDescriptors - Chunk descriptors
   */
  registerActiveWorkload(workloadId, plan, chunkDescriptors) {
    this.activeWorkloads.set(workloadId, {
      id: workloadId,
      plan,
      chunkDescriptors,
      chunkingStrategy: plan.chunkingStrategy,
      assemblyStrategy: plan.assemblyStrategy,
      startedAt: Date.now(),
      completedChunks: new Map(),
      totalChunks: chunkDescriptors.length
    });

    console.log(`📋 Registered active workload ${workloadId} with ${chunkDescriptors.length} chunks`);
  }

  /**
   * NEW: Validate schema constraints for multi-input/output
   * @param {Object} schema - Input/output schema
   * @param {Object} workload - Original workload
   * @returns {Object} - Validation result
   */
  validateSchema(schema, workload) {
    if (!schema) {
      return { valid: true }; // Schema is optional
    }

    // Validate input limits
    if (schema.inputs && schema.inputs.length > 4) {
      return {
        valid: false,
        error: `Schema defines ${schema.inputs.length} inputs, maximum 4 supported`
      };
    }

    // Validate output limits
    if (schema.outputs && schema.outputs.length > 3) {
      return {
        valid: false,
        error: `Schema defines ${schema.outputs.length} outputs, maximum 3 supported`
      };
    }

    // Validate input names are unique
    if (schema.inputs) {
      const inputNames = schema.inputs.map(inp => inp.name);
      const uniqueNames = new Set(inputNames);
      if (inputNames.length !== uniqueNames.size) {
        return {
          valid: false,
          error: 'Schema input names must be unique'
        };
      }
    }

    // Validate output names are unique
    if (schema.outputs) {
      const outputNames = schema.outputs.map(out => out.name);
      const uniqueNames = new Set(outputNames);
      if (outputNames.length !== uniqueNames.size) {
        return {
          valid: false,
          error: 'Schema output names must be unique'
        };
      }
    }

    // Validate binding indices don't conflict
    if (schema.inputs && schema.outputs) {
      const allBindings = [
        ...schema.inputs.map(inp => inp.binding),
        ...schema.outputs.map(out => out.binding)
      ].filter(binding => binding !== undefined);

      const uniqueBindings = new Set(allBindings);
      if (allBindings.length !== uniqueBindings.size) {
        return {
          valid: false,
          error: 'Schema binding indices must be unique'
        };
      }
    }

    return { valid: true };
  }

  /**
   * NEW: Validate chunk descriptors for multi-input/output compatibility
   * @param {Array} chunkDescriptors - Generated chunk descriptors
   * @param {Object} schema - Input/output schema
   * @returns {Object} - Validation result
   */
  validateChunkDescriptors(chunkDescriptors, schema) {
    for (const descriptor of chunkDescriptors) {
      // Validate inputs array (legacy) or inputRefs (streaming)
      if (descriptor.inputs || descriptor.inputRefs) {
        if (!Array.isArray(descriptor.inputs)) {
          return {
            valid: false,
            error: `Chunk ${descriptor.chunkId} inputs must be an array`
          };
        }

        if (schema && schema.inputs && descriptor.inputs.length !== schema.inputs.length) {
          return {
            valid: false,
            error: `Chunk ${descriptor.chunkId} has ${descriptor.inputs.length} inputs, schema expects ${schema.inputs.length}`
          };
        }
      }

      // Validate output sizes array
      if (descriptor.outputSizes) {
        if (!Array.isArray(descriptor.outputSizes)) {
          return {
            valid: false,
            error: `Chunk ${descriptor.chunkId} outputSizes must be an array`
          };
        }

        if (schema && schema.outputs && descriptor.outputSizes.length !== schema.outputs.length) {
          return {
            valid: false,
            error: `Chunk ${descriptor.chunkId} has ${descriptor.outputSizes.length} output sizes, schema expects ${schema.outputs.length}`
          };
        }

        // Validate all output sizes are positive
        for (let i = 0; i < descriptor.outputSizes.length; i++) {
          if (!Number.isInteger(descriptor.outputSizes[i]) || descriptor.outputSizes[i] <= 0) {
            return {
              valid: false,
              error: `Chunk ${descriptor.chunkId} output size ${i} must be a positive integer`
            };
          }
        }
      }
    }

    return { valid: true };
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

      // NEW: Validate phase chunks for multi-input/output
      const phaseValidation = this.validateChunkDescriptors(phaseChunks, plan.schema);
      if (!phaseValidation.valid) {
        throw new Error(`Phase ${phase.phaseId} validation failed: ${phaseValidation.error}`);
      }

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
      const phaseCompletionHandler = (chunkId, results, processingTime) => {
        const chunkDesc = chunkDescriptors.find(desc => desc.chunkId === chunkId);
        if (!chunkDesc) return;

        // NEW: Handle multi-result format
        let finalResults = results;
        if (!Array.isArray(finalResults)) {
          finalResults = [finalResults];
        }

        results.set(chunkId, {
          chunkId,
          results: finalResults, // NEW: Always store as array
          result: finalResults[0], // Backward compatibility
          processingTime,
          assemblyMetadata: chunkDesc.assemblyMetadata,
          metadata: {
            arraySize: chunkDesc.uniforms?.array_size,
            outputCount: finalResults.length
          }
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
   * @param {string|Array} result - Chunk result(s)
   * @param {number} processingTime - Processing time
   * @returns {Object} - Assembly result if complete
   */
  handleChunkCompletion(parentId, chunkId, result, processingTime) {
    // NEW: Handle both single result and multi-result formats
    let results = result;
    if (!Array.isArray(results)) {
      results = [results];
    }

    // Handle iterative workload chunk completion
    const handler = this.phaseCompletionHandlers.get(parentId);
    if (handler) {
      handler(chunkId, results, processingTime);
      return { success: true, status: 'phase_in_progress' };
    }

    // Handle regular chunked workload completion
    return this.handleRegularChunkCompletion(parentId, chunkId, results, processingTime);
  }

  /**
   * ENHANCED: Handle completion of a regular (non-iterative) chunk with assembly coordination
   * @param {string} parentId - Parent workload ID
   * @param {string} chunkId - Chunk ID
   * @param {Array} results - Chunk results (array)
   * @param {number} processingTime - Processing time
   * @returns {Object} - Assembly result
   */
  handleRegularChunkCompletion(parentId, chunkId, results, processingTime) {
    const workload = this.activeWorkloads.get(parentId);
    if (!workload) {
      return {
        success: false,
        error: `No active workload found for ${parentId}`
      };
    }

    // Record completed chunk
    workload.completedChunks.set(chunkId, {
      chunkId,
      results,
      result: results[0], // Backward compatibility
      processingTime,
      completedAt: Date.now()
    });

    const completedCount = workload.completedChunks.size;
    const totalChunks = workload.totalChunks;

    console.log(`📊 Chunk progress for ${parentId}: ${completedCount}/${totalChunks} chunks (${results.length} outputs)`);

    // Check if all chunks are complete
    if (completedCount === totalChunks) {
      console.log(`🎯 All chunks completed for ${parentId}, ready for assembly`);

      return {
        success: true,
        status: 'complete',
        finalResult: {
          completedChunks: Array.from(workload.completedChunks.values()),
          plan: workload.plan
        },
        stats: {
          chunkingStrategy: workload.chunkingStrategy,
          assemblyStrategy: workload.assemblyStrategy,
          totalChunks,
          completedChunks: completedCount,
          totalProcessingTime: Array.from(workload.completedChunks.values())
            .reduce((sum, chunk) => sum + (chunk.processingTime || 0), 0)
        }
      };
    } else {
      return {
        success: true,
        status: 'in_progress',
        progress: (completedCount / totalChunks) * 100,
        outputCount: results.length
      };
    }
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
    // NEW: Handle multi-input format
    let actualInputData = inputData;

    if (typeof inputData === 'string' && inputData.startsWith('{')) {
      try {
        const parsedInputs = JSON.parse(inputData);
        // Use the first input for array parsing
        const firstInputKey = Object.keys(parsedInputs)[0];
        actualInputData = parsedInputs[firstInputKey];
      } catch (e) {
        console.warn('Failed to parse multi-input JSON, using as single input');
      }
    }

    const buffer = Buffer.from(actualInputData, 'base64');
    return new Float32Array(buffer.buffer);
  }

  /**
   * Get available strategies
   * @returns {Object} - Available strategies by type
   */
  getAvailableStrategies() {
    return this.registry.listStrategies();
  }

  /**
   * NEW: Get workload progress information
   * @param {string} workloadId - Workload ID
   * @returns {Object|null} - Progress information
   */
  getWorkloadProgress(workloadId) {
    const workload = this.activeWorkloads.get(workloadId);
    if (!workload) {
      return null;
    }

    const totalChunks = workload.totalChunks || 0;
    const completedChunks = workload.completedChunks.size;

    return {
      workloadId,
      totalChunks,
      completedChunks,
      progress: totalChunks > 0 ? (completedChunks / totalChunks) * 100 : 0,
      startedAt: workload.startedAt,
      chunkingStrategy: workload.chunkingStrategy,
      assemblyStrategy: workload.assemblyStrategy
    };
  }

  /**
   * NEW: Clean up completed workload
   * @param {string} workloadId - Workload ID
   * @returns {boolean} - True if workload was removed
   */
  cleanupWorkload(workloadId) {
    const removed = this.activeWorkloads.delete(workloadId);
    this.phaseCompletionHandlers.delete(workloadId);

    if (removed) {
      console.log(`🧹 Cleaned up workload ${workloadId}`);
    }

    return removed;
  }
}
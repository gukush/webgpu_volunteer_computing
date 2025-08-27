// ENHANCED: EnhancedChunkingManager.js - Complete version with streaming support + file-output opts
import { ChunkingStrategyRegistry } from './ChunkingStrategyRegistry.js';
import { v4 as uuidv4 } from 'uuid';
import { info } from '../logger.js';

const __DEBUG_ON__ = (process.env.LOG_LEVEL || '').toLowerCase() === 'debug';

export class EnhancedChunkingManager {
  constructor() {
    this.registry = new ChunkingStrategyRegistry();
    this.activeWorkloads = new Map();
    this.streamingAssemblers = new Map(); // workloadId -> assembler instance
    this.dispatchCallbacks = new Map(); // workloadId -> dispatch function
    this.phaseCompletionHandlers = new Map();
  }

  // -------- NEW: shared helpers for file-output aware assembly strategies --------

  /**
   * Build assembly options from workload + plan metadata.
   * Mirrors BaseAssemblyStrategy constructor opts.
   */
  buildAssemblyOptions(workload, plan) {
    const md = {
      ...(plan?.metadata || {}),
      ...(workload?.metadata || {})
    };

    return {
      workloadId: workload?.id,
      metadata: md,
      storageRoot: process.env.VOLUNTEER_STORAGE,
      outputMode: md.outputMode,                       // 'file' | 'memory'
      outputPath: md.outputPath,                       // dir or file path
      outputFilename: md.outputFilename ?? 'final.bin',
      splitOutputsAsFiles: !!md.splitOutputsAsFiles,
      suppressInMemoryOutputs: !!md.suppressInMemoryOutputs
    };
  }

  /**
   * Apply assembly options onto a strategy instance (non-breaking).
   * Supports multiple conventions, then falls back to setting _opts.
   */
  _applyAssemblyOptions(strategy, workload, plan) {
    if (!strategy) return;
    const opts = this.buildAssemblyOptions(workload, plan);

    if (typeof strategy.configure === 'function') {
      strategy.configure(opts);
      return;
    }
    if (typeof strategy.setOptions === 'function') {
      strategy.setOptions(opts);
      return;
    }
    if (typeof strategy.applyOptions === 'function') {
      strategy.applyOptions(opts);
      return;
    }

    // Fallback: merge into private _opts bag used by BaseAssemblyStrategy extension
    try {
      strategy._opts = { ...(strategy._opts || {}), ...opts };
    } catch {
      /* no-op */
    }
  }

  /**
   * PHASE 1: Validate workload and create execution plan (WITHOUT processing files)
   */
  async validateAndPlanWorkload(workload) {
    try {
      const chunkingStrategy = this.registry.getChunkingStrategy(workload.chunkingStrategy);
      if (!chunkingStrategy) {
        return {
          success: false,
          error: `Unknown chunking strategy: ${workload.chunkingStrategy}`
        };
      }

      const validation = chunkingStrategy.validateWorkload(workload);
      if (!validation.valid) {
        return {
          success: false,
          error: validation.error
        };
      }

      const plan = chunkingStrategy.planExecution(workload);
      plan.parentId = workload.id;

      const schemaValidation = this.validateSchema(plan.schema, workload);
      if (!schemaValidation.valid) {
        return {
          success: false,
          error: schemaValidation.error
        };
      }

      // Include assembly options early so later phases (batch/stream) can access them
      plan.metadata = { ...(plan.metadata || {}), ...(workload.metadata || {}) };
      plan.assemblyOptions = this.buildAssemblyOptions(workload, plan);

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
   * ENHANCED: PHASE 2 - Process workload with streaming support
   */
  async processChunkedWorkload(workload, streamingMode = false) {
    try {
      const chunkingStrategy = this.registry.getChunkingStrategy(workload.chunkingStrategy);
      if (!chunkingStrategy) {
        return {
          success: false,
          error: `Unknown chunking strategy: ${workload.chunkingStrategy}`
        };
      }

      // Validate files are uploaded if needed
      if (this.strategyRequiresFileUpload(chunkingStrategy, workload)) {
        if (!workload.inputRefs || workload.inputRefs.length === 0) {
          return {
            success: false,
            error: `Strategy '${workload.chunkingStrategy}' requires input files. Upload files first via POST /api/workloads/:id/inputs`
          };
        }
      }

      const validation = chunkingStrategy.validateWorkload(workload);
      if (!validation.valid) {
        return {
          success: false,
          error: validation.error
        };
      }

      const plan = chunkingStrategy.planExecution(workload);
      plan.parentId = workload.id;
      plan.framework = workload.framework;
      plan.metadata = {
        ...plan.metadata,
        ...workload.metadata,
        framework: workload.framework
      };

      // attach assembly options here as well
      plan.assemblyOptions = this.buildAssemblyOptions(workload, plan);

      if (__DEBUG_ON__) console.log(`[CHUNKING MANAGER DEBUG] Plan framework: ${plan.framework}`);
      if (__DEBUG_ON__) console.log(`[CHUNKING MANAGER DEBUG] Workload framework: ${workload.framework}`);
      if (__DEBUG_ON__) console.log(`[STREAMING] Processing workload ${workload.id} in ${streamingMode ? 'streaming' : 'batch'} mode`);

      // Check for multi-phase execution
      if (plan.executionModel === 'iterative_refinement') {
        return await this.processIterativeWorkload(workload, plan, chunkingStrategy);
      }

      // Build full plan used by dispatch/assembly
      const fullPlan = {
        ...plan,
        inputRefs: workload.inputRefs,
        metadata: { ...plan.metadata, ...workload.metadata },
        assemblyOptions: plan.assemblyOptions
      };

      // Initialize streaming assembly if supported
      let assembler = null;
      if (streamingMode && this.supportsStreamingAssembly(workload.assemblyStrategy)) {
        if (__DEBUG_ON__) console.log(` Initializing streaming assembly for ${workload.id} with strategy: ${workload.assemblyStrategy}`);
        try {
          assembler = await this.initializeStreamingAssembly(workload, fullPlan);
          this.streamingAssemblers.set(workload.id, assembler);
          info.bind(null, 'STREAMING')(`Streaming assembler initialized for ${workload.id}`);
        } catch (assemblyError) {
          console.error(` Failed to initialize streaming assembler:`, assemblyError);
          return {
            success: false,
            error: `Failed to initialize streaming assembly: ${assemblyError.message}`
          };
        }
      } else if (streamingMode) {
        console.warn(`️ Streaming mode requested but assembly strategy '${workload.assemblyStrategy}' doesn't support streaming`);
      }

      // Register the workload for tracking
      this.registerActiveWorkload(workload.id, fullPlan, [], streamingMode);

      if (streamingMode && typeof chunkingStrategy.createChunkDescriptorsStreaming === 'function') {
        // STREAMING MODE: Create and dispatch chunks on-demand
        const dispatchCallback = this.createDispatchCallback(workload.id);
        const result = await chunkingStrategy.createChunkDescriptorsStreaming(fullPlan, dispatchCallback);

        if (__DEBUG_ON__) console.log(` Streaming chunk creation started for ${workload.id}`);

        return {
          success: true,
          plan: fullPlan,
          totalChunks: result.totalChunks,
          streamingMode: true,
          message: `Streaming mode: chunks being created and dispatched dynamically`
        };
      } else {
        // BATCH MODE: Create all chunks upfront
        let chunkDescriptors;
        if (typeof chunkingStrategy.createChunkDescriptors === 'function') {
          chunkDescriptors = await chunkingStrategy.createChunkDescriptors(fullPlan);
        } else {
          chunkDescriptors = plan.chunkDescriptors || [];
        }

        const descriptorValidation = this.validateChunkDescriptors(chunkDescriptors, plan.schema);
        if (!descriptorValidation.valid) {
          return {
            success: false,
            error: descriptorValidation.error
          };
        }

        // Update workload registration with actual chunks
        this.updateActiveWorkload(workload.id, chunkDescriptors);

        return {
          success: true,
          plan: fullPlan,
          chunkDescriptors,
          totalChunks: chunkDescriptors.length,
          streamingMode: false
        };
      }

    } catch (error) {
      return {
        success: false,
        error: `Chunking failed: ${error.message}`,
        stack: error.stack
      };
    }
  }

  /**
   * Initialize streaming assembly
   */
  async initializeStreamingAssembly(workload, plan) {
    const assemblyStrategyName = plan.assemblyStrategy || workload.assemblyStrategy;
    let assemblyStrategy = this.registry.getAssemblyStrategy(assemblyStrategyName);

    if (!assemblyStrategy) {
      throw new Error(`Assembly strategy '${assemblyStrategyName}' not found`);
    }

    // The registry may return a class or an instance; support both
    if (typeof assemblyStrategy === 'function') {
      // Assume constructor signature (name, opts) to support BaseAssemblyStrategy extension
      try {
        assemblyStrategy = new assemblyStrategy(assemblyStrategyName, this.buildAssemblyOptions(workload, plan));
      } catch {
        // Fallback: parameterless
        assemblyStrategy = new assemblyStrategy();
        this._applyAssemblyOptions(assemblyStrategy, workload, plan);
      }
    } else {
      // Instance path: apply opts directly
      this._applyAssemblyOptions(assemblyStrategy, workload, plan);
    }

    if (__DEBUG_ON__) console.log(` Initializing streaming assembly with ${assemblyStrategyName}`);

    // Initialize the assembly strategy's output store
    if (typeof assemblyStrategy.initOutputStore === 'function') {
      await assemblyStrategy.initOutputStore(plan);
    }

    // Set up streaming callbacks
    if (typeof assemblyStrategy.onBlockComplete === 'function') {
      assemblyStrategy.onBlockComplete(async (progress) => {
        this.emitAssemblyProgress(workload.id, progress);
      });
    }

    if (typeof assemblyStrategy.onAssemblyComplete === 'function') {
      assemblyStrategy.onAssemblyComplete(async (result) => {
        await this.handleStreamingAssemblyComplete(workload.id, result);
      });
    }

    return assemblyStrategy;
  }

  /**
   * Create dispatch callback for streaming chunk creation
   */
  createDispatchCallback(workloadId) {
    return async (chunkDescriptor) => {
      // Validate the chunk
      if (!chunkDescriptor || !chunkDescriptor.chunkId) {
        throw new Error('Invalid chunk descriptor');
      }

      if (__DEBUG_ON__) console.log(` Dispatching chunk ${chunkDescriptor.chunkId} for workload ${workloadId}`);

      // Add to active workload tracking
      const workload = this.activeWorkloads.get(workloadId);
      if (workload) {
        if (!workload.dispatchedChunks) {
          workload.dispatchedChunks = new Map();
        }
        workload.dispatchedChunks.set(chunkDescriptor.chunkId, {
          descriptor: chunkDescriptor,
          dispatchedAt: Date.now(),
          status: 'dispatched'
        });
      }

      // Call external dispatch function if available
      const externalDispatch = this.dispatchCallbacks.get(workloadId);
      if (externalDispatch) {
        try {
          await externalDispatch(chunkDescriptor);
        } catch (error) {
          console.error(`Failed to dispatch chunk ${chunkDescriptor.chunkId}:`, error);
          throw error;
        }
      }

      return { success: true };
    };
  }

  /**
   * Register external dispatch callback (called by server)
   */
  setDispatchCallback(workloadId, callback) {
    this.dispatchCallbacks.set(workloadId, callback);
  }

  /**
   * Check if strategy supports streaming assembly
   * (works whether registry returns a class or an instance)
   */
  supportsStreamingAssembly(assemblyStrategyName) {
    const s = this.registry.getAssemblyStrategy(assemblyStrategyName);
    const ref = typeof s === 'function' ? s.prototype : s;
    return ref && typeof ref.processChunkResult === 'function';
  }

  /**
   * ENHANCED: Handle chunk completion with streaming assembly support
   */
  async handleChunkCompletion(parentId, chunkId, result, processingTime) {
    let results = result;
    if (!Array.isArray(results)) {
      results = [results];
    }

    console.log(`[CHUNKING MANAGER] Processing chunk completion: ${chunkId} for workload ${parentId}`);
    // Handle iterative workload chunk completion
    const handler = this.phaseCompletionHandlers.get(parentId);
    if (handler) {
      handler(chunkId, results, processingTime);
      return { success: true, status: 'phase_in_progress' };
    }

    // Check if this workload uses streaming assembly
    const assembler = this.streamingAssemblers.get(parentId);
    if (assembler && typeof assembler.processChunkResult === 'function') {
      return this.handleStreamingChunkCompletion(parentId, chunkId, results, processingTime, assembler);
    }

    // Handle regular chunked workload completion
    return this.handleRegularChunkCompletion(parentId, chunkId, results, processingTime);
  }

  /**
   * Handle chunk completion with streaming assembly
   */
  async handleStreamingChunkCompletion(parentId, chunkId, results, processingTime, assembler) {
    try {
      const workload = this.activeWorkloads.get(parentId);
      if (!workload) {
        return {
          success: false,
          error: `No active workload found for ${parentId}`
        };
      }

      // Find the chunk descriptor to get assembly metadata
      let chunkDescriptor = null;
      if (workload.dispatchedChunks) {
        const dispatched = workload.dispatchedChunks.get(chunkId);
        chunkDescriptor = dispatched?.descriptor;
      }

      if (!chunkDescriptor) {
        // Fallback: try to find in chunkDescriptors if available
        chunkDescriptor = workload.chunkDescriptors?.find(cd => cd.chunkId === chunkId);
      }

      if (!chunkDescriptor) {
        console.error(`[STREAMING] Chunk descriptor not found for ${chunkId}`);
        return {
          success: false,
          error: `Chunk descriptor not found for ${chunkId}`
        };
      }

      // Create chunk result object
      const chunkResult = {
        chunkId,
        results,
        result: results[0], // Backward compatibility
        processingTime,
        completedAt: Date.now(),
        assemblyMetadata: chunkDescriptor.assemblyMetadata
      };

      if (__DEBUG_ON__) console.log(` Processing streaming chunk result: ${chunkId}`);

      // Process through streaming assembler
      const assemblyResult = await assembler.processChunkResult(chunkResult);

      if (assemblyResult.success && assemblyResult.complete) {
        info.bind(null, 'STREAMING')(`Streaming assembly completed for workload ${parentId}!`);

        // Clean up
        this.streamingAssemblers.delete(parentId);
        this.activeWorkloads.delete(parentId);

        return {
          success: true,
          status: 'complete',
          finalResult: assemblyResult.result
        };
      } else if (assemblyResult.success) {
        console.log(` Streaming assembly progress: ${assemblyResult.progress.toFixed(1)}% for ${parentId}`);

        return {
          success: true,
          status: 'in_progress',
          progress: assemblyResult.progress
        };
      } else {
        return {
          success: false,
          error: assemblyResult.error || 'Streaming assembly failed'
        };
      }

    } catch (error) {
      console.error(`[STREAMING] Error processing chunk ${chunkId}:`, error);
      return {
        success: false,
        error: `Streaming assembly error: ${error.message}`
      };
    }
  }

  /**
   * Handle streaming assembly completion
   */
  async handleStreamingAssemblyComplete(workloadId, result) {
    info.bind(null, 'STREAMING')(`Streaming assembly completed for workload ${workloadId}`);

    // Emit completion event (this would be handled by server)
    if (typeof this.onWorkloadComplete === 'function') {
      await this.onWorkloadComplete(workloadId, result);
    }

    // Clean up resources
    const assembler = this.streamingAssemblers.get(workloadId);
    if (assembler && typeof assembler.cleanup === 'function') {
      await assembler.cleanup();
    }

    this.streamingAssemblers.delete(workloadId);
  }

  /**
   * Emit assembly progress events
   */
  emitAssemblyProgress(workloadId, progress) {
    if (__DEBUG_ON__) console.log(` Assembly progress for ${workloadId}: ${progress.completedBlocks}/${progress.totalBlocks} blocks (${progress.progress.toFixed(1)}%)`);

    // This would be handled by the server to emit to clients
    if (typeof this.onAssemblyProgress === 'function') {
      this.onAssemblyProgress(workloadId, progress);
    }
  }

  /**
   * Register callbacks for server integration
   */
  setServerCallbacks({ onWorkloadComplete, onAssemblyProgress }) {
    if (onWorkloadComplete) this.onWorkloadComplete = onWorkloadComplete;
    if (onAssemblyProgress) this.onAssemblyProgress = onAssemblyProgress;
  }

  /**
   * ENHANCED: Register workload with streaming support
   */
  registerActiveWorkload(workloadId, plan, chunkDescriptors, streamingMode = false) {
    const workloadInfo = {
      id: workloadId,
      plan,
      chunkDescriptors: streamingMode ? [] : chunkDescriptors, // Empty for streaming mode
      chunkingStrategy: plan.chunkingStrategy,
      assemblyStrategy: plan.assemblyStrategy,
      startedAt: Date.now(),
      completedChunks: new Map(),
      totalChunks: streamingMode ? 0 : chunkDescriptors.length, // Will be updated in streaming mode
      streamingMode,
      // Include assembly options for later (batch) finalization paths
      assemblyOptions: plan.assemblyOptions
    };

    if (streamingMode) {
      workloadInfo.dispatchedChunks = new Map();
      workloadInfo.expectedTotalChunks = plan.totalChunks; // From planning phase
    }

    this.activeWorkloads.set(workloadId, workloadInfo);
    if (__DEBUG_ON__) console.log(` Registered ${streamingMode ? 'streaming' : 'batch'} workload ${workloadId}`);
  }

  /**
   * Update workload with dispatched chunk info (for batch mode)
   */
  updateActiveWorkload(workloadId, chunkDescriptors) {
    const workload = this.activeWorkloads.get(workloadId);
    if (workload && !workload.streamingMode) {
      workload.chunkDescriptors = chunkDescriptors;
      workload.totalChunks = chunkDescriptors.length;
    }
  }

  /**
   * NEW: Check if a strategy requires file upload before processing
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
      
      // ECM strategy generates all data internally - no file upload needed
      if (strategy.name === 'ecm_stage1') {
        return false;
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
   * NEW: Validate schema constraints for multi-input/output
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
   */
  async processIterativeWorkload(workload, plan, strategy) {
    if (__DEBUG_ON__) console.log(`Starting iterative workload ${workload.id} with ${plan.totalPhases} phases`);

    // Initialize global state
    let globalState = {
      currentArray: this.parseInitialArray(workload.input, plan.metadata),
      currentPhase: null,
      completedPhases: 0
    };

    // Execute each phase sequentially
    for (let phaseIndex = 0; phaseIndex < plan.phases.length; phaseIndex++) {
      const phase = plan.phases[phaseIndex];

      if (__DEBUG_ON__) console.log(`Executing phase ${phaseIndex + 1}/${plan.phases.length}: ${phase.phaseId}`);

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
        if (__DEBUG_ON__) console.log(`Early termination after phase ${phase.phaseId}`);
        break;
      }
    }

    // Assemble final result
    const finalResult = strategy.assembleFinalResult(globalState, plan);

    if (__DEBUG_ON__) console.log(` Iterative workload ${workload.id} completed after ${globalState.completedPhases} phases`);

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
   */
  async executePhaseChunks(chunkDescriptors, workloadId) {
    return new Promise((resolve, reject) => {
      const results = new Map();
      const totalChunks = chunkDescriptors.length;
      let completedChunks = 0;

      // Set up completion handler for this phase
      const phaseCompletionHandler = (chunkId, resultsArr, processingTime) => {
        const chunkDesc = chunkDescriptors.find(desc => desc.chunkId === chunkId);
        if (!chunkDesc) return;

        // NEW: Handle multi-result format
        let finalResults = resultsArr;
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
   * ENHANCED: Handle completion of a regular (non-iterative) chunk with assembly coordination
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
    const totalChunks = workload.streamingMode ? workload.expectedTotalChunks : workload.totalChunks;

    if (__DEBUG_ON__) console.log(` Chunk progress for ${parentId}: ${completedCount}/${totalChunks} chunks (${results.length} outputs)`);

    // Check if all chunks are complete
    if (completedCount === totalChunks) {
      if (__DEBUG_ON__) console.log(` All chunks completed for ${parentId}, ready for assembly`);

      // Include assemblyOptions so caller assembling later can honor file-output prefs
      return {
        success: true,
        status: 'complete',
        finalResult: {
          completedChunks: Array.from(workload.completedChunks.values()),
          plan: workload.plan,
          assemblyOptions: workload.assemblyOptions || workload.plan?.assemblyOptions
        },
        stats: {
          chunkingStrategy: workload.chunkingStrategy,
          assemblyStrategy: workload.assemblyStrategy,
          totalChunks,
          completedChunks: completedCount,
          streamingMode: workload.streamingMode,
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
   */
  registerCustomStrategy(strategyCode, type, name = null) {
    return this.registry.loadCustomStrategy(strategyCode, type, name);
  }

  /**
   * Parse initial array from input data
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
   */
  getAvailableStrategies() {
    return this.registry.listStrategies();
  }

  /**
   * NEW: Get workload progress information
   */
  getWorkloadProgress(workloadId) {
    const workload = this.activeWorkloads.get(workloadId);
    if (!workload) {
      return null;
    }

    const totalChunks = workload.streamingMode ? workload.expectedTotalChunks : workload.totalChunks;
    const completedChunks = workload.completedChunks.size;
    const dispatchedChunks = workload.dispatchedChunks ? workload.dispatchedChunks.size : 0;

    return {
      workloadId,
      totalChunks,
      completedChunks,
      dispatchedChunks,
      progress: totalChunks > 0 ? (completedChunks / totalChunks) * 100 : 0,
      startedAt: workload.startedAt,
      chunkingStrategy: workload.chunkingStrategy,
      assemblyStrategy: workload.assemblyStrategy,
      streamingMode: workload.streamingMode
    };
  }

  /**
   * NEW: Clean up completed workload
   */
  cleanupWorkload(workloadId) {
    const removed = this.activeWorkloads.delete(workloadId);
    this.phaseCompletionHandlers.delete(workloadId);
    this.dispatchCallbacks.delete(workloadId);

    // Clean up streaming assembler
    const assembler = this.streamingAssemblers.get(workloadId);
    if (assembler && typeof assembler.cleanup === 'function') {
      assembler.cleanup().catch(err =>
        console.warn(`Cleanup warning for ${workloadId}:`, err.message)
      );
    }
    this.streamingAssemblers.delete(workloadId);

    if (removed) {
      if (__DEBUG_ON__) console.log(` Cleaned up workload ${workloadId}`);
    }

    return removed;
  }
}

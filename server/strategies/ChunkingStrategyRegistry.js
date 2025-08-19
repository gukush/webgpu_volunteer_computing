// strategies/ChunkingStrategyRegistry.js
// Registry for managing all chunking and assembly strategies

import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';
import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';
import vm from 'vm';
import { fileURLToPath } from 'url';
import path from 'path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export class ChunkingStrategyRegistry {
  constructor() {
    this.chunkingStrategies = new Map();
    this.assemblyStrategies = new Map();
    this.shaderTemplates = new Map();

    // Initialize built-in strategies
    this.initializeBuiltInStrategies();
  }

  /**
   * Register a chunking strategy
   * @param {BaseChunkingStrategy} strategy - Strategy instance
   */
  registerChunkingStrategy(strategy) {
    if (!(strategy instanceof BaseChunkingStrategy)) {
      throw new Error('Strategy must extend BaseChunkingStrategy');
    }

    this.chunkingStrategies.set(strategy.name, strategy);
    console.log(`Registered chunking strategy: ${strategy.name}`);
  }

  /**
   * Register an assembly strategy
   * @param {BaseAssemblyStrategy} strategy - Strategy instance
   */
  registerAssemblyStrategy(strategy) {
    if (!(strategy instanceof BaseAssemblyStrategy)) {
      throw new Error('Strategy must extend BaseAssemblyStrategy');
    }

    this.assemblyStrategies.set(strategy.name, strategy);
    console.log(`Registered assembly strategy: ${strategy.name}`);
  }

  /**
   * Register a shader template
   * @param {string} name - Template name
   * @param {string} shaderCode - Shader source code
   */
  registerShaderTemplate(name, shaderCode) {
    this.shaderTemplates.set(name, shaderCode);
    console.log(`Registered shader template: ${name}`);
  }

  /**
   * Get a chunking strategy by name
   * @param {string} name - Strategy name
   * @returns {BaseChunkingStrategy|null}
   */
  getChunkingStrategy(name) {
    return this.chunkingStrategies.get(name) || null;
  }

  /**
   * Get an assembly strategy by name
   * @param {string} name - Strategy name
   * @returns {BaseAssemblyStrategy|null}
   */
  getAssemblyStrategy(name) {
    return this.assemblyStrategies.get(name) || null;
  }

  /**
   * Get a shader template by name
   * @param {string} name - Template name
   * @returns {string|null}
   */
  getShaderTemplate(name) {
    return this.shaderTemplates.get(name) || null;
  }

  /**
   * List all available strategies
   * @returns {Object} - Object with arrays of strategy names
   */
  listStrategies() {
    return {
      chunking: Array.from(this.chunkingStrategies.keys()),
      assembly: Array.from(this.assemblyStrategies.keys()),
      shaders: Array.from(this.shaderTemplates.keys())
    };
  }

  /**
   * Load and register a custom strategy from JavaScript code
   * @param {string} strategyCode - JavaScript code containing strategy class
   * @param {string} type - 'chunking' or 'assembly'
   * @param {string} expectedName - Expected strategy name for validation
   * @returns {Object} - { success: boolean, error?: string, strategyName?: string }
   */
  loadCustomStrategy(strategyCode, type, expectedName = null) {
    try {
      // Create a sandbox environment with required base classes
      const sandbox = {
        BaseChunkingStrategy,
        BaseAssemblyStrategy,
        Buffer,
        console: {
          log: (...args) => console.log('[CustomStrategy]', ...args),
          error: (...args) => console.error('[CustomStrategy]', ...args)
        },
        exports: {},
        module: { exports: {} }
      };

      // Create VM context
      const context = vm.createContext(sandbox);

      // Execute the strategy code
      vm.runInContext(strategyCode, context, {
        filename: `custom_${type}_strategy.js`,
        timeout: 5000 // 5 second timeout for strategy loading
      });

      // Extract the strategy class (try different export patterns)
      let StrategyClass = context.module.exports.default ||
                         context.module.exports ||
                         context.exports.default ||
                         context.exports;

      // If it's still an object, look for a class inside it
      if (typeof StrategyClass === 'object' && StrategyClass.constructor === Object) {
        const keys = Object.keys(StrategyClass);
        if (keys.length === 1) {
          StrategyClass = StrategyClass[keys[0]];
        }
      }

      if (typeof StrategyClass !== 'function') {
        return {
          success: false,
          error: 'Strategy code must export a class (use export default class MyStrategy...)'
        };
      }

      // Instantiate the strategy
      const strategyInstance = new StrategyClass();

      // Validate the strategy type
      if (type === 'chunking' && !(strategyInstance instanceof BaseChunkingStrategy)) {
        return {
          success: false,
          error: 'Chunking strategy must extend BaseChunkingStrategy'
        };
      }

      if (type === 'assembly' && !(strategyInstance instanceof BaseAssemblyStrategy)) {
        return {
          success: false,
          error: 'Assembly strategy must extend BaseAssemblyStrategy'
        };
      }

      // Validate strategy name if provided
      if (expectedName && strategyInstance.name !== expectedName) {
        return {
          success: false,
          error: `Strategy name mismatch: expected "${expectedName}", got "${strategyInstance.name}"`
        };
      }

      // Register the strategy
      if (type === 'chunking') {
        this.registerChunkingStrategy(strategyInstance);
      } else {
        this.registerAssemblyStrategy(strategyInstance);
      }

      return {
        success: true,
        strategyName: strategyInstance.name
      };

    } catch (error) {
      return {
        success: false,
        error: `Failed to load strategy: ${error.message}`
      };
    }
  }

  /**
   * Initialize built-in strategies
   */
  initializeBuiltInStrategies() {
    // Linear chunking strategy
    this.registerChunkingStrategy(new LinearChunkingStrategy());
    this.registerAssemblyStrategy(new LinearAssemblyStrategy());

    // Register built-in shader templates
    this.registerShaderTemplate('linear_process', this.getLinearProcessShader());
  }

  /**
   * Get linear processing shader template
   * @returns {string} - WGSL shader code
   */
  getLinearProcessShader() {
    return `
      struct LinearParams {
          start_element: u32,
          element_count: u32,
          total_elements: u32,
      };

      @group(0) @binding(0) var<uniform> params: LinearParams;
      @group(0) @binding(1) var<storage, read> input_data: array<f32>;
      @group(0) @binding(2) var<storage, read_write> output_data: array<f32>;

      @compute @workgroup_size(64, 1, 1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let local_index = global_id.x;
          if (local_index >= params.element_count) {
              return;
          }

          let global_index = params.start_element + local_index;
          if (global_index >= params.total_elements) {
              return;
          }

          // Example: square the input
          output_data[local_index] = input_data[local_index] * input_data[local_index];
      }
    `;
  }
}

// Built-in Linear Chunking Strategy
class LinearChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('linear');
  }

  planExecution(workload) {
    const { elementSize = 4, chunkSize = 1024 } = workload.metadata;
    const inputBuffer = Buffer.from(workload.input, 'base64');
    const totalElements = inputBuffer.length / elementSize;
    const totalChunks = Math.ceil(totalElements / chunkSize);

    return {
      strategy: this.name,
      totalChunks,
      metadata: {
        elementSize,
        chunkSize,
        totalElements,
        inputBuffer
      },
      assemblyStrategy: 'linear_assembly',
      shaderTemplate: workload.metadata.shaderTemplate || 'linear_process'
    };
  }

  createChunkDescriptors(plan) {
    const { elementSize, chunkSize, totalElements, inputBuffer } = plan.metadata;
    const descriptors = [];

    for (let chunkIndex = 0; chunkIndex < plan.totalChunks; chunkIndex++) {
      const startElement = chunkIndex * chunkSize;
      const endElement = Math.min((chunkIndex + 1) * chunkSize, totalElements);
      const actualChunkSize = endElement - startElement;

      const chunkData = inputBuffer.slice(
        startElement * elementSize,
        endElement * elementSize
      );

      descriptors.push({
        chunkId: `linear-${chunkIndex}`,
        chunkIndex,
        parentId: plan.parentId,

        framework: 'webgpu',
        kernel: plan.customShader || this.getDefaultShader(),
        entry: 'main',
        workgroupCount: [Math.ceil(actualChunkSize / 64), 1, 1],

        inputData: chunkData.toString('base64'),
        outputSize: actualChunkSize * elementSize,

        uniforms: {
          start_element: startElement,
          element_count: actualChunkSize,
          total_elements: totalElements
        },

        assemblyMetadata: {
          chunkIndex,
          startElement,
          elementCount: actualChunkSize
        }
      });
    }

    return descriptors;
  }

  getDefaultShader() {
    // Return the linear processing shader template
    return new ChunkingStrategyRegistry().getLinearProcessShader();
  }
}

// Built-in Linear Assembly Strategy
class LinearAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('linear_assembly');
  }

  assembleResults(completedChunks, plan) {
    const validation = this.validateChunks(completedChunks, plan);
    if (!validation.valid) {
      return {
        success: false,
        error: validation.error,
        missing: validation.missing
      };
    }

    try {
      const sortedChunks = this.sortChunks(completedChunks);
      const result = this.concatenateResults(sortedChunks);

      return {
        success: true,
        data: result.toString('base64'),
        metadata: this.createAssemblyMetadata(plan, sortedChunks)
      };
    } catch (error) {
      return {
        success: false,
        error: `Assembly failed: ${error.message}`
      };
    }
  }
}
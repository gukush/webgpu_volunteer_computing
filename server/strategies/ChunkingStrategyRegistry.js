// strategies/ChunkingStrategyRegistry.js
// Registry for managing all chunking and assembly strategies with multi-input/output support

import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';
import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';


import BlockMatrixChunkingStrategy from './BlockMatrixChunkingStrategy.js';
import BlockMatrixAssemblyStrategy from './BlockMatrixAssemblyStrategy.js';

import DistributedSortChunkingStrategy from './DistributedSortChunkingStrategy.js';
import DistributedSortAssemblyStrategy from './DistributedSortAssemblyStrategy.js';

import DistributedConvolutionChunkingStrategy from './DistributedConvolutionChunkingStrategy.js';
import DistributedConvolutionAssemblyStrategy from './DistributedConvolutionAssemblyStrategy.js';


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
    this.streamingCapabilities = new Map();
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
    const capabilities = this.analyzeChunkingCapabilities(strategy);
    this.streamingCapabilities.set(strategy.name, capabilities);
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
    const capabilities = this.analyzeAssemblyCapabilities(strategy);
    this.streamingCapabilities.set(strategy.name, capabilities);

    console.log(`Registered assembly strategy: ${strategy.name}`);
  }

  analyzeChunkingCapabilities(strategy) {
    return {
      hasStreaming: typeof strategy.createChunkDescriptorsStreaming === 'function',
      hasBatch: typeof strategy.createChunkDescriptors === 'function',
      hasMemoryPlanning: typeof strategy.planMemoryStrategy === 'function',
      hasInputValidation: typeof strategy.validateInputs === 'function',
      hasSchemaDefinition: typeof strategy.defineInputSchema === 'function',
      supportsMultiInput: this.checkMultiInputSupport(strategy),
      supportedFrameworks: this.getSupportedFrameworks(strategy)
    };
  }
  analyzeAssemblyCapabilities(strategy) {
    return {
      hasStreaming: typeof strategy.processChunkResult === 'function',
      hasBatch: typeof strategy.assembleResults === 'function',
      hasMemoryMgmt: typeof strategy.initOutputStore === 'function',
      hasProgress: typeof strategy.onBlockComplete === 'function',
      hasCleanup: typeof strategy.cleanup === 'function',
      supportsMultiOutput: this.checkMultiOutputSupport(strategy)
    };
  }

  checkMultiInputSupport(strategy) {
    if (typeof strategy.defineInputSchema !== 'function') {
      return false;
    }
    try {
      const schema = strategy.defineInputSchema();
      return schema.inputs && schema.inputs.length > 1;
    } catch (e) {
      return false;
    }
  }

  checkMultiOutputSupport(strategy) {
    if (typeof strategy.getDefaultSchema !== 'function') {
      return false;
    }
    try {
      const schema = strategy.getDefaultSchema();
      return schema.outputs && schema.outputs.length > 1;
    } catch (e) {
      return false;
    }
  }

  getSupportedFrameworks(strategy) {
    const frameworks = ['webgpu']; // Default

    // Check for framework-specific methods
    if (typeof strategy.getWebGLVertexShader === 'function') frameworks.push('webgl');
    if (typeof strategy.getCUDAKernel === 'function') frameworks.push('cuda');
    if (typeof strategy.getOpenCLKernel === 'function') frameworks.push('opencl');
    if (typeof strategy.getVulkanShader === 'function') frameworks.push('vulkan');
    if (typeof strategy.getJavaScriptKernel === 'function') frameworks.push('javascript');

    return frameworks;
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
    const chunkingList = Array.from(this.chunkingStrategies.keys()).map(name => {
      const capabilities = this.streamingCapabilities.get(name) || {};
      return {
        name,
        ...capabilities
      };
    });

    const assemblyList = Array.from(this.assemblyStrategies.keys()).map(name => {
      const capabilities = this.streamingCapabilities.get(name) || {};
      return {
        name,
        ...capabilities
      };
    });

    return {
      chunking: chunkingList,
      assembly: assemblyList,
      shaders: Array.from(this.shaderTemplates.keys()),
      streamingSupport: {
        totalStrategies: this.chunkingStrategies.size + this.assemblyStrategies.size,
        streamingEnabled: chunkingList.filter(s => s.hasStreaming).length + assemblyList.filter(s => s.hasStreaming).length
      }
    };
  }
 getStreamingCapableStrategies() {
    const streamingChunking = Array.from(this.chunkingStrategies.keys())
      .filter(name => {
        const caps = this.streamingCapabilities.get(name);
        return caps && caps.hasStreaming;
      });

    const streamingAssembly = Array.from(this.assemblyStrategies.keys())
      .filter(name => {
        const caps = this.streamingCapabilities.get(name);
        return caps && caps.hasStreaming;
      });

    return {
      chunking: streamingChunking,
      assembly: streamingAssembly
    };
  }
   getFrameworkSupportedStrategies(framework) {
    const supported = [];

    for (const [name, capabilities] of this.streamingCapabilities.entries()) {
      if (capabilities.supportedFrameworks && capabilities.supportedFrameworks.includes(framework)) {
        supported.push({
          name,
          type: this.chunkingStrategies.has(name) ? 'chunking' : 'assembly',
          capabilities
        });
      }
    }

    return supported;
  }
   validateStreamingCompatibility(chunkingStrategy, assemblyStrategy, framework) {
    const chunkingCaps = this.streamingCapabilities.get(chunkingStrategy);
    const assemblyCaps = this.streamingCapabilities.get(assemblyStrategy);

    const issues = [];

    // Check if chunking strategy exists and supports streaming
    if (!chunkingCaps) {
      issues.push(`Chunking strategy '${chunkingStrategy}' not found`);
    } else if (!chunkingCaps.hasStreaming) {
      issues.push(`Chunking strategy '${chunkingStrategy}' does not support streaming`);
    }

    // Check if assembly strategy exists and supports streaming
    if (!assemblyCaps) {
      issues.push(`Assembly strategy '${assemblyStrategy}' not found`);
    } else if (!assemblyCaps.hasStreaming) {
      issues.push(`Assembly strategy '${assemblyStrategy}' does not support streaming`);
    }

    // Check framework support
    if (chunkingCaps && !chunkingCaps.supportedFrameworks.includes(framework)) {
      issues.push(`Chunking strategy '${chunkingStrategy}' does not support framework '${framework}'`);
    }

    return {
      compatible: issues.length === 0,
      issues,
      recommendations: this.generateCompatibilityRecommendations(chunkingStrategy, assemblyStrategy, framework, issues)
    };
  }
generateCompatibilityRecommendations(chunkingStrategy, assemblyStrategy, framework, issues) {
    const recommendations = [];

    if (issues.some(i => i.includes('does not support streaming'))) {
      const streamingStrategies = this.getStreamingCapableStrategies();
      if (streamingStrategies.chunking.length > 0) {
        recommendations.push(`Consider using streaming-capable chunking strategies: ${streamingStrategies.chunking.join(', ')}`);
      }
      if (streamingStrategies.assembly.length > 0) {
        recommendations.push(`Consider using streaming-capable assembly strategies: ${streamingStrategies.assembly.join(', ')}`);
      }
    }

    if (issues.some(i => i.includes('does not support framework'))) {
      const frameworkStrategies = this.getFrameworkSupportedStrategies(framework);
      if (frameworkStrategies.length > 0) {
        recommendations.push(`Strategies supporting ${framework}: ${frameworkStrategies.map(s => s.name).join(', ')}`);
      } else {
        recommendations.push(`No strategies support ${framework}. Try 'webgpu' as fallback.`);
      }
    }

    return recommendations;
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
      // Your existing VM context creation (keep as-is)
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

      const context = vm.createContext(sandbox);
      vm.runInContext(strategyCode, context, {
        filename: `custom_${type}_strategy.js`,
        timeout: 5000
      });

      // Your existing strategy class extraction (keep as-is)
      let StrategyClass = context.module.exports.default ||
                         context.module.exports ||
                         context.exports.default ||
                         context.exports;

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

      const strategyInstance = new StrategyClass();

      // Your existing validation (keep as-is)
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

      if (expectedName && strategyInstance.name !== expectedName) {
        return {
          success: false,
          error: `Strategy name mismatch: expected "${expectedName}", got "${strategyInstance.name}"`
        };
      }

      // Register using your existing methods (they now include capability detection)
      if (type === 'chunking') {
        this.registerChunkingStrategy(strategyInstance);
      } else {
        this.registerAssemblyStrategy(strategyInstance);
      }

      // NEW: Return enhanced info including streaming support
      const capabilities = this.streamingCapabilities.get(strategyInstance.name);
      return {
        success: true,
        strategyName: strategyInstance.name,
        capabilities,
        isStreamingCapable: type === 'chunking' ? capabilities.hasStreaming : capabilities.hasStreaming,
        supportedFrameworks: capabilities.supportedFrameworks || ['webgpu']
      };

    } catch (error) {
      return {
        success: false,
        error: `Failed to load strategy: ${error.message}`
      };
    }
  }
  analyzeStreamingReadiness() {
    const analysis = {
      ready: [],
      needsUpgrade: [],
      recommendations: []
    };

    for (const [name, strategy] of this.chunkingStrategies.entries()) {
      const caps = this.streamingCapabilities.get(name);
      if (caps.hasStreaming && caps.hasBatch) {
        analysis.ready.push({ name, type: 'chunking', capabilities: caps });
      } else {
        analysis.needsUpgrade.push({
          name,
          type: 'chunking',
          missing: [
            !caps.hasStreaming ? 'createChunkDescriptorsStreaming method' : null,
            !caps.hasMemoryPlanning ? 'planMemoryStrategy method' : null,
            !caps.hasSchemaDefinition ? 'defineInputSchema method' : null
          ].filter(Boolean)
        });
      }
    }

    for (const [name, strategy] of this.assemblyStrategies.entries()) {
      const caps = this.streamingCapabilities.get(name);
      if (caps.hasStreaming && caps.hasBatch) {
        analysis.ready.push({ name, type: 'assembly', capabilities: caps });
      } else {
        analysis.needsUpgrade.push({
          name,
          type: 'assembly',
          missing: [
            !caps.hasStreaming ? 'processChunkResult method' : null,
            !caps.hasMemoryMgmt ? 'initOutputStore method' : null,
            !caps.hasProgress ? 'onBlockComplete callback support' : null
          ].filter(Boolean)
        });
      }
    }

    // Generate recommendations
    if (analysis.needsUpgrade.length > 0) {
      analysis.recommendations.push('Consider upgrading strategies to support streaming for better performance');
      analysis.recommendations.push('Use block_matrix strategy as a reference for streaming implementation');
    }

    return analysis;
  }
  /**
   * UPDATED: Initialize built-in strategies including matrix tiled
   */
  initializeBuiltInStrategies() {
    // Linear chunking strategy
    this.registerChunkingStrategy(new LinearChunkingStrategy());
    this.registerAssemblyStrategy(new LinearAssemblyStrategy());

    this.registerChunkingStrategy(new DistributedSortChunkingStrategy());
    this.registerAssemblyStrategy(new DistributedSortAssemblyStrategy());

    this.registerChunkingStrategy(new DistributedConvolutionChunkingStrategy());
    this.registerAssemblyStrategy(new DistributedConvolutionAssemblyStrategy());

    this.registerChunkingStrategy(new BlockMatrixChunkingStrategy());
    this.registerAssemblyStrategy(new BlockMatrixAssemblyStrategy());
    // Register built-in shader templates
    this.registerShaderTemplate('linear_process', this.getLinearProcessShader());
    this.registerShaderTemplate('multi_buffer_process', this.getMultiBufferProcessShader());

    // NEW: Matrix-specific shader templates
    this.registerShaderTemplate('matrix_multiply_tiled', this.getMatrixMultiplyTiledShader());
    this.registerShaderTemplate('matrix_tile_extract', this.getMatrixTileExtractShader());

    console.log(`Initialized ${this.chunkingStrategies.size} chunking strategies and ${this.assemblyStrategies.size} assembly strategies`);
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

  /**
   * Get multi-buffer processing shader template
   * @returns {string} - WGSL shader code for multi-input/output
   */
  getMultiBufferProcessShader() {
    return `
      struct MultiBufferParams {
          element_count: u32,
          operation_type: u32,
      };

      @group(0) @binding(0) var<uniform> params: MultiBufferParams;
      @group(0) @binding(1) var<storage, read> input_a: array<f32>;
      @group(0) @binding(2) var<storage, read> input_b: array<f32>;
      @group(0) @binding(3) var<storage, read_write> output_sum: array<f32>;
      @group(0) @binding(4) var<storage, read_write> output_product: array<f32>;

      @compute @workgroup_size(64, 1, 1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let index = global_id.x;
          if (index >= params.element_count) {
              return;
          }

          let a = input_a[index];
          let b = input_b[index];

          // Output 0: sum
          output_sum[index] = a + b;

          // Output 1: product
          output_product[index] = a * b;
      }
    `;
  }

  /**
   * NEW: Get matrix multiply tiled shader template
   * @returns {string} - WGSL shader code for tiled matrix multiplication
   */
  getMatrixMultiplyTiledShader() {
    return `
      struct TileParams {
          matrix_n: u32,
          tile_start_row: u32,
          tile_start_col: u32,
          tile_rows: u32,
          tile_cols: u32,
          tile_size: u32,
      }

      @group(0) @binding(0) var<uniform> params: TileParams;
      @group(0) @binding(1) var<storage, read> input_data: array<f32>;
      @group(0) @binding(2) var<storage, read_write> tile_output: array<f32>;

      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let n = params.matrix_n;
          let local_row = global_id.x;
          let local_col = global_id.y;

          if (local_row >= params.tile_rows || local_col >= params.tile_cols) {
              return;
          }

          let global_row = params.tile_start_row + local_row;
          let global_col = params.tile_start_col + local_col;

          if (global_row >= n || global_col >= n) {
              return;
          }

          // Input layout from server: [matrix_size_header, A_data..., B_data...]
          let header_size = 1u;
          let a_offset = header_size;
          let b_offset = header_size + n * n;

          var sum = 0.0;
          for (var k = 0u; k < n; k = k + 1u) {
              let a_val = input_data[a_offset + global_row * n + k];
              let b_val = input_data[b_offset + k * n + global_col];
              sum = sum + a_val * b_val;
          }

          let output_index = local_row * params.tile_cols + local_col;
          tile_output[output_index] = sum;
      }
    `;
  }

  /**
   * NEW: Get matrix tile extraction shader template
   * @returns {string} - WGSL shader code for extracting matrix tiles
   */
  getMatrixTileExtractShader() {
    return `
      struct TileParams {
          matrix_n: u32,
          tile_start_row: u32,
          tile_start_col: u32,
          tile_rows: u32,
          tile_cols: u32,
          tile_size: u32,
      }

      @group(0) @binding(0) var<uniform> params: TileParams;
      @group(0) @binding(1) var<storage, read> input_data: array<f32>;
      @group(0) @binding(2) var<storage, read_write> tile_output: array<f32>;

      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let local_row = global_id.x;
          let local_col = global_id.y;

          if (local_row >= params.tile_rows || local_col >= params.tile_cols) {
              return;
          }

          let global_row = params.tile_start_row + local_row;
          let global_col = params.tile_start_col + local_col;

          if (global_row >= params.matrix_n || global_col >= params.matrix_n) {
              return;
          }

          // Simple tile extraction from matrix
          let input_index = global_row * params.matrix_n + global_col;
          let output_index = local_row * params.tile_cols + local_col;

          tile_output[output_index] = input_data[input_index];
      }
    `;
  }
}

// Built-in Linear Chunking Strategy (Updated for Multi-Input/Output)
class LinearChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('linear');
  }

  /**
   * Define schema for linear processing
   */
  defineInputSchema() {
    return {
      inputs: [
        {
          name: 'input',
          type: 'storage_buffer',
          binding: 1,
          elementType: 'f32',
          chunking: 'parallel'
        }
      ],
      outputs: [
        {
          name: 'output',
          type: 'storage_buffer',
          binding: 2,
          elementType: 'f32'
        }
      ]
    };
  }

  planExecution(workload) {
    const { elementSize = 4, chunkSize = 1024 } = workload.metadata;
    const schema = this.defineInputSchema();
    const parsedInputs = this.parseMultipleInputs(workload.input, schema);

    // Calculate based on first input
    const firstInputKey = Object.keys(parsedInputs)[0];
    const inputBuffer = firstInputKey ? Buffer.from(parsedInputs[firstInputKey], 'base64') : Buffer.alloc(0);
    const totalElements = inputBuffer.length / elementSize;
    const totalChunks = Math.ceil(totalElements / chunkSize);

    return {
      strategy: this.name,
      totalChunks,
      schema: schema,
      metadata: {
        elementSize,
        chunkSize,
        totalElements,
        inputData: workload.input,
        outputSizes: workload.outputSizes || [totalElements * elementSize]
      },
      assemblyStrategy: 'linear_assembly',
      shaderTemplate: workload.metadata.shaderTemplate || 'linear_process'
    };
  }

  createChunkDescriptors(plan) {
    const { elementSize, chunkSize, totalElements } = plan.metadata;
    const schema = plan.schema;
    const parsedInputs = this.parseMultipleInputs(plan.metadata.inputData, schema);
    const descriptors = [];

    for (let chunkIndex = 0; chunkIndex < plan.totalChunks; chunkIndex++) {
      const startElement = chunkIndex * chunkSize;
      const endElement = Math.min((chunkIndex + 1) * chunkSize, totalElements);
      const actualChunkSize = endElement - startElement;

      // Create input chunks
      const inputChunks = this.chunkInputs(schema, parsedInputs, chunkIndex, plan.totalChunks, plan.metadata);

      // Compute output sizes
      const outputSizes = this.computeChunkOutputSizes(schema, plan.metadata, chunkIndex, plan.totalChunks);

      descriptors.push({
        chunkId: `linear-${chunkIndex}`,
        chunkIndex,
        parentId: plan.parentId,

        framework: 'webgpu',
        kernel: plan.customShader || this.getDefaultShader(),
        entry: 'main',
        workgroupCount: [Math.ceil(actualChunkSize / 64), 1, 1],

        // Multi-input/output support
        inputs: inputChunks,
        outputSizes: outputSizes,

        // Backward compatibility
        inputData: inputChunks[0] || '',
        outputSize: outputSizes[0] || 0,

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

  /**
   * Linear chunking: divide input data evenly
   */
  chunkInput(inputData, chunkIndex, totalChunks, chunkingParams = {}) {
    const { elementSize = 4 } = chunkingParams;
    const buffer = Buffer.from(inputData, 'base64');
    const totalElements = buffer.length / elementSize;
    const elementsPerChunk = Math.ceil(totalElements / totalChunks);

    const startElement = chunkIndex * elementsPerChunk;
    const endElement = Math.min((chunkIndex + 1) * elementsPerChunk, totalElements);

    const startByte = startElement * elementSize;
    const endByte = endElement * elementSize;

    return buffer.slice(startByte, endByte).toString('base64');
  }

  getDefaultShader() {
    return new ChunkingStrategyRegistry().getLinearProcessShader();
  }
}

// Built-in Linear Assembly Strategy (Updated for Multi-Output)
class LinearAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() {
    super('linear_assembly');
  }

  /**
   * Get default schema for linear assembly
   */
  getDefaultSchema() {
    return {
      outputs: [
        {
          name: 'output',
          type: 'storage_buffer',
          elementType: 'f32'
        }
      ]
    };
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
      const schema = plan.schema || this.getDefaultSchema();
      const sortedChunks = this.sortChunks(completedChunks);

      // Handle both single and multi-output
      if (schema.outputs.length > 1) {
        return this.assembleMultipleOutputs(sortedChunks, plan, schema);
      } else {
        return this.assembleSingleOutput(sortedChunks, plan, schema);
      }
    } catch (error) {
      return {
        success: false,
        error: `Linear assembly failed: ${error.message}`
      };
    }
  }

  /**
   * Assembly single output by concatenation
   */
  assembleSingleOutputBuffers(buffers, plan, outputDef) {
    return Buffer.concat(buffers);
  }
}
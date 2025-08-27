// strategies/ChunkingStrategyRegistry.js
// Registry for managing all chunking and assembly strategies with multi-input/output support

import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';
import { BaseAssemblyStrategy } from './base/BaseAssemblyStrategy.js';

// Built-ins already present in your project
import BlockMatrixChunkingStrategy from './BlockMatrixChunkingStrategy.js';
import BlockMatrixAssemblyStrategy from './BlockMatrixAssemblyStrategy.js';

import SimpleNeuralNetworkChunkingStrategy from './SimpleNeuralNetworkChunkingStrategy.js';
import SimpleNeuralNetworkAssemblyStrategy from './SimpleNeuralNetworkAssemblyStrategy.js';

import DistributedSortChunkingStrategy from './DistributedSortChunkingStrategy.js';
import DistributedSortAssemblyStrategy from './DistributedSortAssemblyStrategy.js';

import DistributedConvolutionChunkingStrategy from './DistributedConvolutionChunkingStrategy.js';
import DistributedConvolutionAssemblyStrategy from './DistributedConvolutionAssemblyStrategy.js';

import ECMStage1ChunkingStrategy from './ECMStage1ChunkingStrategy.js';
import ECMStage1AssemblyStrategy from './ECMStage1AssemblyStrategy.js';

// NEW: Transformer strategies (commented out until implemented)
// import AttentionChunkingStrategy from './AttentionChunkingStrategy.js';
// import AttentionAssemblyStrategy from './AttentionAssemblyStrategy.js';
// import FeedForwardChunkingStrategy from './FeedForwardChunkingStrategy.js';
// import FeedForwardAssemblyStrategy from './FeedForwardAssemblyStrategy.js';
// import LayerNormChunkingStrategy from './LayerNormChunkingStrategy.js';
// import LayerNormAssemblyStrategy from './LayerNormAssemblyStrategy.js';

import vm from 'vm';
import { fileURLToPath } from 'url';
import path from 'path';

const __DEBUG_ON__ = (process.env.LOG_LEVEL || '').toLowerCase() === 'debug';
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
    if (__DEBUG_ON__) console.log(`Registered chunking strategy: ${strategy.name}`, capabilities);
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
    if (__DEBUG_ON__) console.log(`Registered assembly strategy: ${strategy.name}`, capabilities);
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
    if (typeof strategy.defineInputSchema !== 'function') return false;
    try {
      const schema = strategy.defineInputSchema();
      return !!(schema && Array.isArray(schema.inputs) && schema.inputs.length > 1);
    } catch {
      return false;
    }
  }

  checkMultiOutputSupport(strategy) {
    if (typeof strategy.getDefaultSchema !== 'function') return false;
    try {
      const schema = strategy.getDefaultSchema();
      return !!(schema && Array.isArray(schema.outputs) && schema.outputs.length > 1);
    } catch {
      return false;
    }
  }

  getSupportedFrameworks(strategy) {
    const frameworks = ['webgpu']; // default assumption
    if (typeof strategy.getWebGLVertexShader === 'function') frameworks.push('webgl');
    if (typeof strategy.getCUDAKernel === 'function') frameworks.push('cuda');
    if (typeof strategy.getOpenCLKernel === 'function') frameworks.push('opencl');
    if (typeof strategy.getVulkanShader === 'function') frameworks.push('vulkan');
    if (typeof strategy.getJavaScriptKernel === 'function') frameworks.push('javascript');
    return frameworks;
  }

  /**
   * Register a shader template
   */
  registerShaderTemplate(name, shaderCode) {
    this.shaderTemplates.set(name, shaderCode);
    if (__DEBUG_ON__) console.log(`Registered shader template: ${name}`);
  }

  getChunkingStrategy(name) { return this.chunkingStrategies.get(name) || null; }
  getAssemblyStrategy(name) { return this.assemblyStrategies.get(name) || null; }
  getShaderTemplate(name) { return this.shaderTemplates.get(name) || null; }

  listStrategies() {
    const chunkingList = Array.from(this.chunkingStrategies.keys()).map(name => {
      const capabilities = this.streamingCapabilities.get(name) || {};
      return { name, ...capabilities };
    });
    const assemblyList = Array.from(this.assemblyStrategies.keys()).map(name => {
      const capabilities = this.streamingCapabilities.get(name) || {};
      return { name, ...capabilities };
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
      .filter(name => (this.streamingCapabilities.get(name) || {}).hasStreaming);
    const streamingAssembly = Array.from(this.assemblyStrategies.keys())
      .filter(name => (this.streamingCapabilities.get(name) || {}).hasStreaming);
    return { chunking: streamingChunking, assembly: streamingAssembly };
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
    if (!chunkingCaps) issues.push(`Chunking strategy '${chunkingStrategy}' not found`);
    else if (!chunkingCaps.hasStreaming) issues.push(`Chunking strategy '${chunkingStrategy}' does not support streaming`);
    if (!assemblyCaps) issues.push(`Assembly strategy '${assemblyStrategy}' not found`);
    else if (!assemblyCaps.hasStreaming) issues.push(`Assembly strategy '${assemblyStrategy}' does not support streaming`);
    if (chunkingCaps && framework && !chunkingCaps.supportedFrameworks.includes(framework)) {
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
      const s = this.getStreamingCapableStrategies();
      if (s.chunking.length) recommendations.push(`Consider streaming-capable chunking strategies: ${s.chunking.join(', ')}`);
      if (s.assembly.length) recommendations.push(`Consider streaming-capable assembly strategies: ${s.assembly.join(', ')}`);
    }
    if (issues.some(i => i.includes('does not support framework'))) {
      const f = this.getFrameworkSupportedStrategies(framework);
      if (f.length) recommendations.push(`Strategies supporting ${framework}: ${f.map(x => x.name).join(', ')}`);
      else recommendations.push(`No strategies support ${framework}. Try 'webgpu' as fallback.`);
    }
    return recommendations;
  }

  loadCustomStrategy(strategyCode, type, expectedName = null) {
    try {
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
      vm.runInContext(strategyCode, context, { filename: `custom_${type}_strategy.js`, timeout: 5000 });
      let StrategyClass = context.module.exports.default ||
                          context.module.exports ||
                          context.exports.default ||
                          context.exports;
      if (typeof StrategyClass === 'object' && StrategyClass.constructor === Object) {
        const keys = Object.keys(StrategyClass);
        if (keys.length === 1) StrategyClass = StrategyClass[keys[0]];
      }
      if (typeof StrategyClass !== 'function') {
        return { success: false, error: 'Strategy code must export a class (use export default class MyStrategy...)' };
      }
      const instance = new StrategyClass();
      if (type === 'chunking' && !(instance instanceof BaseChunkingStrategy)) {
        return { success: false, error: 'Chunking strategy must extend BaseChunkingStrategy' };
      }
      if (type === 'assembly' && !(instance instanceof BaseAssemblyStrategy)) {
        return { success: false, error: 'Assembly strategy must extend BaseAssemblyStrategy' };
      }
      if (expectedName && instance.name !== expectedName) {
        return { success: false, error: `Strategy name mismatch: expected "${expectedName}", got "${instance.name}"` };
      }
      if (type === 'chunking') this.registerChunkingStrategy(instance);
      else this.registerAssemblyStrategy(instance);
      const capabilities = this.streamingCapabilities.get(instance.name);
      return {
        success: true,
        strategyName: instance.name,
        capabilities,
        isStreamingCapable: capabilities.hasStreaming,
        supportedFrameworks: capabilities.supportedFrameworks || ['webgpu']
      };
    } catch (error) {
      return { success: false, error: `Failed to load strategy: ${error.message}` };
    }
  }

  analyzeStreamingReadiness() {
    const analysis = { ready: [], needsUpgrade: [], recommendations: [] };
    for (const [name] of this.chunkingStrategies.entries()) {
      const caps = this.streamingCapabilities.get(name);
      if (caps.hasStreaming && caps.hasBatch) {
        analysis.ready.push({ name, type: 'chunking', capabilities: caps });
      } else {
        analysis.needsUpgrade.push({
          name, type: 'chunking',
          missing: [
            !caps.hasStreaming ? 'createChunkDescriptorsStreaming method' : null,
            !caps.hasMemoryPlanning ? 'planMemoryStrategy method' : null,
            !caps.hasSchemaDefinition ? 'defineInputSchema method' : null
          ].filter(Boolean)
        });
      }
    }
    for (const [name] of this.assemblyStrategies.entries()) {
      const caps = this.streamingCapabilities.get(name);
      if (caps.hasStreaming && caps.hasBatch) {
        analysis.ready.push({ name, type: 'assembly', capabilities: caps });
      } else {
        analysis.needsUpgrade.push({
          name, type: 'assembly',
          missing: [
            !caps.hasStreaming ? 'processChunkResult method' : null,
            !caps.hasMemoryMgmt ? 'initOutputStore method' : null,
            !caps.hasProgress ? 'onBlockComplete callback support' : null
          ].filter(Boolean)
        });
      }
    }
    if (analysis.needsUpgrade.length > 0) {
      analysis.recommendations.push('Consider upgrading strategies to support streaming for better performance');
      analysis.recommendations.push('Use block_matrix strategy as a reference for streaming implementation');
    }
    return analysis;
  }

  /**
   * Initialize built-in strategies, including Transformer strategies.
   */
  initializeBuiltInStrategies() {
    // Linear (debug/simple) strategies
    this.registerChunkingStrategy(new LinearChunkingStrategy());
    this.registerAssemblyStrategy(new LinearAssemblyStrategy());

    // Existing project strategies
    this.registerChunkingStrategy(new DistributedSortChunkingStrategy());
    this.registerAssemblyStrategy(new DistributedSortAssemblyStrategy());

    this.registerChunkingStrategy(new DistributedConvolutionChunkingStrategy());
    this.registerAssemblyStrategy(new DistributedConvolutionAssemblyStrategy());

    this.registerChunkingStrategy(new ECMStage1ChunkingStrategy());
    this.registerAssemblyStrategy(new ECMStage1AssemblyStrategy());

    this.registerChunkingStrategy(new BlockMatrixChunkingStrategy());
    this.registerAssemblyStrategy(new BlockMatrixAssemblyStrategy());

    this.registerChunkingStrategy(new SimpleNeuralNetworkChunkingStrategy());
    this.registerAssemblyStrategy(new SimpleNeuralNetworkAssemblyStrategy());

    // NEW: Transformer strategies
    // this.registerChunkingStrategy(new AttentionChunkingStrategy());
    // this.registerAssemblyStrategy(new AttentionAssemblyStrategy());
    // this.registerChunkingStrategy(new FeedForwardChunkingStrategy());
    // this.registerAssemblyStrategy(new FeedForwardAssemblyStrategy());
    // this.registerChunkingStrategy(new LayerNormChunkingStrategy());
    // this.registerAssemblyStrategy(new LayerNormAssemblyStrategy());

    // Shader templates
    this.registerShaderTemplate('linear_process', this.getLinearProcessShader());
    this.registerShaderTemplate('multi_buffer_process', this.getMultiBufferProcessShader());
    this.registerShaderTemplate('matrix_multiply_tiled', this.getMatrixMultiplyTiledShader());
    this.registerShaderTemplate('matrix_tile_extract', this.getMatrixTileExtractShader());

    if (__DEBUG_ON__) console.log(`Initialized ${this.chunkingStrategies.size} chunking strategies and ${this.assemblyStrategies.size} assembly strategies`);
  }

  /** WGSL: linear processing (fixed indexing bug) */
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
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= params.element_count) { return; }
        let g = params.start_element + i;
        if (g >= params.total_elements) { return; }
        let v = input_data[g];
        output_data[i] = v * v;
      }`;
  }

  /** WGSL: two input buffers -> sum & product */
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
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let i = gid.x;
        if (i >= params.element_count) { return; }
        let a = input_a[i];
        let b = input_b[i];
        output_sum[i] = a + b;
        output_product[i] = a * b;
      }`;
  }

  /** WGSL: tiled matmul (fixed missing semicolons) */
  getMatrixMultiplyTiledShader() {
    return `
      struct TileParams {
        matrix_n: u32,
        tile_start_row: u32,
        tile_start_col: u32,
        tile_rows: u32,
        tile_cols: u32,
        tile_size: u32,
      };
      @group(0) @binding(0) var<uniform> params: TileParams;
      @group(0) @binding(1) var<storage, read> input_data: array<f32>;
      @group(0) @binding(2) var<storage, read_write> tile_output: array<f32>;
      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let n = params.matrix_n;
        let r = gid.x;
        let c = gid.y;
        if (r >= params.tile_rows || c >= params.tile_cols) { return; }
        let gr = params.tile_start_row + r;
        let gc = params.tile_start_col + c;
        if (gr >= n || gc >= n) { return; }
        let header_size = 1u;
        let a_offset = header_size;
        let b_offset = header_size + n * n;
        var sum: f32 = 0.0;
        for (var k: u32 = 0u; k < n; k = k + 1u) {
          let a_val = input_data[a_offset + gr * n + k];
          let b_val = input_data[b_offset + k * n + gc];
          sum = sum + a_val * b_val;
        }
        let out_idx = r * params.tile_cols + c;
        tile_output[out_idx] = sum;
      }`;
  }

  /** WGSL: extract a rectangular tile from a square matrix (fixed missing semicolons) */
  getMatrixTileExtractShader() {
    return `
      struct TileParams {
        matrix_n: u32,
        tile_start_row: u32,
        tile_start_col: u32,
        tile_rows: u32,
        tile_cols: u32,
        tile_size: u32,
      };
      @group(0) @binding(0) var<uniform> params: TileParams;
      @group(0) @binding(1) var<storage, read> input_data: array<f32>;
      @group(0) @binding(2) var<storage, read_write> tile_output: array<f32>;
      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let r = gid.x;
        let c = gid.y;
        if (r >= params.tile_rows || c >= params.tile_cols) { return; }
        let gr = params.tile_start_row + r;
        let gc = params.tile_start_col + c;
        if (gr >= params.matrix_n || gc >= params.matrix_n) { return; }
        let in_idx = gr * params.matrix_n + gc;
        let out_idx = r * params.tile_cols + c;
        tile_output[out_idx] = input_data[in_idx];
      }`;
  }
}

// Built-in Linear Chunking Strategy (Updated for Multi-Input/Output)
class LinearChunkingStrategy extends BaseChunkingStrategy {
  constructor() { super('linear'); }

  defineInputSchema() {
    return {
      inputs: [ { name: 'input', type: 'storage_buffer', binding: 1, elementType: 'f32', chunking: 'parallel' } ],
      outputs: [ { name: 'output', type: 'storage_buffer', binding: 2, elementType: 'f32' } ]
    };
  }

  planExecution(workload) {
    const { elementSize = 4, chunkSize = 1024 } = workload.metadata || {};
    const schema = this.defineInputSchema();
    const parsedInputs = this.parseMultipleInputs(workload.input, schema);
    const firstInputKey = Object.keys(parsedInputs)[0];
    const inputBuffer = firstInputKey ? Buffer.from(parsedInputs[firstInputKey], 'base64') : Buffer.alloc(0);
    const totalElements = Math.floor(inputBuffer.length / elementSize);
    const totalChunks = Math.ceil(totalElements / chunkSize);
    return {
      strategy: this.name,
      totalChunks,
      schema,
      metadata: {
        elementSize, chunkSize, totalElements,
        inputData: workload.input,
        outputSizes: workload.outputSizes || [totalElements * elementSize]
      },
      assemblyStrategy: 'linear_assembly',
      shaderTemplate: (workload.metadata && workload.metadata.shaderTemplate) || 'linear_process'
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
      const inputChunks = this.chunkInputs(schema, parsedInputs, chunkIndex, plan.totalChunks, plan.metadata);
      const outputSizes = this.computeChunkOutputSizes(schema, plan.metadata, chunkIndex, plan.totalChunks);
      descriptors.push({
        chunkId: `linear-${chunkIndex}`,
        chunkIndex,
        parentId: plan.parentId,
        framework: 'webgpu',
        kernel: this.getDefaultShader(), // fixed to avoid creating a new registry
        entry: 'main',
        workgroupCount: [Math.ceil(actualChunkSize / 64), 1, 1],
        inputs: inputChunks,
        outputSizes: outputSizes,
        inputData: inputChunks[0] || '',
        outputSize: outputSizes[0] || 0,
        uniforms: { start_element: startElement, element_count: actualChunkSize, total_elements: totalElements },
        assemblyMetadata: { chunkIndex, startElement, elementCount: actualChunkSize }
      });
    }
    return descriptors;
  }

  chunkInput(inputData, chunkIndex, totalChunks, chunkingParams = {}) {
    const { elementSize = 4 } = chunkingParams;
    const buffer = Buffer.from(inputData, 'base64');
    const totalElements = Math.floor(buffer.length / elementSize);
    const elementsPerChunk = Math.ceil(totalElements / totalChunks);
    const startElement = chunkIndex * elementsPerChunk;
    const endElement = Math.min((chunkIndex + 1) * elementsPerChunk, totalElements);
    const startByte = startElement * elementSize;
    const endByte = endElement * elementSize;
    return buffer.slice(startByte, endByte).toString('base64');
  }

  getDefaultShader() { return this.getLinearProcessShader?.() || `/* fallback provided by registry */`; }
}

// Built-in Linear Assembly Strategy (Updated for Multi-Output)
class LinearAssemblyStrategy extends BaseAssemblyStrategy {
  constructor() { super('linear_assembly'); }

  getDefaultSchema() {
    return { outputs: [ { name: 'output', type: 'storage_buffer', elementType: 'f32' } ] };
  }

  assembleResults(completedChunks, plan) {
    const validation = this.validateChunks(completedChunks, plan);
    if (!validation.valid) {
      return { success: false, error: validation.error, missing: validation.missing };
    }
    try {
      const schema = plan.schema || this.getDefaultSchema();
      const sortedChunks = this.sortChunks(completedChunks);
      if (schema.outputs.length > 1) {
        return this.assembleMultipleOutputs(sortedChunks, plan, schema);
      } else {
        return this.assembleSingleOutput(sortedChunks, plan, schema.outputs[0]);
      }
    } catch (error) {
      return { success: false, error: `Linear assembly failed: ${error.message}` };
    }
  }

  assembleSingleOutput(chunks, plan, outputDef) {
    const buffers = chunks.map(c => Buffer.from(c.output || c.outputs?.[0] || '', 'base64'));
    return Buffer.concat(buffers);
  }
}

export default ChunkingStrategyRegistry;

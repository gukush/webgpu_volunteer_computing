import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';
import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default class DistributedSortChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('distributed_sort');
  }

  defineInputSchema() {
    return {
      inputs: [{
        name: 'unsorted_data_chunk',
        type: 'storage_buffer',
        binding: 1,
        elementType: 'f32'
      }],
      outputs: [{
        name: 'sorted_data_chunk',
        type: 'storage_buffer',
        binding: 2,
        elementType: 'f32'
      }],
      uniforms: [{
        name: 'params',
        type: 'uniform_buffer',
        binding: 0,
        fields: [{ name: 'element_count', type: 'u32' }]
      }]
    };
  }

  planExecution(workload) {
    const { elementCount, chunkSize = 1024 * 64, elementSize = 4 } = workload.metadata;

    if (!elementCount) {
      throw new Error("workload.metadata must include 'elementCount'.");
    }

    const elementsPerChunk = Math.floor(chunkSize / elementSize);
    const totalChunks = Math.ceil(elementCount / elementsPerChunk);

    return {
      strategy: this.name,
      totalChunks,
      assemblyStrategy: 'distributed_sort_assembly',
      metadata: {
        ...workload.metadata,
        elementsPerChunk,
        elementSize,
        totalChunks
      }
    };
  }

  async createChunkDescriptorsStreaming(plan, dispatchCallback) {
    const { elementCount, elementsPerChunk, elementSize } = plan.metadata;
    const framework = plan.framework || 'webgpu';

    const inputFileRef = (plan.inputRefs || []).find(r => r.name === 'unsorted_data');
    if (!inputFileRef?.path) {
      throw new Error('Distributed sort requires an input file named "unsorted_data".');
    }

    let fileHandle;
    try {
      fileHandle = await fs.open(inputFileRef.path, 'r');
      let dispatchedCount = 0;

      for (let i = 0; i < plan.totalChunks; i++) {
        const startElement = i * elementsPerChunk;
        const numElementsInChunk = Math.min(elementsPerChunk, elementCount - startElement);
        const chunkByteSize = numElementsInChunk * elementSize;
        const fileOffset = 4 + startElement * elementSize; // +4 for header

        const chunkBuffer = Buffer.alloc(chunkByteSize);
        await fileHandle.read(chunkBuffer, 0, chunkByteSize, fileOffset);

        const descriptor = await this.createFrameworkSpecificDescriptor(
          framework, i, chunkBuffer, numElementsInChunk, plan.parentId
        );

        await dispatchCallback(descriptor);
        dispatchedCount++;
      }
      return { success: true, totalChunks: dispatchedCount };
    } finally {
      await fileHandle?.close();
    }
  }

  async createFrameworkSpecificDescriptor(framework, chunkIndex, chunkBuffer, elementCount, parentId) {
    const baseDescriptor = {
      chunkId: `sort-chunk-${chunkIndex}`,
      chunkIndex,
      parentId,
      framework,
      inputs: [{ name: 'unsorted_data_chunk', data: chunkBuffer.toString('base64') }],
      outputs: [{ name: 'sorted_data_chunk', size: chunkBuffer.length }],
      metadata: { element_count: elementCount },
      assemblyMetadata: { chunkIndex, elementCount }
    };

    switch (framework) {
      case 'webgpu':
        const kernelPath = path.join(__dirname, '..', 'kernels', 'sorting', 'bitonic_sort_webgpu_compute.wgsl');
        const kernel = await fs.readFile(kernelPath, 'utf8');
        // A workgroup size of 256 is common for sorting kernels.
        // We dispatch enough workgroups to cover all elements.
        return {
          ...baseDescriptor,
          kernel,
          entry: 'main',
          workgroupCount: [Math.ceil(elementCount / 256), 1, 1]
        };
      // Add cases for 'webgl', 'cuda', etc. here, loading their respective kernels.
      default:
        throw new Error(`Unsupported framework for sorting: ${framework}`);
    }
  }
}
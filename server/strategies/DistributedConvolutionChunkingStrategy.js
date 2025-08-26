// DistributedConvolutionChunkingStrategy.js - Multi-channel convolution with spatial tiling
import { BaseChunkingStrategy } from './base/BaseChunkingStrategy.js';

export default class DistributedConvolutionChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('distributed_convolution');
  }

  defineInputSchema() {
    return {
      inputs: [
        {
          name: 'input_tensor',
          type: 'storage_buffer',
          binding: 1,
          elementType: 'f32',
          chunking: 'spatial_tiled' // Custom chunking type for spatial overlap
        },
        {
          name: 'filter_weights',
          type: 'storage_buffer',
          binding: 2,
          elementType: 'f32',
          chunking: 'replicate' // Same filter for all chunks
        }
      ],
      outputs: [
        {
          name: 'output_tensor',
          type: 'storage_buffer',
          binding: 3,
          elementType: 'f32'
        }
      ],
      uniforms: [
        {
          name: 'conv_params',
          type: 'uniform_buffer',
          binding: 0,
          fields: [
            { name: 'input_height', type: 'u32' },
            { name: 'input_width', type: 'u32' },
            { name: 'input_channels', type: 'u32' },
            { name: 'output_channels', type: 'u32' },
            { name: 'filter_height', type: 'u32' },
            { name: 'filter_width', type: 'u32' },
            { name: 'stride_y', type: 'u32' },
            { name: 'stride_x', type: 'u32' },
            { name: 'padding_y', type: 'u32' },
            { name: 'padding_x', type: 'u32' },
            { name: 'tile_start_y', type: 'u32' },
            { name: 'tile_start_x', type: 'u32' },
            { name: 'tile_height', type: 'u32' },
            { name: 'tile_width', type: 'u32' }
          ]
        }
      ]
    };
  }

  planExecution(workload) {
    const {
      inputHeight,
      inputWidth,
      inputChannels,
      outputChannels,
      filterHeight,
      filterWidth,
      strideY = 1,
      strideX = 1,
      paddingY = 0,
      paddingX = 0,
      tileHeight = 128,
      tileWidth = 128
    } = workload.metadata;

    if (!inputHeight || !inputWidth || !inputChannels || !outputChannels) {
      throw new Error('Input dimensions and channel counts required in metadata');
    }

    if (!filterHeight || !filterWidth) {
      throw new Error('Filter dimensions required in metadata');
    }

    // Calculate output dimensions
    const outputHeight = Math.floor((inputHeight + 2 * paddingY - filterHeight) / strideY) + 1;
    const outputWidth = Math.floor((inputWidth + 2 * paddingX - filterWidth) / strideX) + 1;

    // Calculate tiling
    const tilesY = Math.ceil(outputHeight / tileHeight);
    const tilesX = Math.ceil(outputWidth / tileWidth);
    const totalChunks = tilesY * tilesX;

    // Calculate halo padding needed for convolution
    const haloY = Math.ceil((filterHeight - 1) / 2);
    const haloX = Math.ceil((filterWidth - 1) / 2);

    return {
      strategy: this.name,
      totalChunks,
      schema: this.defineInputSchema(),
      metadata: {
        ...workload.metadata,
        outputHeight,
        outputWidth,
        tilesY,
        tilesX,
        tileHeight,
        tileWidth,
        haloY,
        haloX,
        strideY,
        strideX,
        paddingY,
        paddingX
      },
      assemblyStrategy: 'distributed_convolution_assembly',
      shaderTemplate: workload.metadata.shaderTemplate || 'convolution_2d'
    };
  }

  async createChunkDescriptors(plan) {
    const {
      inputHeight, inputWidth, inputChannels, outputChannels,
      filterHeight, filterWidth,
      strideY, strideX, paddingY, paddingX,
      tileHeight, tileWidth, tilesY, tilesX,
      haloY, haloX, outputHeight, outputWidth
    } = plan.metadata;

    const framework = plan.framework || 'webgpu';

    // Get input data
    const inputFileRef = (plan.inputRefs || []).find(r => r.name === 'input_tensor')
                      || (plan.inputRefs || []).find(r => r.name === 'input');

    const filterFileRef = (plan.inputRefs || []).find(r => r.name === 'filter_weights')
                       || (plan.inputRefs || []).find(r => r.name === 'filter');

    let inputTensorBuffer = null;
    let filterWeightsBuffer = null;

    // Handle input tensor
    if (!inputFileRef && plan.metadata?.inputData) {
      const parsedInputs = typeof plan.metadata.inputData === 'string'
        ? JSON.parse(plan.metadata.inputData)
        : plan.metadata.inputData;

      if (parsedInputs.input_tensor) {
        inputTensorBuffer = Buffer.from(parsedInputs.input_tensor, 'base64');
      }
      if (parsedInputs.filter_weights) {
        filterWeightsBuffer = Buffer.from(parsedInputs.filter_weights, 'base64');
      }
    }

    if (!inputTensorBuffer && !inputFileRef) {
      throw new Error('DistributedConvolution requires input tensor. Upload files first via POST /api/workloads/:id/inputs');
    }

    const descriptors = [];
    let chunkIndex = 0;

    for (let tileY = 0; tileY < tilesY; tileY++) {
      for (let tileX = 0; tileX < tilesX; tileX++) {
        // Calculate tile boundaries in output space
        const outputStartY = tileY * tileHeight;
        const outputStartX = tileX * tileWidth;
        const outputEndY = Math.min(outputStartY + tileHeight, outputHeight);
        const outputEndX = Math.min(outputStartX + tileWidth, outputWidth);

        // Calculate required input region (including halo)
        const inputStartY = Math.max(0, outputStartY * strideY - paddingY - haloY);
        const inputStartX = Math.max(0, outputStartX * strideX - paddingX - haloX);
        const inputEndY = Math.min(inputHeight, outputEndY * strideY - paddingY + filterHeight + haloY);
        const inputEndX = Math.min(inputWidth, outputEndX * strideX - paddingX + filterWidth + haloX);

        const tileInputHeight = inputEndY - inputStartY;
        const tileInputWidth = inputEndX - inputStartX;
        const tileOutputHeight = outputEndY - outputStartY;
        const tileOutputWidth = outputEndX - outputStartX;

        // Extract input tile data
        let inputTileData;
        if (inputTensorBuffer) {
          inputTileData = this.extractTensorTile(
            inputTensorBuffer, inputHeight, inputWidth, inputChannels,
            inputStartY, inputStartX, tileInputHeight, tileInputWidth
          );
        } else {
          // For file input, create a reference to the region
          inputTileData = {
            type: 'file_region',
            path: inputFileRef.path,
            startY: inputStartY,
            startX: inputStartX,
            height: tileInputHeight,
            width: tileInputWidth
          };
        }

        // Calculate output size
        const outputTileSize = tileOutputHeight * tileOutputWidth * outputChannels * 4; // 4 bytes per float

        const descriptor = this.createFrameworkSpecificDescriptor(
          framework, chunkIndex, tileY, tileX,
          inputTileData, filterWeightsBuffer,
          {
            inputHeight, inputWidth, inputChannels,
            outputChannels, filterHeight, filterWidth,
            strideY, strideX, paddingY, paddingX,
            tileInputHeight, tileInputWidth,
            tileOutputHeight, tileOutputWidth,
            outputStartY, outputStartX,
            inputStartY, inputStartX
          },
          outputTileSize, plan.parentId
        );

        descriptors.push(descriptor);
        chunkIndex++;
      }
    }

    return descriptors;
  }

  createFrameworkSpecificDescriptor(framework, chunkIndex, tileY, tileX, inputTileData, filterBuffer, params, outputSize, parentId) {
    const baseDescriptor = {
      chunkId: `conv-tile-${tileY}-${tileX}`,
      chunkIndex,
      parentId,
      framework,

      inputs: [
        {
          name: 'input_tensor',
          data: typeof inputTileData === 'object' && inputTileData.type === 'file_region'
            ? null : inputTileData.toString('base64')
        },
        {
          name: 'filter_weights',
          data: filterBuffer ? filterBuffer.toString('base64') : ''
        }
      ],
      outputs: [
        { name: 'output_tensor', size: outputSize }
      ],

      metadata: {
        ...params,
        tileY, tileX
      },

      assemblyMetadata: {
        tileY, tileX,
        outputStartY: params.outputStartY,
        outputStartX: params.outputStartX,
        tileOutputHeight: params.tileOutputHeight,
        tileOutputWidth: params.tileOutputWidth,
        outputChannels: params.outputChannels
      }
    };

    switch (framework) {
      case 'webgpu':
        return {
          ...baseDescriptor,
          kernel: this.getWebGPUConvolutionShader(),
          entry: 'conv2d_main',
          workgroupCount: [
            Math.ceil(params.tileOutputWidth / 16),
            Math.ceil(params.tileOutputHeight / 16),
            Math.ceil(params.outputChannels / 4)
          ]
        };

      case 'webgl':
        return {
          ...baseDescriptor,
          webglVertexShader: this.getWebGLConvolutionVertexShader(),
          webglFragmentShader: this.getWebGLConvolutionFragmentShader(),
          webglVaryings: ['v_result'],
          webglNumElements: params.tileOutputHeight * params.tileOutputWidth * params.outputChannels,
          webglInputSpec: {
            type: 'texture',
            format: 'float32',
            internalFormat: 'R32F'
          }
        };

      case 'javascript':
        return {
          ...baseDescriptor,
          kernel: this.getJavaScriptConvolutionKernel(),
          entry: 'conv2d_cpu',
          jsExecutionHints: {
            algorithm: 'convolution_2d',
            parallelizable: true,
            memoryAccess: 'spatial_locality'
          }
        };

      case 'cuda':
        return {
          ...baseDescriptor,
          kernel: this.getCUDAConvolutionKernel(),
          entry: 'conv2d_cuda',
          blockDim: [16, 16, 1],
          gridDim: [
            Math.ceil(params.tileOutputWidth / 16),
            Math.ceil(params.tileOutputHeight / 16),
            Math.ceil(params.outputChannels / 1)
          ]
        };

      default:
        throw new Error(`Unsupported framework: ${framework}`);
    }
  }

  extractTensorTile(tensorBuffer, height, width, channels, startY, startX, tileHeight, tileWidth) {
    const floatSize = 4;
    const tileBuffer = Buffer.alloc(tileHeight * tileWidth * channels * floatSize);

    let destOffset = 0;
    for (let y = 0; y < tileHeight; y++) {
      for (let x = 0; x < tileWidth; x++) {
        const srcY = startY + y;
        const srcX = startX + x;

        if (srcY < height && srcX < width) {
          // Copy all channels for this spatial position
          const srcOffset = (srcY * width + srcX) * channels * floatSize;
          const channelBytes = channels * floatSize;

          tensorBuffer.copy(tileBuffer, destOffset, srcOffset, srcOffset + channelBytes);
        }
        // Note: Out-of-bounds regions remain zero-padded

        destOffset += channels * floatSize;
      }
    }

    return tileBuffer;
  }

  // Shader implementations for different frameworks

  getWebGPUConvolutionShader() {
    return `
      struct ConvParams {
        input_height: u32,
        input_width: u32,
        input_channels: u32,
        output_channels: u32,
        filter_height: u32,
        filter_width: u32,
        stride_y: u32,
        stride_x: u32,
        padding_y: u32,
        padding_x: u32,
        tile_start_y: u32,
        tile_start_x: u32,
        tile_height: u32,
        tile_width: u32,
      }

      @group(0) @binding(0) var<uniform> params: ConvParams;
      @group(0) @binding(1) var<storage, read> input_tensor: array<f32>;
      @group(0) @binding(2) var<storage, read> filter_weights: array<f32>;
      @group(0) @binding(3) var<storage, read_write> output_tensor: array<f32>;

      @compute @workgroup_size(16, 16, 4)
      fn conv2d_main(@builtin(global_invocation_id) gid: vec3<u32>) {
        let out_x = gid.x;
        let out_y = gid.y;
        let out_c = gid.z;

        if (out_x >= params.tile_width || out_y >= params.tile_height || out_c >= params.output_channels) {
          return;
        }

        var sum = 0.0;

        // Convolution loop
        for (var fy: u32 = 0u; fy < params.filter_height; fy++) {
          for (var fx: u32 = 0u; fx < params.filter_width; fx++) {
            for (var ic: u32 = 0u; ic < params.input_channels; ic++) {

              // Input position (accounting for stride and padding)
              let in_y = out_y * params.stride_y + fy;
              let in_x = out_x * params.stride_x + fx;

              // Check bounds (with implicit padding)
              if (in_y < params.input_height && in_x < params.input_width) {
                // Input tensor index: [y, x, channel]
                let input_idx = (in_y * params.input_width + in_x) * params.input_channels + ic;

                // Filter weight index: [out_channel, in_channel, fy, fx]
                let filter_idx = ((out_c * params.input_channels + ic) * params.filter_height + fy) * params.filter_width + fx;

                sum += input_tensor[input_idx] * filter_weights[filter_idx];
              }
            }
          }
        }

        // Output tensor index: [y, x, out_channel]
        let output_idx = (out_y * params.tile_width + out_x) * params.output_channels + out_c;
        output_tensor[output_idx] = sum;
      }
    `;
  }

  getWebGLConvolutionVertexShader() {
    return `#version 300 es
      precision highp float;

      in float a_index;

      uniform int u_input_height;
      uniform int u_input_width;
      uniform int u_input_channels;
      uniform int u_output_channels;
      uniform int u_filter_height;
      uniform int u_filter_width;
      uniform int u_stride_y;
      uniform int u_stride_x;
      uniform int u_tile_height;
      uniform int u_tile_width;

      uniform sampler2D u_input_0;  // Input tensor (as texture)
      uniform sampler2D u_input_1;  // Filter weights (as texture)

      out float v_result;

      void main() {
        int idx = int(a_index);
        int tile_area = u_tile_height * u_tile_width;

        // Decode 3D position from linear index
        int out_c = idx / tile_area;
        int spatial_idx = idx % tile_area;
        int out_y = spatial_idx / u_tile_width;
        int out_x = spatial_idx % u_tile_width;

        if (out_c >= u_output_channels) {
          v_result = 0.0;
          gl_Position = vec4(0.0);
          return;
        }

        float sum = 0.0;

        // Convolution
        for (int fy = 0; fy < u_filter_height; fy++) {
          for (int fx = 0; fx < u_filter_width; fx++) {
            for (int ic = 0; ic < u_input_channels; ic++) {

              int in_y = out_y * u_stride_y + fy;
              int in_x = out_x * u_stride_x + fx;

              if (in_y < u_input_height && in_x < u_input_width) {
                // Sample input tensor (packed as texture)
                int input_linear = (in_y * u_input_width + in_x) * u_input_channels + ic;
                ivec2 input_coord = ivec2(input_linear % 1024, input_linear / 1024); // Assume max 1024 width
                float input_val = texelFetch(u_input_0, input_coord, 0).r;

                // Sample filter weights
                int filter_linear = ((out_c * u_input_channels + ic) * u_filter_height + fy) * u_filter_width + fx;
                ivec2 filter_coord = ivec2(filter_linear % 1024, filter_linear / 1024);
                float filter_val = texelFetch(u_input_1, filter_coord, 0).r;

                sum += input_val * filter_val;
              }
            }
          }
        }

        v_result = sum;
        gl_Position = vec4(0.0);
      }
    `;
  }

  getWebGLConvolutionFragmentShader() {
    return `#version 300 es
      precision highp float;
      out vec4 fragColor;
      void main() { fragColor = vec4(1.0); }
    `;
  }

  getJavaScriptConvolutionKernel() {
    return `
      // CPU implementation of 2D convolution
      function conv2d_cpu(inputTensor, filterWeights, params) {
        const {
          input_height, input_width, input_channels,
          output_channels, filter_height, filter_width,
          stride_y, stride_x, padding_y, padding_x,
          tile_height, tile_width
        } = params;

        const outputSize = tile_height * tile_width * output_channels;
        const result = new Float32Array(outputSize);

        // Main convolution loop
        for (let out_c = 0; out_c < output_channels; out_c++) {
          for (let out_y = 0; out_y < tile_height; out_y++) {
            for (let out_x = 0; out_x < tile_width; out_x++) {

              let sum = 0.0;

              // Convolution kernel
              for (let fy = 0; fy < filter_height; fy++) {
                for (let fx = 0; fx < filter_width; fx++) {
                  for (let ic = 0; ic < input_channels; ic++) {

                    const in_y = out_y * stride_y + fy - padding_y;
                    const in_x = out_x * stride_x + fx - padding_x;

                    // Bounds check
                    if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                      const input_idx = (in_y * input_width + in_x) * input_channels + ic;
                      const filter_idx = ((out_c * input_channels + ic) * filter_height + fy) * filter_width + fx;

                      sum += inputTensor[input_idx] * filterWeights[filter_idx];
                    }
                  }
                }
              }

              const output_idx = (out_y * tile_width + out_x) * output_channels + out_c;
              result[output_idx] = sum;
            }
          }
        }

        return result;
      }
    `;
  }

  getCUDAConvolutionKernel() {
    return `
      extern "C" __global__ void conv2d_cuda(
          int input_height, int input_width, int input_channels,
          int output_channels, int filter_height, int filter_width,
          int stride_y, int stride_x, int padding_y, int padding_x,
          int tile_height, int tile_width,
          const float* input_tensor,
          const float* filter_weights,
          float* output_tensor
      ) {
          int out_x = blockIdx.x * blockDim.x + threadIdx.x;
          int out_y = blockIdx.y * blockDim.y + threadIdx.y;
          int out_c = blockIdx.z;

          if (out_x >= tile_width || out_y >= tile_height || out_c >= output_channels) {
              return;
          }

          float sum = 0.0f;

          // Convolution
          for (int fy = 0; fy < filter_height; fy++) {
              for (int fx = 0; fx < filter_width; fx++) {
                  for (int ic = 0; ic < input_channels; ic++) {

                      int in_y = out_y * stride_y + fy - padding_y;
                      int in_x = out_x * stride_x + fx - padding_x;

                      if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                          int input_idx = (in_y * input_width + in_x) * input_channels + ic;
                          int filter_idx = ((out_c * input_channels + ic) * filter_height + fy) * filter_width + fx;

                          sum += input_tensor[input_idx] * filter_weights[filter_idx];
                      }
                  }
              }
          }

          int output_idx = (out_y * tile_width + out_x) * output_channels + out_c;
          output_tensor[output_idx] = sum;
      }
    `;
  }
}
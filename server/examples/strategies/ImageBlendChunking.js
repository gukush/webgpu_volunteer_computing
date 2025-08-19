// examples/custom_strategies/ImageBlendChunking.js
// Example: Custom image processing with multi-input support

import { BaseChunkingStrategy } from '../../strategies/base/BaseChunkingStrategy.js';

export default class ImageBlendChunking extends BaseChunkingStrategy {
  constructor() {
    super('image_blend_custom');
  }

  // Define what inputs this strategy expects
  defineInputSchema() {
    return {
      inputs: [
        {
          name: 'image_a',
          type: 'storage_buffer',
          binding: 1,
          elementType: 'u32',  // RGBA pixels as u32
          chunking: 'tiled_2d'
        },
        {
          name: 'image_b', 
          type: 'storage_buffer',
          binding: 2,
          elementType: 'u32',
          chunking: 'tiled_2d'  // Same tiling as image_a
        }
      ],
      outputs: [
        {
          name: 'blended_image',
          type: 'storage_buffer',
          binding: 3,
          elementType: 'u32'
        }
      ],
      uniforms: [
        {
          name: 'blend_params',
          type: 'uniform_buffer',
          binding: 0,
          fields: ['image_width', 'image_height', 'tile_start_x', 'tile_start_y', 'tile_width', 'tile_height']
        }
      ]
    };
  }

  validateWorkload(workload) {
    const required = ['imageWidth', 'imageHeight', 'tileSize'];
    const missing = required.filter(field => !workload.metadata[field]);
    
    if (missing.length > 0) {
      return { valid: false, error: `Missing: ${missing.join(', ')}` };
    }

    return { valid: true };
  }

  planExecution(workload) {
    const { imageWidth, imageHeight, tileSize } = workload.metadata;
    const tilesX = Math.ceil(imageWidth / tileSize);
    const tilesY = Math.ceil(imageHeight / tileSize);
    
    return {
      strategy: this.name,
      totalChunks: tilesX * tilesY,
      inputSchema: this.defineInputSchema(),
      metadata: {
        imageWidth, imageHeight, tileSize, tilesX, tilesY,
        // Parse multi-input data
        parsedInputs: this.parseMultipleInputs(workload.input, this.defineInputSchema())
      },
      assemblyStrategy: 'image_blend_assembly'
    };
  }

  createChunkDescriptors(plan) {
    const { imageWidth, imageHeight, tileSize, tilesX, tilesY, parsedInputs } = plan.metadata;
    const descriptors = [];
    
    let chunkIndex = 0;
    for (let tileY = 0; tileY < tilesY; tileY++) {
      for (let tileX = 0; tileX < tilesX; tileX++) {
        const startX = tileX * tileSize;
        const startY = tileY * tileSize;
        const endX = Math.min((tileX + 1) * tileSize, imageWidth);
        const endY = Math.min((tileY + 1) * tileSize, imageHeight);
        
        const actualWidth = endX - startX;
        const actualHeight = endY - startY;

        // Extract tile data for both images
        const tileDataA = this.extractImageTile(
          parsedInputs.image_a, imageWidth, imageHeight, 
          startX, startY, actualWidth, actualHeight
        );
        
        const tileDataB = this.extractImageTile(
          parsedInputs.image_b, imageWidth, imageHeight,
          startX, startY, actualWidth, actualHeight  
        );

        descriptors.push({
          chunkId: `tile-${tileX}-${tileY}`,
          chunkIndex,
          parentId: plan.parentId,
          
          framework: 'webgpu',
          kernel: this.getImageBlendShader(),
          entry: 'main',
          workgroupCount: [
            Math.ceil(actualWidth / 16),
            Math.ceil(actualHeight / 16), 
            1
          ],
          
          // Multi-input data for this chunk
          inputSchema: plan.inputSchema,
          chunkInputs: {
            image_a: tileDataA.toString('base64'),
            image_b: tileDataB.toString('base64')
          },
          
          outputSize: actualWidth * actualHeight * 4, // RGBA bytes
          
          uniforms: {
            image_width: imageWidth,
            image_height: imageHeight,
            tile_start_x: startX,
            tile_start_y: startY,
            tile_width: actualWidth,
            tile_height: actualHeight
          },
          
          assemblyMetadata: {
            tileX, tileY, startX, startY, actualWidth, actualHeight
          }
        });
        
        chunkIndex++;
      }
    }

    return descriptors;
  }

  extractImageTile(imageData, imageWidth, imageHeight, startX, startY, tileWidth, tileHeight) {
    // Convert base64 to buffer
    const imageBuffer = Buffer.from(imageData, 'base64');
    const imageArray = new Uint32Array(imageBuffer.buffer);
    
    // Extract the tile
    const tileBuffer = Buffer.alloc(tileWidth * tileHeight * 4);
    const tileArray = new Uint32Array(tileBuffer.buffer);
    
    for (let y = 0; y < tileHeight; y++) {
      for (let x = 0; x < tileWidth; x++) {
        const srcIndex = (startY + y) * imageWidth + (startX + x);
        const dstIndex = y * tileWidth + x;
        tileArray[dstIndex] = imageArray[srcIndex];
      }
    }
    
    return tileBuffer;
  }

  getImageBlendShader() {
    return `
      struct BlendParams {
          image_width: u32,
          image_height: u32,
          tile_start_x: u32,
          tile_start_y: u32,
          tile_width: u32,
          tile_height: u32,
      };

      @group(0) @binding(0) var<uniform> params: BlendParams;
      @group(0) @binding(1) var<storage, read> image_a: array<u32>;      // RGBA packed
      @group(0) @binding(2) var<storage, read> image_b: array<u32>;      // RGBA packed  
      @group(0) @binding(3) var<storage, read_write> output: array<u32>; // RGBA packed

      @compute @workgroup_size(16, 16, 1)
      fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
          let local_x = global_id.x;
          let local_y = global_id.y;
          
          if (local_x >= params.tile_width || local_y >= params.tile_height) {
              return;
          }
          
          let tile_index = local_y * params.tile_width + local_x;
          
          // Read pixels from both input images
          let pixel_a = unpack_rgba(image_a[tile_index]);
          let pixel_b = unpack_rgba(image_b[tile_index]);
          
          // Simple blend: 50% mix
          let blended = mix(pixel_a, pixel_b, 0.5);
          
          output[tile_index] = pack_rgba(blended);
      }

      fn unpack_rgba(packed: u32) -> vec4<f32> {
          let r = f32((packed >> 24u) & 0xFFu) / 255.0;
          let g = f32((packed >> 16u) & 0xFFu) / 255.0;
          let b = f32((packed >> 8u) & 0xFFu) / 255.0;
          let a = f32(packed & 0xFFu) / 255.0;
          return vec4<f32>(r, g, b, a);
      }

      fn pack_rgba(color: vec4<f32>) -> u32 {
          let r = u32(clamp(color.r * 255.0, 0.0, 255.0));
          let g = u32(clamp(color.g * 255.0, 0.0, 255.0));
          let b = u32(clamp(color.b * 255.0, 0.0, 255.0));
          let a = u32(clamp(color.a * 255.0, 0.0, 255.0));
          return (r << 24u) | (g << 16u) | (b << 8u) | a;
      }
    `;
  }
}

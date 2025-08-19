import { BaseChunkingStrategy } from '../../strategies/base/BaseChunkingStrategy.js';

export default class MatrixTiledChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('matrix_tiled');
  }

  validateWorkload(workload) {
     const { matrixSize, tileSize, customShader } = workload.metadata || {};

    if (!matrixSize || !tileSize) {
      return {
        valid: false,
        error: 'matrixSize and tileSize are required in metadata'
      };
    }

    if (matrixSize <= 0 || tileSize <= 0) {
      return {
        valid: false,
        error: 'matrixSize and tileSize must be positive'
      };
    }

    if (tileSize > matrixSize) {
      return {
        valid: false,
        error: 'tileSize cannot be larger than matrixSize'
      };
    }

    if (customShader != null && typeof customShader !== 'string') {
      return { valid: false, error: 'customShader must be a string when provided' };
    }

    return { valid: true };
  }

  planExecution(workload) {
    const { matrixSize, tileSize } = workload.metadata;
    const tilesPerDim = Math.ceil(matrixSize / tileSize);
    const totalTiles = tilesPerDim * tilesPerDim;

    return {
      strategy: this.name,
      totalChunks: totalTiles,
      metadata: {
        matrixSize,
        tileSize,
        tilesPerDim,
        inputData: workload.input,
        customShader: workload.metadata?.customShader,
        entry: workload.metadata?.entry || 'main',
        inputSchema: workload.metadata?.inputSchema || null
      },
      assemblyStrategy: 'matrix_tiled_assembly'
    };
  }

  createChunkDescriptors(plan) {
    const { matrixSize, tileSize, tilesPerDim } = plan.metadata;
    const descriptors = [];

    let tileIndex = 0;
    for (let tileRow = 0; tileRow < tilesPerDim; tileRow++) {
      for (let tileCol = 0; tileCol < tilesPerDim; tileCol++) {
        const startRow = tileRow * tileSize;
        const startCol = tileCol * tileSize;
        const endRow = Math.min(startRow + tileSize, matrixSize);
        const endCol = Math.min(startCol + tileSize, matrixSize);

        const actualTileRows = endRow - startRow;
        const actualTileCols = endCol - startCol;
        const tileOutputSize = actualTileRows * actualTileCols * 4; // float32

        const entry = plan.metadata.entry || 'main';
        const kernel = plan.metadata.customShader || this.getTiledMatrixShader();

        // Default schema (uniforms + one input storage + one output storage).
        // If the user supplied inputSchema, use that instead.
        const defaultSchema = {
          uniforms: [{ name: 'TileParams', fields: [
            'matrix_n','tile_start_row','tile_start_col','tile_rows','tile_cols','tile_size'
          ]}],
          inputs:  [{ name: 'input_data', type: 'storage_buffer', elementType: 'f32' }],
          outputs: [{ name: 'tile_output', type: 'storage_buffer', elementType: 'f32' }]
        };
        const inputSchema = plan.metadata.inputSchema || defaultSchema;


        descriptors.push({
          chunkId: `tile-${tileRow}-${tileCol}`,
          chunkIndex: tileIndex,
          parentId: plan.parentId,

          framework: 'webgpu',
          kernel,
          entry,
          workgroupCount: [
            Math.ceil(actualTileRows / 16),
            Math.ceil(actualTileCols / 16),
            1
          ],

          inputData: plan.metadata.inputData,
          outputSize: tileOutputSize,

          uniforms: {
            matrix_n: matrixSize,
            tile_start_row: startRow,
            tile_start_col: startCol,
            tile_rows: actualTileRows,
            tile_cols: actualTileCols,
            tile_size: tileSize
          },

          inputSchema,
          chunkInputs: {
            input_data: plan.metadata.inputData
          },
          assemblyMetadata: {
            tileRow,
            tileCol,
            startRow,
            startCol,
            tileRows: actualTileRows,
            tileCols: actualTileCols,
            matrixSize
          }
        });

        tileIndex++;
      }
    }

    return descriptors;
  }

  getTiledMatrixShader() {
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

          // Input layout: [matrix_size_header, A_data..., B_data...]
          let header_size = 1u;
          let a_offset = header_size;
          let b_offset = header_size + n * n;

          var sum = 0.0;
          for (var k = 0u; k < n; k = k + 1u) {
              let a_val = input_data[a_offset + global_row * n + k];
              let b_val = input_data[b_offset + k * n + global_col];
              sum = sum + a_val * b_val;
          }

          tile_output[local_row * params.tile_cols + local_col] = sum;
      }
    `;
  }
}

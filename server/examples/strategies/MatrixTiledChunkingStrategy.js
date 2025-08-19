import { BaseChunkingStrategy } from '../../strategies/base/BaseChunkingStrategy.js';

export default class MatrixTiledChunkingStrategy extends BaseChunkingStrategy {
  constructor() {
    super('matrix_tiled');
  }

  /**
   * Define input/output schema for matrix tiled computation
   */
  defineInputSchema() {
    return {
      inputs: [
        {
          name: 'input_data',
          type: 'storage_buffer',
          binding: 1,
          elementType: 'f32',
          chunking: 'replicate' // All tiles need full matrix data
        }
      ],
      outputs: [
        {
          name: 'tile_output',
          type: 'storage_buffer',
          binding: 2,
          elementType: 'f32'
        }
      ]
    };
  }

  validateWorkload(workload) {
    // First run base validation
    const baseValidation = super.validateWorkload(workload);
    if (!baseValidation.valid) {
      return baseValidation;
    }

    // Strategy-specific validation
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

    const schema = this.defineInputSchema();

    return {
      strategy: this.name,
      totalChunks: totalTiles,
      schema: schema, // NEW: Include schema in plan
      metadata: {
        matrixSize,
        tileSize,
        tilesPerDim,
        inputData: workload.input,
        customShader: workload.metadata?.customShader,
        entry: workload.metadata?.entry || 'main',
        inputSchema: workload.metadata?.inputSchema || schema // Allow override
      },
      assemblyStrategy: 'matrix_tiled_assembly'
    };
  }

  createChunkDescriptors(plan) {
    const { matrixSize, tileSize, tilesPerDim } = plan.metadata;
    const schema = plan.schema;
    const parsedInputs = this.parseMultipleInputs(plan.metadata.inputData, schema);
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

        // NEW: Create inputs using the chunking system
        const inputChunks = this.chunkInputs(schema, parsedInputs, tileIndex, plan.totalChunks, plan.metadata);

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

          // NEW: Multi-input/output support
          inputs: inputChunks, // Array of base64 strings
          outputSizes: [tileOutputSize], // Array with one output

          // Backward compatibility
          inputData: inputChunks[0] || plan.metadata.inputData,
          outputSize: tileOutputSize,

          uniforms: {
            matrix_n: matrixSize,
            tile_start_row: startRow,
            tile_start_col: startCol,
            tile_rows: actualTileRows,
            tile_cols: actualTileCols,
            tile_size: tileSize
          },

          inputSchema: schema,
          assemblyMetadata: {
            tileRow,
            tileCol,
            startRow,
            startCol,
            tileRows: actualTileRows,
            tileCols: actualTileCols,
            matrixSize,
            tileIndex: tileIndex
          }
        });

        tileIndex++;
      }
    }

    return descriptors;
  }

  /**
   * Override chunking to replicate input data to all tiles
   */
  chunkInputs(schema, parsedInputs, chunkIndex, totalChunks, chunkingParams = {}) {
    const inputChunks = [];

    for (const inputDef of schema.inputs) {
      const inputData = parsedInputs[inputDef.name];

      if (!inputData) {
        inputChunks.push('');
        continue;
      }

      // Matrix tiled strategy always replicates input data
      // All tiles need access to the full matrices
      inputChunks.push(inputData);
    }

    return inputChunks;
  }

  /**
   * Compute workgroup count for matrix tiles
   */
  computeWorkgroupCount(chunkIndex, plan) {
    const { matrixSize, tileSize, tilesPerDim } = plan.metadata;
    const tileRow = Math.floor(chunkIndex / tilesPerDim);
    const tileCol = chunkIndex % tilesPerDim;

    const startRow = tileRow * tileSize;
    const startCol = tileCol * tileSize;
    const endRow = Math.min(startRow + tileSize, matrixSize);
    const endCol = Math.min(startCol + tileSize, matrixSize);

    const actualTileRows = endRow - startRow;
    const actualTileCols = endCol - startCol;

    return [
      Math.ceil(actualTileRows / 16),
      Math.ceil(actualTileCols / 16),
      1
    ];
  }

  /**
   * Compute output sizes for each tile
   */
  computeChunkOutputSizes(schema, metadata, chunkIndex, totalChunks) {
    const { matrixSize, tileSize, tilesPerDim } = metadata;
    const tileRow = Math.floor(chunkIndex / tilesPerDim);
    const tileCol = chunkIndex % tilesPerDim;

    const startRow = tileRow * tileSize;
    const startCol = tileCol * tileSize;
    const endRow = Math.min(startRow + tileSize, matrixSize);
    const endCol = Math.min(startCol + tileSize, matrixSize);

    const actualTileRows = endRow - startRow;
    const actualTileCols = endCol - startCol;
    const tileOutputSize = actualTileRows * actualTileCols * 4; // float32

    return [tileOutputSize];
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
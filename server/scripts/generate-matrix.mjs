#!/usr/bin/env node
// generate-matrix.mjs - Generate random matrix data for volunteer computing

import fs from 'fs/promises';
import path from 'path';

function parseArgs(argv) {
  const args = { _: [] };
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith('--')) {
      const key = a.slice(2);
      const val = (i + 1 < argv.length && !argv[i+1].startsWith('--')) ? argv[++i] : true;
      args[key] = val;
    } else {
      args._.push(a);
    }
  }
  return args;
}

function printHelp() {
  console.log(`
Matrix Data Generator for Volunteer Computing

Usage:
  node generate-matrix.mjs --size <int> [options]

Options:
  --size <int>           Matrix size (e.g., 1024 for 1024x1024 matrix)
  --output <dir>         Output directory (default: ./matrix_data)
  --format <type>        Output format: combined|separate (default: combined)
  --type <datatype>      Data type: float32|int32 (default: float32)
  --seed <int>           Random seed for reproducible results
  --range <min,max>      Value range (default: 0,1 for float32, 0,1000 for int32)

Formats:
  combined    - Single input file with both matrices (for matrix_tiled strategy)
  separate    - Two separate matrix files A and B
  inputs-json - Creates inputs.json file pointing to matrix files

Examples:
  # Generate 1024x1024 matrices for tiled multiplication
  node generate-matrix.mjs --size 1024 --output ./matrices

  # Generate with specific range and seed
  node generate-matrix.mjs --size 512 --range -10,10 --seed 42

  # Generate integer matrices
  node generate-matrix.mjs --size 256 --type int32 --range 0,100
`);
}

function generateRandomMatrix(size, dataType = 'float32', range = null, seed = null) {
  if (seed !== null) {
    // Simple seeded random number generator
    let seedValue = seed;
    Math.random = function() {
      seedValue = (seedValue * 9301 + 49297) % 233280;
      return seedValue / 233280;
    };
  }

  const matrix = [];
  let [min, max] = range || (dataType === 'float32' ? [0, 1] : [0, 1000]);
  
  for (let i = 0; i < size; i++) {
    matrix[i] = [];
    for (let j = 0; j < size; j++) {
      if (dataType === 'float32') {
        matrix[i][j] = Math.random() * (max - min) + min;
      } else {
        matrix[i][j] = Math.floor(Math.random() * (max - min + 1)) + min;
      }
    }
  }
  
  return matrix;
}

function matrixToBuffer(matrix, dataType = 'float32') {
  const size = matrix.length;
  
  if (dataType === 'float32') {
    const buffer = new ArrayBuffer(4 + size * size * 4); // size header + matrix data
    const view = new DataView(buffer);
    
    // Write size header
    view.setUint32(0, size, true);
    
    // Write matrix data
    let offset = 4;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        view.setFloat32(offset, matrix[i][j], true);
        offset += 4;
      }
    }
    
    return Buffer.from(buffer);
  } else if (dataType === 'int32') {
    const buffer = new ArrayBuffer(4 + size * size * 4);
    const view = new DataView(buffer);
    
    view.setUint32(0, size, true);
    
    let offset = 4;
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        view.setInt32(offset, matrix[i][j], true);
        offset += 4;
      }
    }
    
    return Buffer.from(buffer);
  }
}

function packCombinedMatrices(matrixA, matrixB, dataType = 'float32') {
  const size = matrixA.length;
  const elementSize = 4; // Both float32 and int32 are 4 bytes
  
  const buffer = new ArrayBuffer(4 + size * size * 2 * elementSize); // size header + A + B
  const view = new DataView(buffer);
  
  // Write size header
  view.setUint32(0, size, true);
  
  // Write matrix A
  let offset = 4;
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (dataType === 'float32') {
        view.setFloat32(offset, matrixA[i][j], true);
      } else {
        view.setInt32(offset, matrixA[i][j], true);
      }
      offset += elementSize;
    }
  }
  
  // Write matrix B
  for (let i = 0; i < size; i++) {
    for (let j = 0; j < size; j++) {
      if (dataType === 'float32') {
        view.setFloat32(offset, matrixB[i][j], true);
      } else {
        view.setInt32(offset, matrixB[i][j], true);
      }
      offset += elementSize;
    }
  }
  
  return Buffer.from(buffer);
}

async function main() {
  const [, , ...rest] = process.argv;
  const args = parseArgs(rest);
  
  if (args.help || args.h || !args.size) {
    printHelp();
    process.exit(0);
  }
  
  const size = parseInt(args.size, 10);
  if (!Number.isInteger(size) || size <= 0) {
    console.error('Error: --size must be a positive integer');
    process.exit(1);
  }
  
  if (size > 4096) {
    console.warn(`Warning: Size ${size} is very large and may consume significant memory`);
  }
  
  const outputDir = args.output || './matrix_data';
  const format = args.format || 'combined';
  const dataType = args.type || 'float32';
  const seed = args.seed ? parseInt(args.seed, 10) : null;
  
  let range = null;
  if (args.range) {
    const [min, max] = args.range.split(',').map(x => parseFloat(x));
    if (isNaN(min) || isNaN(max)) {
      console.error('Error: --range must be in format "min,max"');
      process.exit(1);
    }
    range = [min, max];
  }
  
  console.log(`Generating ${size}×${size} ${dataType} matrices...`);
  if (seed !== null) console.log(`Using seed: ${seed}`);
  if (range) console.log(`Value range: [${range[0]}, ${range[1]}]`);
  
  // Create output directory
  await fs.mkdir(outputDir, { recursive: true });
  
  // Generate matrices
  console.log('Generating matrix A...');
  const matrixA = generateRandomMatrix(size, dataType, range, seed);
  
  console.log('Generating matrix B...');
  const matrixB = generateRandomMatrix(size, dataType, range, seed ? seed + 1 : null);
  
  if (format === 'combined') {
    // Pack both matrices into single file (for matrix_tiled strategy)
    console.log('Packing combined matrices...');
    const combinedBuffer = packCombinedMatrices(matrixA, matrixB, dataType);
    const combinedPath = path.join(outputDir, `matrices_${size}x${size}_${dataType}.bin`);
    await fs.writeFile(combinedPath, combinedBuffer);
    
    // Create inputs.json for easy use
    const inputsJsonPath = path.join(outputDir, 'inputs.json');
    const inputsConfig = {
      input: path.relative(process.cwd(), combinedPath)
    };
    await fs.writeFile(inputsJsonPath, JSON.stringify(inputsConfig, null, 2));
    
    console.log(`✅ Combined matrices saved to: ${combinedPath}`);
    console.log(`✅ Inputs config saved to: ${inputsJsonPath}`);
    console.log(`\nUsage:`);
    console.log(`node scripts/submit-task.mjs matrix-tiled --size ${size} --tile-size 64 --inputs ${inputsJsonPath}`);
    
  } else if (format === 'separate') {
    // Save matrices separately
    console.log('Saving separate matrices...');
    const matrixABuffer = matrixToBuffer(matrixA, dataType);
    const matrixBBuffer = matrixToBuffer(matrixB, dataType);
    
    const matrixAPath = path.join(outputDir, `matrix_A_${size}x${size}_${dataType}.bin`);
    const matrixBPath = path.join(outputDir, `matrix_B_${size}x${size}_${dataType}.bin`);
    
    await fs.writeFile(matrixAPath, matrixABuffer);
    await fs.writeFile(matrixBPath, matrixBBuffer);
    
    // Create inputs.json
    const inputsJsonPath = path.join(outputDir, 'inputs.json');
    const inputsConfig = {
      matrixA: path.relative(process.cwd(), matrixAPath),
      matrixB: path.relative(process.cwd(), matrixBPath)
    };
    await fs.writeFile(inputsJsonPath, JSON.stringify(inputsConfig, null, 2));
    
    console.log(`✅ Matrix A saved to: ${matrixAPath}`);
    console.log(`✅ Matrix B saved to: ${matrixBPath}`);
    console.log(`✅ Inputs config saved to: ${inputsJsonPath}`);
    
  } else {
    console.error(`Error: Unknown format "${format}". Use: combined, separate`);
    process.exit(1);
  }
  
  const totalElements = size * size * 2;
  const totalSizeMB = (totalElements * 4) / (1024 * 1024);
  console.log(`\nGenerated ${totalElements} total elements (${totalSizeMB.toFixed(2)} MB)`);
}

main().catch(err => {
  console.error('Error:', err.message);
  process.exit(1);
});

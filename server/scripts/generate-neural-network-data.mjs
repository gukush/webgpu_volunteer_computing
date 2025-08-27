#!/usr/bin/env node
// generate-neural-network-data.mjs - Generate random neural network data for testing

import fs from 'fs/promises';
import path from 'path';
import { fileURLToPath } from 'url';
import minimist from 'minimist';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const args = minimist(process.argv.slice(2));

const inputSize = parseInt(args['input-size'], 10) || 784; // Default: MNIST-like input size
const hiddenSize = parseInt(args['hidden-size'], 10) || 1024; // Default: 1024 hidden neurons
const outputSize = parseInt(args['output-size'], 10) || 10; // Default: 10 output classes
const batchSize = parseInt(args['batch-size'], 10) || 100; // Default: 100 samples per batch
const outputPath = args.output || `neural_network_${inputSize}x${hiddenSize}x${outputSize}_batch${batchSize}.bin`;
const seed = parseInt(args.seed, 10) || Date.now();

if (!Number.isInteger(inputSize) || !Number.isInteger(hiddenSize) ||
    !Number.isInteger(outputSize) || !Number.isInteger(batchSize)) {
  console.error('Usage: --input-size N --hidden-size M --output-size P --batch-size Q [options]');
  console.error('');
  console.error('Required:');
  console.error('  --input-size     Input layer size (e.g., 784 for MNIST)');
  console.error('  --hidden-size    Hidden layer size (e.g., 1024)');
  console.error('  --output-size    Output layer size (e.g., 10 for classification)');
  console.error('  --batch-size     Number of samples per batch');
  console.error('');
  console.error('Options:');
  console.error('  --output         Output file path [default: auto-generated]');
  console.error('  --seed           Random seed [default: current timestamp]');
  console.error('');
  console.error('Examples:');
  console.error('  # MNIST-like network');
  console.error('  node generate-neural-network-data.mjs --input-size 784 --hidden-size 1024 --output-size 10 --batch-size 100');
  console.error('');
  console.error('  # Custom network with specific output');
  console.error('  node generate-neural-network-data.mjs --input-size 512 --hidden-size 256 --output-size 5 --batch-size 50 --output my_network.bin');
  process.exit(1);
}

// Set random seed for reproducible results
Math.seedrandom = function(seed) {
  const m = 0x80000000;
  const a = 1103515245;
  const c = 12345;
  let state = seed ? seed : Math.floor(Math.random() * (m - 1));

  return function() {
    state = (a * state + c) % m;
    return state / (m - 1);
  };
};

const random = Math.seedrandom(seed);

function generateRandomArray(size, min = -1.0, max = 1.0) {
  const array = new Float32Array(size);
  for (let i = 0; i < size; i++) {
    array[i] = min + random() * (max - min);
  }
  return array;
}

function generateRandomMatrix(rows, cols, min = -1.0, max = 1.0) {
  const array = new Float32Array(rows * cols);
  for (let i = 0; i < rows * cols; i++) {
    array[i] = min + random() * (max - min);
  }
  return array;
}

function generateXavierInitializedWeights(inputSize, outputSize) {
  const scale = Math.sqrt(2.0 / (inputSize + outputSize));
  const weights = new Float32Array(inputSize * outputSize);
  for (let i = 0; i < weights.length; i++) {
    weights[i] = (random() - 0.5) * 2 * scale;
  }
  return weights;
}

function generateXavierInitializedBiases(size) {
  const scale = Math.sqrt(2.0 / size);
  const biases = new Float32Array(size);
  for (let i = 0; i < biases.length; i++) {
    biases[i] = (random() - 0.5) * 2 * scale;
  }
  return biases;
}

async function generateNeuralNetworkData() {
  console.log(`🧠 Generating Neural Network Data`);
  console.log(`   Input Size: ${inputSize}`);
  console.log(`   Hidden Size: ${hiddenSize}`);
  console.log(`   Output Size: ${outputSize}`);
  console.log(`   Batch Size: ${batchSize}`);
  console.log(`   Random Seed: ${seed}`);
  console.log(`   Output File: ${outputPath}`);

  // Calculate sizes
  const inputDataSize = batchSize * inputSize;
  const weights1Size = hiddenSize * inputSize;
  const weights2Size = outputSize * hiddenSize;
  const biases1Size = hiddenSize;
  const biases2Size = outputSize;

  const totalElements = inputDataSize + weights1Size + weights2Size + biases1Size + biases2Size;
  const totalBytes = totalElements * 4 + 4; // +4 for header

  console.log(`\n📊 Data Sizes:`);
  console.log(`   Input Data: ${inputDataSize} elements (${Math.round(inputDataSize * 4 / 1024)}KB)`);
  console.log(`   Weights Layer 1: ${weights1Size} elements (${Math.round(weights1Size * 4 / 1024)}KB)`);
  console.log(`   Weights Layer 2: ${weights2Size} elements (${Math.round(weights2Size * 4 / 1024)}KB)`);
  console.log(`   Biases Layer 1: ${biases1Size} elements (${Math.round(biases1Size * 4 / 1024)}KB)`);
  console.log(`   Biases Layer 2: ${biases2Size} elements (${Math.round(biases2Size * 4 / 1024)}KB)`);
  console.log(`   Total: ${totalElements} elements (${Math.round(totalBytes / 1024 / 1024)}MB)`);

  // Generate data
  console.log(`\n⚙️ Generating random data...`);

  // Input data: random values between 0 and 1 (normalized input)
  const inputData = generateRandomArray(inputDataSize, 0.0, 1.0);
  console.log(`   ✅ Input data generated`);

  // Weights: Xavier initialization for better training
  const weights1 = generateXavierInitializedWeights(inputSize, hiddenSize);
  console.log(`   ✅ Weights Layer 1 generated`);

  const weights2 = generateXavierInitializedWeights(hiddenSize, outputSize);
  console.log(`   ✅ Weights Layer 2 generated`);

  // Biases: Xavier initialization
  const biases1 = generateXavierInitializedBiases(hiddenSize);
  console.log(`   ✅ Biases Layer 1 generated`);

  const biases2 = generateXavierInitializedBiases(outputSize);
  console.log(`   ✅ Biases Layer 2 generated`);

  // Pack into binary buffer
  console.log(`\n📦 Packing data into binary format...`);
  const buffer = Buffer.alloc(totalBytes);
  let offset = 0;

  // Header: network dimensions
  buffer.writeUInt32LE(inputSize, offset);
  offset += 4;

  // Input data
  buffer.set(new Uint8Array(inputData.buffer), offset);
  offset += inputDataSize * 4;

  // Weights Layer 1
  buffer.set(new Uint8Array(weights1.buffer), offset);
  offset += weights1Size * 4;

  // Weights Layer 2
  buffer.set(new Uint8Array(weights2.buffer), offset);
  offset += weights2Size * 4;

  // Biases Layer 1
  buffer.set(new Uint8Array(biases1.buffer), offset);
  offset += biases1Size * 4;

  // Biases Layer 2
  buffer.set(new Uint8Array(biases2.buffer), offset);
  offset += biases2Size * 4;

  console.log(`   ✅ Data packed (${offset} bytes)`);

  // Write to file
  console.log(`\n💾 Writing to file: ${outputPath}`);
  await fs.writeFile(outputPath, buffer);
  console.log(`   ✅ File written successfully`);

  // Verify file size
  const stats = await fs.stat(outputPath);
  if (stats.size !== totalBytes) {
    console.warn(`⚠️ Warning: Expected ${totalBytes} bytes, got ${stats.size} bytes`);
  } else {
    console.log(`   ✅ File size verified: ${stats.size} bytes`);
  }

  // Generate metadata file for reference
  const metadataPath = outputPath.replace('.bin', '.json');
  const metadata = {
    inputSize,
    hiddenSize,
    outputSize,
    batchSize,
    totalElements,
    totalBytes,
    seed,
    generatedAt: new Date().toISOString(),
    dataLayout: {
      header: '4 bytes (inputSize as uint32)',
      inputData: `${inputDataSize} elements (${inputDataSize * 4} bytes)`,
      weights1: `${weights1Size} elements (${weights1Size * 4} bytes)`,
      weights2: `${weights2Size} elements (${weights2Size * 4} bytes)`,
      biases1: `${biases1Size} elements (${biases1Size * 4} bytes)`,
      biases2: `${biases2Size} elements (${biases2Size * 4} bytes)`
    },
    usage: {
      description: 'Simple 2-layer neural network with ReLU activation',
      inputFormat: 'Normalized values between 0 and 1',
      weightInitialization: 'Xavier/Glorot initialization',
      activationFunction: 'ReLU for hidden layer, linear for output layer'
    }
  };

  await fs.writeFile(metadataPath, JSON.stringify(metadata, null, 2));
  console.log(`   ✅ Metadata written to: ${metadataPath}`);

  console.log(`\n🎉 Neural network data generation completed!`);
  console.log(`\n📋 Usage:`);
  console.log(`   # Test with the volunteer computing system:`);
  console.log(`   node test-neural-network.mjs --input-size ${inputSize} --hidden-size ${hiddenSize} --output-size ${outputSize} --batch-size ${batchSize} --input ${outputPath}`);
  console.log(`\n   # Or use the generated data directly in your neural network implementation`);
  console.log(`\n🔍 File Contents:`);
  console.log(`   Binary data: ${outputPath}`);
  console.log(`   Metadata: ${metadataPath}`);
  console.log(`   Total size: ${Math.round(totalBytes / 1024 / 1024 * 100) / 100} MB`);

  return {
    inputSize,
    hiddenSize,
    outputSize,
    batchSize,
    totalElements,
    totalBytes,
    outputPath,
    metadataPath
  };
}

// Run the generator
generateNeuralNetworkData().catch(error => {
  console.error('\n❌ Generation failed:', error.message);
  process.exit(1);
});

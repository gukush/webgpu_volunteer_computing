#!/usr/bin/env node
import fs from 'fs/promises';
import path from 'path';
import minimist from 'minimist';

const args = minimist(process.argv.slice(2));
const count = parseInt(args.count, 10);
const output = args.output || `unsorted_data_${count}.bin`;

if (!Number.isInteger(count) || count <= 0) {
  console.error('Usage: node generate-sort-data.mjs --count <number_of_elements> [--output <filename>]');
  process.exit(1);
}

async function generateData() {
  console.log(`🚀 Generating ${count} random float32 numbers...`);

  // Each float32 is 4 bytes. We add 4 bytes for the header (element count).
  const buffer = Buffer.alloc(4 + count * 4);

  // Write header: total number of elements
  buffer.writeUInt32LE(count, 0);

  // Write random float32 data
  let offset = 4;
  for (let i = 0; i < count; i++) {
    buffer.writeFloatLE(Math.random() * 1000, offset);
    offset += 4;
  }

  const outputPath = path.resolve(output);
  await fs.writeFile(outputPath, buffer);

  console.log(`✅ Data successfully written to: ${outputPath}`);
  console.log(`   Total size: ${buffer.length} bytes`);
}

generateData().catch(err => {
  console.error('❌ Error generating data:', err);
  process.exit(1);
});
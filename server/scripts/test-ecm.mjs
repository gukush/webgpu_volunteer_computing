#!/usr/bin/env node
// test-ecm.mjs - Test script for ECM Stage 1 strategy
// Tests WGSL code generation and chunk creation without requiring client execution

import { fileURLToPath } from 'url';
import path from 'path';
import minimist from 'minimist';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Add the parent directory to the module path so we can import the strategy
import { createRequire } from 'module';
const require = createRequire(import.meta.url);

// Import the ECM strategy
const ECMStage1ChunkingStrategy = (await import('../strategies/ECMStage1ChunkingStrategy.js')).default;

const args = minimist(process.argv.slice(2));

const n = args.n || '18446744065119617077';
const B1 = parseInt(args.B1, 10) || 50000;
const curvesTotal = parseInt(args.curvesTotal, 10) || 1024;
const curvesPerChunk = parseInt(args.curvesPerChunk, 10) || 256;
const framework = args.framework || 'webgpu';

console.log('=== ECM Stage 1 Strategy Test ===');
console.log(`n: ${n}`);
console.log(`B1: ${B1}`);
console.log(`curvesTotal: ${curvesTotal}`);
console.log(`curvesPerChunk: ${curvesPerChunk}`);
console.log(`framework: ${framework}`);
console.log('');

// Test 1: Create strategy instance
console.log('1. Creating strategy instance...');
const strategy = new ECMStage1ChunkingStrategy();
console.log('✅ Strategy created successfully');
console.log('');

// Test 2: Validate workload
console.log('2. Validating workload...');
const workload = {
  metadata: { n, B1, curvesTotal, curvesPerChunk },
  framework
};

const validation = strategy.validateWorkload(workload);
if (validation.valid) {
  console.log('✅ Workload validation passed');
} else {
  console.log('❌ Workload validation failed:', validation.error);
  process.exit(1);
}
console.log('');

// Test 3: Plan execution
console.log('3. Planning execution...');
const plan = strategy.planExecution(workload);
console.log('✅ Execution plan created:', {
  totalChunks: plan.totalChunks,
  chunkingStrategy: plan.chunkingStrategy,
  assemblyStrategy: plan.assemblyStrategy
});
console.log('');

// Test 4: Create common inputs
console.log('4. Creating common inputs...');
const commonInputs = await strategy.createCommonInputs(plan);
console.log('✅ Common inputs created:', {
  numPrimePowers: commonInputs.numPrimePowers,
  peBase64Length: commonInputs.peBase64.length,
  outInitB64Length: commonInputs.outInitB64.length
});
console.log('');

// Test 5: Test WGSL code generation
console.log('5. Testing WGSL code generation...');
try {
  // Access the WGSL constant directly
  const WGSL_ECM_STAGE1 = strategy.constructor.name === 'ECMStage1ChunkingStrategy'
    ? require('../strategies/ECMStage1ChunkingStrategy.js').WGSL_ECM_STAGE1
    : null;

  if (WGSL_ECM_STAGE1) {
    console.log('✅ WGSL constant found');
    console.log(`   Length: ${WGSL_ECM_STAGE1.length} characters`);
    console.log(`   Starts with: ${WGSL_ECM_STAGE1.substring(0, 50)}...`);
    console.log(`   Ends with: ...${WGSL_ECM_STAGE1.substring(WGSL_ECM_STAGE1.length - 50)}`);

    // Check for basic WGSL syntax
    if (WGSL_ECM_STAGE1.includes('struct') && WGSL_ECM_STAGE1.includes('fn main')) {
      console.log('✅ Basic WGSL structure looks correct');
    } else {
      console.log('❌ Basic WGSL structure missing key elements');
    }

    // Check for balanced braces
    const openBraces = (WGSL_ECM_STAGE1.match(/\{/g) || []).length;
    const closeBraces = (WGSL_ECM_STAGE1.match(/\}/g) || []).length;
    console.log(`   Braces: ${openBraces} open, ${closeBraces} close`);

    if (openBraces === closeBraces) {
      console.log('✅ Braces are balanced');
    } else {
      console.log('❌ Braces are unbalanced!');
    }

    // Check for balanced parentheses
    const openParens = (WGSL_ECM_STAGE1.match(/\(/g) || []).length;
    const closeParens = (WGSL_ECM_STAGE1.match(/\)/g) || []).length;
    console.log(`   Parentheses: ${openParens} open, ${closeParens} close`);

    if (openParens === closeParens) {
      console.log('✅ Parentheses are balanced');
    } else {
      console.log('❌ Parentheses are unbalanced!');
    }

  } else {
    console.log('❌ WGSL constant not found');
  }
} catch (error) {
  console.log('❌ Error accessing WGSL code:', error.message);
}
console.log('');

// Test 6: Test chunk descriptor creation
console.log('6. Testing chunk descriptor creation...');
try {
  let chunkCount = 0;
  const chunks = [];

  await strategy.createChunkDescriptorsStreaming(plan, (descriptor) => {
    chunks.push(descriptor);
    chunkCount++;

    if (chunkCount <= 2) { // Only show first 2 chunks to avoid spam
      console.log(`   Chunk ${chunkCount}:`, {
        chunkId: descriptor.chunkId,
        framework: descriptor.framework,
        hasKernel: !!descriptor.kernel,
        hasWgsl: !!descriptor.wgsl,
        entry: descriptor.entry,
        workgroupCount: descriptor.workgroupCount,
        inputsCount: descriptor.inputs?.length || 0,
        outputsCount: descriptor.outputs?.length || 0
      });

      if (descriptor.kernel) {
        console.log(`     Kernel length: ${descriptor.kernel.length} characters`);
        console.log(`     Kernel starts: ${descriptor.kernel.substring(0, 100)}...`);
      }
    }
  });

  console.log(`✅ Created ${chunkCount} chunks successfully`);

  if (chunks.length > 0) {
    const firstChunk = chunks[0];
    console.log('   First chunk structure:', Object.keys(firstChunk));

    // Verify the kernel field is present
    if (firstChunk.kernel) {
      console.log('✅ Kernel field present in chunk descriptor');
    } else {
      console.log('❌ Kernel field missing from chunk descriptor');
    }
  }

} catch (error) {
  console.log('❌ Error creating chunk descriptors:', error.message);
  console.log('   Stack trace:', error.stack);
}
console.log('');

// Test 7: Test specific WGSL syntax issues
console.log('7. Testing for known WGSL syntax issues...');
try {
  const WGSL_ECM_STAGE1 = require('../strategies/ECMStage1ChunkingStrategy.js').WGSL_ECM_STAGE1;

  // Check for semicolons in struct fields (should be commas)
  const semicolonInStruct = WGSL_ECM_STAGE1.match(/struct\s+\w+\s*\{[^}]*\w+:\s*\w+;/);
  if (semicolonInStruct) {
    console.log('❌ Found semicolon in struct field (should be comma)');
    console.log('   Context:', semicolonInStruct[0]);
  } else {
    console.log('✅ No semicolons in struct fields');
  }

  // Check for missing commas in function calls
  const vec2Call = WGSL_ECM_STAGE1.match(/vec2<u32>\([^)]*\)/g);
  if (vec2Call) {
    console.log('✅ Found vec2<u32> calls');
    vec2Call.forEach((call, i) => {
      if (call.includes(',')) {
        console.log(`   Call ${i + 1}: ✅ Has comma`);
      } else {
        console.log(`   Call ${i + 1}: ❌ Missing comma`);
      }
    });
  }

  // Check for atomic syntax
  const atomicUsage = WGSL_ECM_STAGE1.match(/atomic<[^>]+>/g);
  if (atomicUsage) {
    console.log('❌ Found atomic syntax (not supported in this WebGPU version)');
    atomicUsage.forEach(atomic => console.log(`   ${atomic}`));
  } else {
    console.log('✅ No atomic syntax found');
  }

} catch (error) {
  console.log('❌ Error in syntax checking:', error.message);
}
console.log('');

console.log('=== Test Complete ===');
console.log('');
console.log('If all tests passed, the issue might be:');
console.log('1. Client-side WebGPU implementation differences');
console.log('2. Runtime execution errors (not syntax errors)');
console.log('3. Browser-specific WGSL parsing differences');
console.log('');
console.log('Check the Chrome console for additional error details.');



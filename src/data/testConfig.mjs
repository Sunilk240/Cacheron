// Quick test script — run with: node src/data/testConfig.mjs
// Verifies all formulas against the pre-computed table in the master design doc

import {
    MODELS, GPUS,
    kvCacheSize, modelWeightSize, totalMemory, fits, formatBytes, tileSize
} from './modelConfig.js';

const T = 2048;
const B = 1;
const FP16 = 2;
const INT4 = 0.5;

console.log('='.repeat(80));
console.log('CACHERON — modelConfig.js Verification');
console.log('='.repeat(80));

// --- KV Cache at T=2048, B=1, FP16 ---
console.log('\n--- KV Cache (FP16, T=2048, B=1) ---');
console.log('Expected from master doc:');
console.log('  SmolLM2-135M: 45 MB');
console.log('  SmolLM2-360M: 80 MB');
console.log('  Gemma-3-1B:   ~52 MB (hybrid)');
console.log('  Llama-3.2-3B: 224 MB');
console.log('  Llama-2-7B:   1,024 MB');
console.log('');
console.log('Computed:');
for (const [key, model] of Object.entries(MODELS)) {
    const kv = kvCacheSize(model, T, B, FP16);
    console.log(`  ${model.name}: ${formatBytes(kv)} (${kv} bytes)`);
}

// --- Model Weights (FP16) ---
console.log('\n--- Model Weights (FP16) ---');
console.log('Expected: 270 MB, 720 MB, 2000 MB, 6000 MB, 14000 MB');
console.log('');
console.log('Computed:');
for (const [key, model] of Object.entries(MODELS)) {
    const w = modelWeightSize(model, FP16);
    console.log(`  ${model.name}: ${formatBytes(w)}`);
}

// --- Total FP16 ---
console.log('\n--- Total (FP16 weights + FP16 KV, T=2048) ---');
console.log('Expected: 315, 800, ~2052, 6224, 15024 MB');
console.log('');
console.log('Computed:');
for (const [key, model] of Object.entries(MODELS)) {
    const t = totalMemory(model, T, B, FP16, FP16);
    console.log(`  ${model.name}: ${formatBytes(t)}`);
}

// --- Fits/Doesn't Fit (FP16) ---
console.log('\n--- Fits Table (FP16) ---');
const header = 'Model'.padEnd(20) + Object.values(GPUS).map(g => g.name.padEnd(18)).join('');
console.log(header);
for (const [key, model] of Object.entries(MODELS)) {
    const row = model.name.padEnd(20) + Object.values(GPUS).map(gpu => {
        const f = fits(model, gpu, T, B, FP16, FP16);
        const t = totalMemory(model, T, B, FP16, FP16);
        return (f ? '✅' : '❌').padEnd(3) + formatBytes(t).padEnd(15);
    }).join('');
    console.log(row);
}

// --- Fits/Doesn't Fit (INT4 weights + FP16 KV) ---
console.log('\n--- Fits Table (INT4 weights + FP16 KV) ---');
console.log(header);
for (const [key, model] of Object.entries(MODELS)) {
    const row = model.name.padEnd(20) + Object.values(GPUS).map(gpu => {
        const f = fits(model, gpu, T, B, INT4, FP16);
        const t = totalMemory(model, T, B, INT4, FP16);
        return (f ? '✅' : '❌').padEnd(3) + formatBytes(t).padEnd(15);
    }).join('');
    console.log(row);
}

// --- Tile Sizes per GPU ---
console.log('\n--- Flash Attention Tile Sizes ---');
for (const [gkey, gpu] of Object.entries(GPUS)) {
    const sizes = Object.values(MODELS).map(m => `${m.name}:${tileSize(gpu, m)}`).join(', ');
    console.log(`  ${gpu.name}: ${sizes}`);
}

console.log('\n' + '='.repeat(80));
console.log('Verification complete. Compare above with master design doc tables.');
console.log('='.repeat(80));

// ============================================================
// modelConfig.js — Single source of truth for Cacheron
// ============================================================
// Every formula, every memory bar, every "fits / doesn't fit"
// verdict on the entire site is computed from this file.
// If these numbers are wrong, EVERYTHING is wrong.
// ============================================================

// ----- 5 MODELS -----
// Each model's actual architecture dimensions.
// Sources: HuggingFace model cards, official papers.

export const MODELS = {
    'smollm2-135m': {
        name: 'SmolLM2-135M',
        params: 135e6,
        L: 30,        // layers
        Hq: 9,        // query heads
        Hkv: 3,       // KV heads (GQA 3:1)
        dhead: 64,    // head dimension
        dmodel: 576,  // model dimension
        attnType: 'GQA 3:1',
    },
    'smollm2-360m': {
        name: 'SmolLM2-360M',
        params: 360e6,
        L: 32,
        Hq: 15,
        Hkv: 5,
        dhead: 64,
        dmodel: 960,
        attnType: 'GQA 3:1',
    },
    'gemma-3-1b': {
        name: 'Gemma-3-1B',
        params: 1e9,
        L: 26,
        Hq: 8,
        Hkv: 4,
        dhead: 256,
        dmodel: 1152,
        attnType: 'GQA + Hybrid',
        // SPECIAL CASE: Gemma-3-1B uses 5:1 local/global hybrid attention.
        // Only global layers (every 6th) use full-context KV cache.
        // Local layers use sliding window of 1024 tokens.
        // If you use the standard formula, KV cache will be ~5× too large
        // and the fits table will show a false ❌ on iPhone.
        hybrid: true,
        localWindow: 1024,
        globalEvery: 6, // 1 global layer per 6 layers
    },
    'llama-3.2-3b': {
        name: 'Llama-3.2-3B',
        params: 3e9,
        L: 28,
        Hq: 24,
        Hkv: 8,
        dhead: 128,
        dmodel: 3072,
        attnType: 'GQA 3:1',
    },
    'llama-2-7b': {
        name: 'Llama-2-7B',
        params: 7e9,
        L: 32,
        Hq: 32,
        Hkv: 32,       // Pure MHA — H_kv = H_q
        dhead: 128,
        dmodel: 4096,
        attnType: 'Pure MHA',
        isMHA: true,    // The "villain" — no GQA savings
    },
};

// ----- 4 GPUS / DEVICES -----

export const GPUS = {
    'rpi4': {
        name: 'Raspberry Pi 4',
        budget_mb: 1500,     // ~1.5 GB usable
        sram_mb: 1,          // ~1 MB L1+L2 cache
        bandwidth_gbs: 25,   // ~25 GB/s RAM bandwidth
    },
    'iphone-15p': {
        name: 'iPhone 15 Pro',
        budget_mb: 3000,     // ~3 GB usable
        sram_mb: 32,         // ~32 MB L2 cache
        bandwidth_gbs: 68,   // ~68 GB/s unified memory
    },
    'rtx-3060': {
        name: 'RTX 3060',
        budget_mb: 12000,    // 12 GB VRAM
        sram_mb: 3,          // ~3 MB L2 cache
        bandwidth_gbs: 360,  // 360 GB/s
    },
    'a100-40': {
        name: 'A100 40GB',
        budget_mb: 40000,    // 40 GB VRAM
        sram_mb: 20,         // ~20 MB SRAM across SMs
        bandwidth_gbs: 1555, // 1,555 GB/s HBM2e
    },
};

// ----- DEFAULTS -----

export const DEFAULT_MODEL = 'llama-3.2-3b';
export const DEFAULT_GPU = 'rtx-3060';

// ============================================================
// CORE FORMULAS
// ============================================================

/**
 * KV Cache size in bytes for standard (non-hybrid) models.
 * Formula: 2 × L × H_kv × d_head × T × B × precision_bytes
 *
 * The "2" accounts for both K and V vectors.
 */
export function kvCacheSize(model, tokens, batch = 1, precisionBytes = 2) {
    if (model.hybrid) {
        return kvCacheSizeHybrid(model, tokens, batch, precisionBytes);
    }
    return 2 * model.L * model.Hkv * model.dhead * tokens * batch * precisionBytes;
}

/**
 * KV Cache size for Gemma-3-1B hybrid attention.
 * Only global layers (every 6th) contribute full KV cache.
 * Local layers use a sliding window (capped at localWindow tokens).
 */
function kvCacheSizeHybrid(model, tokens, batch = 1, precisionBytes = 2) {
    const globalLayers = Math.ceil(model.L / model.globalEvery);
    const localLayers = model.L - globalLayers;
    const effectiveLocalTokens = Math.min(tokens, model.localWindow);

    const globalKV = 2 * globalLayers * model.Hkv * model.dhead * tokens * batch * precisionBytes;
    const localKV = 2 * localLayers * model.Hkv * model.dhead * effectiveLocalTokens * batch * precisionBytes;

    return globalKV + localKV;
}

/**
 * Model weight size in bytes.
 * Formula: params × precision_bytes
 */
export function modelWeightSize(model, precisionBytes = 2) {
    return model.params * precisionBytes;
}

/**
 * Total memory = model weights + KV cache.
 */
export function totalMemory(model, tokens, batch = 1, weightPrecision = 2, kvPrecision = 2) {
    return modelWeightSize(model, weightPrecision) + kvCacheSize(model, tokens, batch, kvPrecision);
}

/**
 * Does this model fit on this GPU?
 */
export function fits(model, gpu, tokens, batch = 1, weightPrecision = 2, kvPrecision = 2) {
    const totalBytes = totalMemory(model, tokens, batch, weightPrecision, kvPrecision);
    const budgetBytes = gpu.budget_mb * 1024 * 1024;
    return totalBytes <= budgetBytes;
}

/**
 * Flash Attention tile size for a given GPU + model.
 * Tile must fit: Q_tile + K_tile + V_tile + S_tile + O_tile in SRAM.
 */
export function tileSize(gpu, model, precisionBytes = 2) {
    const sramBytes = gpu.sram_mb * 1024 * 1024;
    const dhead = model.dhead;
    // block_size ≈ sqrt(sram_bytes / ((2 + 4 × dhead) × precision_bytes))
    return Math.floor(Math.sqrt(sramBytes / ((2 + 4 * dhead) * precisionBytes)));
}

// ============================================================
// HELPER FORMATTERS
// ============================================================

/**
 * Format bytes into human-readable KB, MB, or GB.
 * Always shows 2 decimal places for MB to avoid "0 MB" display.
 */
export function formatBytes(bytes) {
    const mb = bytes / (1024 * 1024);
    if (mb >= 1024) {
        return `${(mb / 1024).toFixed(2)} GB`;
    }
    if (mb < 0.01) {
        const kb = bytes / 1024;
        return `${kb.toFixed(2)} KB`;
    }
    return `${mb.toFixed(2)} MB`;
}

/**
 * KV cache per token per layer in bytes (for a specific attention mode).
 * Useful for showing step-by-step KV cache derivation.
 * kvHeads parameter allows overriding for MHA/GQA/MQA comparison.
 */
export function kvPerTokenPerLayer(model, kvHeads = null, precisionBytes = 2) {
    const hkv = kvHeads !== null ? kvHeads : model.Hkv;
    return 2 * hkv * model.dhead * precisionBytes;
}

/**
 * KV cache per token across all layers.
 */
export function kvPerToken(model, kvHeads = null, precisionBytes = 2) {
    return kvPerTokenPerLayer(model, kvHeads, precisionBytes) * model.L;
}

/**
 * Get all model entries as an array of [key, model] pairs.
 */
export function getModelEntries() {
    return Object.entries(MODELS);
}

/**
 * Get all GPU entries as an array of [key, gpu] pairs.
 */
export function getGPUEntries() {
    return Object.entries(GPUS);
}

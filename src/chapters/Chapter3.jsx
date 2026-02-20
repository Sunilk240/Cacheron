import { useState, useMemo } from 'react';
import { MODELS, GPUS, tileSize, formatBytes } from '../data/modelConfig';
import './Chapter3.css';

// ============================================================
// 1. GPU MEMORY HIERARCHY
//    Pyramid: SRAM (fast, tiny) → HBM (medium) → DRAM (slow, huge)
// ============================================================

function MemoryHierarchy({ gpu }) {
    const tiers = [
        {
            name: 'SRAM (On-chip)',
            size: `${gpu.sram_mb} MB`,
            speed: '~19 TB/s',
            ratio: '1×',
            desc: 'Registers + L1/L2 cache. Blazing fast, tiny capacity.',
            cls: 'sram-tier',
            icon: '⚡',
        },
        {
            name: 'HBM / VRAM',
            size: `${(gpu.budget_mb / 1024).toFixed(0)} GB`,
            speed: `${gpu.bandwidth_gbs} GB/s`,
            ratio: `~${Math.round(19000 / gpu.bandwidth_gbs)}× slower`,
            desc: 'Main GPU memory. Where model weights and KV cache live.',
            cls: 'hbm-tier',
            icon: '🧠',
        },
        {
            name: 'CPU DRAM',
            size: '16-64 GB',
            speed: '~50 GB/s',
            ratio: `~${Math.round(19000 / 50)}× slower`,
            desc: 'System RAM. Used for offloading when GPU memory is full.',
            cls: 'dram-tier',
            icon: '💾',
        },
    ];

    return (
        <section className="chapter-section">
            <h3 className="section-title">The GPU Memory Hierarchy — Speed vs Capacity</h3>
            <p className="section-desc">
                To understand why Flash Attention matters, you need to understand the GPU's memory hierarchy.
                GPUs have a small but <strong>extremely fast</strong> on-chip SRAM (registers and L1/L2 cache)
                and a large but <strong>much slower</strong> HBM (High Bandwidth Memory, the main VRAM).
                The speed difference is enormous — SRAM is roughly <strong>50-100×</strong> faster than HBM.
            </p>
            <p className="section-desc">
                Standard attention computes everything in HBM — reading Q, K, V from HBM, writing the
                intermediate attention matrix back to HBM, then reading it again. Flash Attention's key
                insight is: <strong>do all the work in SRAM and never materialize the full attention matrix</strong>.
            </p>

            <div className="mem-hierarchy glass-card">
                <div className="mem-pyramid">
                    {tiers.map((tier, i) => (
                        <div key={i} className={`mem-tier ${tier.cls}`}>
                            <div className="mem-tier-icon">{tier.icon}</div>
                            <div className="mem-tier-info">
                                <div className="mem-tier-name">{tier.name}</div>
                                <div className="mem-tier-specs">{tier.size} — {tier.desc}</div>
                            </div>
                            <div className="mem-tier-speed">
                                {tier.speed}
                                <div className="mem-tier-ratio">{tier.ratio}</div>
                            </div>
                        </div>
                    ))}
                </div>
                <div style={{ textAlign: 'center', fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>
                    {gpu.name}: {gpu.sram_mb} MB SRAM, {(gpu.budget_mb / 1024).toFixed(0)} GB HBM at {gpu.bandwidth_gbs} GB/s
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 2. STANDARD vs FLASH ATTENTION — Side-by-side IO flow
// ============================================================

function StandardVsFlash({ model, gpu }) {
    const seqLen = 1024;
    const d = model.dhead;

    // Standard attention IO (simplified)
    // Reads: Q, K, V from HBM (3 × N × d)
    // Writes: S = Q×K^T to HBM (N × N)
    // Reads: S from HBM
    // Writes: P = softmax(S) to HBM (N × N)
    // Reads: P, V from HBM
    // Writes: O to HBM (N × d)
    const standardReads = 3 * seqLen * d + seqLen * seqLen + seqLen * seqLen + seqLen * d;
    const standardWrites = seqLen * seqLen + seqLen * seqLen + seqLen * d;
    const standardTotal = standardReads + standardWrites;

    // Flash attention: reads Q,K,V once, writes O once. No intermediate materialization.
    const flashReads = 3 * seqLen * d;
    const flashWrites = seqLen * d;
    const flashTotal = flashReads + flashWrites;

    const savings = ((1 - flashTotal / standardTotal) * 100).toFixed(0);

    return (
        <section className="chapter-section">
            <h3 className="section-title">Standard Attention vs Flash Attention — IO Comparison</h3>
            <p className="section-desc">
                The critical difference is how many times data travels between slow HBM and fast SRAM.
                Standard attention writes and reads the full <strong>N×N attention matrix</strong> to/from HBM
                multiple times. Flash Attention processes tiles entirely in SRAM, reading Q/K/V from HBM only
                once and writing the output once. For a sequence of {seqLen} tokens:
            </p>

            <div className="attn-compare glass-card">
                <div className="attn-compare-grid">
                    {/* Standard */}
                    <div className="attn-flow-col">
                        <div className="attn-flow-title">
                            🐌 Standard Attention
                        </div>
                        <div className="attn-flow-subtitle">
                            Materializes the full N×N attention matrix in HBM. Multiple round-trips between HBM and compute.
                        </div>
                        <div className="flow-steps">
                            <div className="flow-step">
                                <div className="flow-step-num">1</div>
                                <div className="flow-step-content">
                                    <div className="flow-step-action">Load Q, K from HBM</div>
                                    <div className="flow-step-io read">↑ READ {seqLen}×{d} × 2</div>
                                </div>
                            </div>
                            <div className="flow-step">
                                <div className="flow-step-num">2</div>
                                <div className="flow-step-content">
                                    <div className="flow-step-action">Compute S = Q × Kᵀ</div>
                                    <div className="flow-step-io write">↓ WRITE {seqLen}×{seqLen} to HBM</div>
                                    <div className="flow-step-detail">Full N×N matrix materialized!</div>
                                </div>
                            </div>
                            <div className="flow-step">
                                <div className="flow-step-num">3</div>
                                <div className="flow-step-content">
                                    <div className="flow-step-action">Read S, compute softmax(S)</div>
                                    <div className="flow-step-io read">↑ READ {seqLen}×{seqLen}</div>
                                    <div className="flow-step-io write">↓ WRITE {seqLen}×{seqLen}</div>
                                </div>
                            </div>
                            <div className="flow-step">
                                <div className="flow-step-num">4</div>
                                <div className="flow-step-content">
                                    <div className="flow-step-action">Read P, V → compute O = P × V</div>
                                    <div className="flow-step-io read">↑ READ {seqLen}×{seqLen} + {seqLen}×{d}</div>
                                    <div className="flow-step-io write">↓ WRITE {seqLen}×{d}</div>
                                </div>
                            </div>
                        </div>
                        <div className="io-counter standard-io">
                            <div className="io-counter-value">{(standardTotal / 1e6).toFixed(1)}M</div>
                            <div className="io-counter-label">total values transferred to/from HBM</div>
                        </div>
                    </div>

                    {/* Flash */}
                    <div className="attn-flow-col flash-col">
                        <div className="attn-flow-title">
                            ⚡ Flash Attention
                        </div>
                        <div className="attn-flow-subtitle">
                            Tiles Q×K^T into SRAM-sized blocks. Never materializes the full N×N matrix.
                        </div>
                        <div className="flow-steps">
                            <div className="flow-step">
                                <div className="flow-step-num">1</div>
                                <div className="flow-step-content">
                                    <div className="flow-step-action">Load Q, K, V tiles from HBM</div>
                                    <div className="flow-step-io read">↑ READ 3×{seqLen}×{d} (once)</div>
                                </div>
                            </div>
                            <div className="flow-step">
                                <div className="flow-step-num">2</div>
                                <div className="flow-step-content">
                                    <div className="flow-step-action">Compute tile of S, softmax, × V in SRAM</div>
                                    <div className="flow-step-io sram-op">⚡ ALL in SRAM — no HBM write</div>
                                    <div className="flow-step-detail">Uses online softmax to update running result</div>
                                </div>
                            </div>
                            <div className="flow-step">
                                <div className="flow-step-num">3</div>
                                <div className="flow-step-content">
                                    <div className="flow-step-action">Write final output O to HBM</div>
                                    <div className="flow-step-io write">↓ WRITE {seqLen}×{d} (once)</div>
                                </div>
                            </div>
                            <div className="flow-step">
                                <div className="flow-step-num">✓</div>
                                <div className="flow-step-content">
                                    <div className="flow-step-action">No intermediate N×N matrix ever stored</div>
                                    <div className="flow-step-detail">The attention matrix is computed tile-by-tile and immediately consumed</div>
                                </div>
                            </div>
                        </div>
                        <div className="io-counter flash-io">
                            <div className="io-counter-value">{(flashTotal / 1e6).toFixed(1)}M</div>
                            <div className="io-counter-label">total values transferred — {savings}% less IO</div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 3. TILE SIZE CALCULATOR
//    Shows how big each tile can be given SRAM budget
// ============================================================

function TileCalculator({ model, gpu }) {
    const blockSize = tileSize(gpu, model);
    const sramBytes = gpu.sram_mb * 1024 * 1024;
    const d = model.dhead;

    // What fits in one tile
    const tileQMem = blockSize * d * 2; // Q tile
    const tileKMem = blockSize * d * 2; // K tile
    const tileVMem = blockSize * d * 2; // V tile
    const tileSMem = blockSize * blockSize * 2; // S tile (attention scores)
    const tileOMem = blockSize * d * 2; // O tile
    const totalTileMem = tileQMem + tileKMem + tileVMem + tileSMem + tileOMem;

    // Number of tiles needed to cover full sequence
    const seqLen = 1024;
    const numTiles = Math.ceil(seqLen / blockSize);

    // Mini visual: grid showing tiles
    const tileGridSize = Math.min(blockSize, 12);

    return (
        <section className="chapter-section">
            <h3 className="section-title">How Big Can Each Tile Be?</h3>
            <p className="section-desc">
                The tile size is determined by how much SRAM is available. Each tile must fit:
                a block of Q, a block of K, a block of V, the local attention scores (S), and the
                output block (O) — all simultaneously in SRAM. A bigger tile means fewer round-trips
                to HBM but requires more SRAM.
            </p>

            <div className="tile-calc glass-card">
                <div className="tile-calc-grid">
                    <div className="tile-params">
                        <div className="tile-param-row">
                            <span className="tile-param-label">GPU</span>
                            <span className="tile-param-value">{gpu.name}</span>
                        </div>
                        <div className="tile-param-row">
                            <span className="tile-param-label">Available SRAM</span>
                            <span className="tile-param-value">{gpu.sram_mb} MB ({formatBytes(sramBytes)})</span>
                        </div>
                        <div className="tile-param-row">
                            <span className="tile-param-label">Head dimension (d)</span>
                            <span className="tile-param-value">{d}</span>
                        </div>
                        <div className="tile-param-row" style={{ borderLeft: '3px solid var(--accent-warm)', paddingLeft: 'var(--space-sm)' }}>
                            <span className="tile-param-label">Q tile</span>
                            <span className="tile-param-value">[{blockSize}×{d}] = {formatBytes(tileQMem)}</span>
                        </div>
                        <div className="tile-param-row" style={{ borderLeft: '3px solid var(--text-accent)', paddingLeft: 'var(--space-sm)' }}>
                            <span className="tile-param-label">K tile</span>
                            <span className="tile-param-value">[{blockSize}×{d}] = {formatBytes(tileKMem)}</span>
                        </div>
                        <div className="tile-param-row" style={{ borderLeft: '3px solid var(--accent-secondary)', paddingLeft: 'var(--space-sm)' }}>
                            <span className="tile-param-label">V tile</span>
                            <span className="tile-param-value">[{blockSize}×{d}] = {formatBytes(tileVMem)}</span>
                        </div>
                        <div className="tile-param-row" style={{ borderLeft: '3px solid var(--accent-gold)', paddingLeft: 'var(--space-sm)' }}>
                            <span className="tile-param-label">S tile (scores)</span>
                            <span className="tile-param-value">[{blockSize}×{blockSize}] = {formatBytes(tileSMem)}</span>
                        </div>
                        <div className="tile-param-row">
                            <span className="tile-param-label">Total per tile</span>
                            <span className="tile-param-value">{formatBytes(totalTileMem)}</span>
                        </div>
                        <div className="tile-param-row">
                            <span className="tile-param-label">Tiles for 1K seq</span>
                            <span className="tile-param-value">{numTiles} × {numTiles} = {numTiles * numTiles} tile pairs</span>
                        </div>
                    </div>

                    <div className="tile-result">
                        <div className="tile-result-value">{blockSize}</div>
                        <div className="tile-result-label">
                            tokens per tile
                            <br />
                            <strong>{model.name} on {gpu.name}</strong>
                        </div>
                        {/* Mini tile grid */}
                        <div className="tile-visual" style={{ gridTemplateColumns: `repeat(${Math.min(numTiles, 16)}, 1fr)` }}>
                            {Array.from({ length: Math.min(numTiles * numTiles, 256) }, (_, i) => (
                                <div
                                    key={i}
                                    className={`tile-cell ${i % (numTiles + 1) === 0 ? 'active' : ''}`}
                                    title={`Tile [${Math.floor(i / numTiles)}, ${i % numTiles}]`}
                                />
                            ))}
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 4. ONLINE SOFTMAX — The Mathematical Trick
// ============================================================

function OnlineSoftmax() {
    return (
        <section className="chapter-section">
            <h3 className="section-title">The Online Softmax Trick — Computing Softmax Without Seeing All Values</h3>
            <p className="section-desc">
                The standard softmax requires seeing <em>all</em> scores before normalizing (you need the
                global max for numerical stability and the global sum for the denominator). This seems to
                require materializing the full N×N attention matrix. Flash Attention solves this with
                <strong> online softmax</strong> — a clever way to update the softmax incrementally as each
                tile is processed.
            </p>

            <div className="softmax-section glass-card">
                <div className="softmax-comparison">
                    <div className="softmax-card">
                        <div className="softmax-card-title">Standard Softmax (2-pass)</div>
                        <div className="softmax-formula">
                            softmax(x<sub>i</sub>) = e<sup>x<sub>i</sub> - max(x)</sup> / Σ e<sup>x<sub>j</sub> - max(x)</sup>
                        </div>
                        <div className="softmax-steps">
                            <ol>
                                <li><strong>Pass 1:</strong> Scan all scores to find the global <code>max</code></li>
                                <li><strong>Pass 2:</strong> Compute <code>e^(x-max)</code> for each and sum them</li>
                                <li><strong>Pass 3:</strong> Divide each by the sum to normalize</li>
                                <li><strong>Problem:</strong> Requires the entire N×N score matrix in memory</li>
                            </ol>
                        </div>
                    </div>

                    <div className="softmax-card" style={{ borderColor: 'rgba(78, 205, 196, 0.25)' }}>
                        <div className="softmax-card-title">Online Softmax (1-pass, tiled)</div>
                        <div className="softmax-formula">
                            m' = max(m, max(x<sub>tile</sub>)), d' = d × e<sup>m-m'</sup> + Σ e<sup>x-m'</sup>
                        </div>
                        <div className="softmax-steps">
                            <ol>
                                <li><strong>Initialize:</strong> running max <code>m = -∞</code>, running sum <code>d = 0</code></li>
                                <li><strong>Per tile:</strong> Update running max and rescale previous partial results</li>
                                <li><strong>Accumulate:</strong> Add tile's contribution to running output <code>O</code></li>
                                <li><strong>Key insight:</strong> When m changes, rescale everything by <code>e<sup>m_old - m_new</sup></code></li>
                            </ol>
                            <p style={{ marginTop: 'var(--space-sm)', color: 'var(--accent-secondary)', fontWeight: 'var(--fw-semibold)' }}>
                                ✅ Never needs the full N×N matrix — only one tile at a time!
                            </p>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 5. THE PARADOX — More FLOPS, less time
// ============================================================

function FlashParadox() {
    return (
        <section className="chapter-section">
            <div className="paradox-box glass-card">
                <div className="paradox-title">The Flash Attention Paradox</div>
                <div className="paradox-desc">
                    Flash Attention actually does <strong>more floating-point operations</strong> than standard
                    attention (due to the rescaling in online softmax). But it's <strong>2-4× faster</strong>
                    because it eliminates the massive HBM reads and writes of the N×N attention matrix.
                    <br /><br />
                    This is the GPU performance paradox: <strong>memory bandwidth, not compute, is the bottleneck</strong>.
                    Doing extra math in fast SRAM is far cheaper than moving data through slow HBM. This is
                    the same insight from the Prologue — LLM inference is fundamentally memory-bound.
                </div>
            </div>
        </section>
    );
}


// ============================================================
// CHAPTER 3 — Main Component
// ============================================================

export default function Chapter3({ selectedModel, selectedGPU }) {
    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    return (
        <div className="chapter chapter3 animate-in">
            <div className="chapter-header">
                <h2 className="chapter-title">Flash Attention — Making the Math Fit in Fast Memory</h2>
                <p className="chapter-hook">
                    Chapter 2 showed how to shrink the KV cache — fewer heads, paged allocation, compression.
                    But even with a smaller cache, the attention computation itself is expensive. The full
                    N×N attention matrix can be enormous, and standard implementations waste time moving it
                    between slow GPU memory and fast compute cores. Flash Attention solves this by keeping
                    everything in the GPU's tiny but lightning-fast on-chip SRAM.
                </p>
            </div>

            {/* Section 1: Memory Hierarchy */}
            <MemoryHierarchy gpu={gpu} />

            {/* Bridge */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    SRAM is 50-100× faster than HBM but 1000× smaller. The challenge is: how do we fit
                    an N×N attention computation into a few megabytes of SRAM? Answer: process it in tiles.
                </p>
            </section>

            {/* Section 2: Standard vs Flash */}
            <StandardVsFlash model={model} gpu={gpu} />

            {/* Bridge */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    But how big can each tile be? That depends on the GPU's SRAM budget and the model's head dimension.
                </p>
            </section>

            {/* Section 3: Tile Calculator */}
            <TileCalculator model={model} gpu={gpu} />

            {/* Bridge */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    Processing tiles is straightforward for the matrix multiply — but softmax normally needs
                    all values at once. How does Flash Attention handle that?
                </p>
            </section>

            {/* Section 4: Online Softmax */}
            <OnlineSoftmax />

            {/* Section 5: The Paradox */}
            <FlashParadox />

            {/* Hook to Chapter 4 */}
            <section className="chapter-section">
                <div className="chapter-next-hook glass-card">
                    <p>
                        Flash Attention optimizes how attention is <em>computed</em> — keeping data in fast SRAM.
                        But the model weights themselves are still stored in full precision, consuming
                        {' '}{formatBytes(model.params * 2)} for {model.name} in FP16.
                        Can we store the weights in even lower precision to fit larger models on smaller GPUs?
                    </p>
                    <p className="chapter-next-question">
                        → Quantization: Trading precision for memory
                    </p>
                </div>
            </section>
        </div>
    );
}

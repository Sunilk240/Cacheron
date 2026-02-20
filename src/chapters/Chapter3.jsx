import { useState } from 'react';
import { MODELS, GPUS, tileSize, formatBytes } from '../data/modelConfig';
import ExplanationPanel from '../components/ExplanationPanel';
import GoDeeper from '../components/GoDeeper';
import './Chapter3.css';

// --- Memory Hierarchy Diagram ---
function MemoryHierarchy({ gpu }) {
    const levels = [
        { label: 'Registers', size: '~KB', speed: 'Instant', color: '#4ecdc4', width: '15%' },
        { label: 'SRAM (L2 Cache)', size: `${gpu.sram_mb} MB`, speed: '~10 TB/s', color: '#7c6aff', width: '30%' },
        { label: 'HBM / DRAM', size: `${gpu.budget_mb >= 1024 ? (gpu.budget_mb / 1024).toFixed(0) + ' GB' : gpu.budget_mb + ' MB'}`, speed: `${gpu.bandwidth_gbs} GB/s`, color: '#ff6b6b', width: '60%' },
        { label: 'CPU RAM / SSD', size: '16-64 GB', speed: '~25 GB/s', color: '#6b688a', width: '100%' },
    ];

    return (
        <div className="mem-hierarchy glass-card">
            <h4 className="mem-hier-title">Memory Hierarchy — {gpu.name}</h4>
            <div className="mem-hier-pyramid">
                {levels.map((level, i) => (
                    <div key={i} className="mem-hier-level" style={{ width: level.width }}>
                        <div className="mem-hier-bar" style={{ background: level.color }}>
                            <span className="mem-hier-label">{level.label}</span>
                        </div>
                        <div className="mem-hier-stats mono">
                            <span>{level.size}</span>
                            <span>{level.speed}</span>
                        </div>
                    </div>
                ))}
            </div>
            <div className="mem-hier-legend">
                <span>← Faster, smaller</span>
                <span>Slower, larger →</span>
            </div>
        </div>
    );
}

// --- Standard vs Flash Attention Comparison ---
function AttentionComparison({ model, gpu }) {
    const [activeView, setActiveView] = useState('standard');
    const blockSize = tileSize(gpu, model, 2);
    const seqLen = 2048;

    // Standard attention IO: read Q,K,V from HBM + write S to HBM + read S from HBM + write O to HBM
    // S = N×N attention matrix
    const sMatrix = seqLen * seqLen * 2; // FP16
    const qkvSize = 3 * seqLen * model.dhead * model.Hq * 2;
    const standardIO = qkvSize + sMatrix * 2 + seqLen * model.dhead * model.Hq * 2; // rough

    // Flash Attention: read Q,K,V from HBM (in tiles) + write O to HBM. NO S matrix in HBM.
    const flashIO = qkvSize + seqLen * model.dhead * model.Hq * 2;

    return (
        <div className="attn-compare-section">
            <div className="attn-compare-toggle">
                <button
                    className={`attn-compare-btn ${activeView === 'standard' ? 'active' : ''}`}
                    onClick={() => setActiveView('standard')}
                >
                    Standard Attention
                </button>
                <button
                    className={`attn-compare-btn ${activeView === 'flash' ? 'active' : ''}`}
                    onClick={() => setActiveView('flash')}
                >
                    Flash Attention
                </button>
            </div>

            <div className="attn-compare-content glass-card">
                {activeView === 'standard' ? (
                    <div className="attn-view animate-in">
                        <h4>Standard Attention — The IO Problem</h4>
                        <div className="attn-flow">
                            <div className="attn-flow-step">
                                <div className="attn-flow-box hbm">HBM</div>
                                <div className="attn-flow-arrow">→ read Q, K, V</div>
                                <div className="attn-flow-box sram">SRAM</div>
                            </div>
                            <div className="attn-flow-step">
                                <div className="attn-flow-box sram">compute S = Q×K<sup>T</sup></div>
                            </div>
                            <div className="attn-flow-step warning">
                                <div className="attn-flow-box sram">S matrix</div>
                                <div className="attn-flow-arrow warn-arrow">→ write to HBM!</div>
                                <div className="attn-flow-box hbm">HBM</div>
                            </div>
                            <div className="attn-flow-step warning">
                                <div className="attn-flow-box hbm">HBM</div>
                                <div className="attn-flow-arrow warn-arrow">→ read back S</div>
                                <div className="attn-flow-box sram">softmax(S)</div>
                            </div>
                            <div className="attn-flow-step warning">
                                <div className="attn-flow-box sram">softmax result</div>
                                <div className="attn-flow-arrow warn-arrow">→ write to HBM!</div>
                                <div className="attn-flow-box hbm">HBM</div>
                            </div>
                            <div className="attn-flow-step">
                                <div className="attn-flow-box sram">O = P × V</div>
                                <div className="attn-flow-arrow">→ write O</div>
                                <div className="attn-flow-box hbm">HBM</div>
                            </div>
                        </div>
                        <div className="attn-io-counter">
                            <span className="io-label">Total HBM IO:</span>
                            <span className="io-value mono warn-text">{formatBytes(standardIO)}</span>
                            <span className="io-problem">S matrix ({seqLen}×{seqLen} = {formatBytes(sMatrix)}) causes 2 extra round trips</span>
                        </div>
                    </div>
                ) : (
                    <div className="attn-view animate-in">
                        <h4>Flash Attention — Tiled, No Extra IO</h4>
                        <div className="attn-flow">
                            <div className="attn-flow-step">
                                <div className="attn-flow-box hbm">HBM</div>
                                <div className="attn-flow-arrow">→ read Q tile ({blockSize}×{model.dhead})</div>
                                <div className="attn-flow-box sram">SRAM</div>
                            </div>
                            <div className="attn-flow-step">
                                <div className="attn-flow-box hbm">HBM</div>
                                <div className="attn-flow-arrow">→ read K,V tile ({blockSize}×{model.dhead})</div>
                                <div className="attn-flow-box sram">SRAM</div>
                            </div>
                            <div className="attn-flow-step success-step">
                                <div className="attn-flow-box sram wide">
                                    Compute S_tile = Q_tile × K_tile<sup>T</sup><br />
                                    <strong>softmax + multiply V — all in SRAM! ✓</strong>
                                </div>
                            </div>
                            <div className="attn-flow-step">
                                <div className="attn-flow-box sram">Accumulate O_tile</div>
                                <div className="attn-flow-arrow">→ next tile ↗</div>
                            </div>
                            <div className="attn-flow-step">
                                <div className="attn-flow-box sram">Final O</div>
                                <div className="attn-flow-arrow">→ write once</div>
                                <div className="attn-flow-box hbm">HBM</div>
                            </div>
                        </div>
                        <div className="attn-io-counter success-counter">
                            <span className="io-label">Total HBM IO:</span>
                            <span className="io-value mono success-text">{formatBytes(flashIO)}</span>
                            <span className="io-saved">S matrix NEVER leaves SRAM — saved {formatBytes(standardIO - flashIO)} of IO</span>
                        </div>
                    </div>
                )}
            </div>

            {/* Tile size info */}
            <div className="tile-info glass-card">
                <h4>Tile Size for {gpu.name}</h4>
                <div className="tile-stats">
                    <div className="tile-stat">
                        <span className="tile-stat-label">SRAM available</span>
                        <span className="tile-stat-value mono">{gpu.sram_mb} MB</span>
                    </div>
                    <div className="tile-stat">
                        <span className="tile-stat-label">Block size</span>
                        <span className="tile-stat-value mono">{blockSize} × {blockSize}</span>
                    </div>
                    <div className="tile-stat">
                        <span className="tile-stat-label">Tiles needed (N={seqLen})</span>
                        <span className="tile-stat-value mono">{Math.ceil(seqLen / blockSize)}²</span>
                    </div>
                    <div className="tile-stat">
                        <span className="tile-stat-label">d_head</span>
                        <span className="tile-stat-value mono">{model.dhead}</span>
                    </div>
                </div>
            </div>
        </div>
    );
}


// --- Main Chapter 3 ---
export default function Chapter3({ selectedModel, selectedGPU }) {
    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    return (
        <div className="chapter chapter3 animate-in">
            <div className="chapter-header">
                <h2 className="chapter-title">How do we make memory reads faster?</h2>
                <p className="chapter-hook">
                    Even after shrinking the KV cache, standard attention writes intermediate results to slow HBM memory.
                    Flash Attention rearranges the computation so everything stays in fast SRAM.
                </p>
            </div>

            {/* Section 1: The Memory Hierarchy */}
            <section className="chapter-section">
                <ExplanationPanel title="Not all memory is created equal" variant="what">
                    <p>
                        GPUs have a <strong>memory hierarchy</strong>. At the top: tiny, blazing-fast SRAM (on-chip cache).
                        At the bottom: large, slow HBM (High Bandwidth Memory). The speed difference is massive —
                        SRAM is roughly <strong>10–100× faster</strong> than HBM.
                    </p>
                    <p>
                        Standard attention computes the full {model.Hq}-head attention matrix and writes it to HBM
                        before computing softmax. For a 2048-token sequence, that's a {2048}×{2048} matrix per head —
                        written to slow memory and read back. This IO overhead dominates the computation.
                    </p>
                </ExplanationPanel>
                <MemoryHierarchy gpu={gpu} />
            </section>

            {/* Section 2: Standard vs Flash */}
            <section className="chapter-section">
                <ExplanationPanel title="Flash Attention: Keep everything in SRAM" variant="what">
                    <p>
                        <strong>Flash Attention</strong> (Tri Dao, 2022) restructures the attention computation to work
                        in <em>tiles</em> (blocks of rows). Instead of computing the full N×N attention matrix, it
                        processes small tiles that fit entirely in SRAM:
                    </p>
                    <p>
                        1. Load a block of Q, K, V into SRAM<br />
                        2. Compute the attention scores <em>and</em> multiply by V — all in SRAM<br />
                        3. Accumulate the output, move to the next tile<br />
                        4. The attention matrix S <strong>never touches HBM</strong>
                    </p>
                </ExplanationPanel>
                <AttentionComparison model={model} gpu={gpu} />
            </section>

            {/* Go Deeper: Online Softmax */}
            <GoDeeper title="Go Deeper — Online Softmax: The Mathematical Trick">
                <ExplanationPanel title="Why tiling attention is harder than it looks" variant="math">
                    <p>
                        The reason you can't naively tile softmax: softmax needs the <strong>maximum value</strong> and
                        the <strong>sum of exponentials</strong> across the entire row. If you only see part of the row,
                        you don't know these values.
                    </p>
                    <p>
                        <strong>Online softmax</strong> solves this by maintaining a running maximum and a running sum
                        as you process tiles left to right:
                    </p>
                    <p>
                        <code>m_new = max(m_old, max(S_tile))</code><br />
                        <code>l_new = l_old × exp(m_old − m_new) + sum(exp(S_tile − m_new))</code><br />
                        <code>O_new = O_old × (l_old / l_new) × exp(m_old − m_new) + P_tile × V_tile / l_new</code>
                    </p>
                    <p>
                        The correction factor <code>exp(m_old − m_new)</code> adjusts all previous results when a new
                        maximum is discovered. Once you've seen all tiles, the output is <strong>numerically identical</strong> to
                        standard softmax — just computed without ever writing the full S matrix.
                    </p>
                </ExplanationPanel>
            </GoDeeper>

            {/* Section 3: The Paradox */}
            <section className="chapter-section">
                <ExplanationPanel title="More FLOPs, less time" variant="why">
                    <p>
                        Flash Attention actually does <strong>more arithmetic</strong> than standard attention — the
                        online softmax rescaling adds extra multiplications. But it does <strong>far fewer memory operations</strong>.
                    </p>
                    <p>
                        Since GPU compute is 10–100× faster than memory bandwidth, trading extra FLOPs for fewer HBM reads
                        is a huge win. Flash Attention achieves <strong>2–4× wall-clock speedup</strong> and uses
                        <strong>5–20× less memory</strong> (no N×N matrix stored).
                    </p>
                    <p>
                        This is the same insight as the Prologue: <strong>inference is memory-bound</strong>.
                        Any optimization that reduces memory traffic — even at the cost of more arithmetic — wins.
                    </p>
                </ExplanationPanel>
            </section>

            {/* Hook to Chapter 4 */}
            <section className="chapter-section">
                <div className="chapter-next-hook glass-card">
                    <p>
                        Flash Attention optimized <em>how</em> we read attention data. But the largest memory consumer
                        hasn't been touched: <strong>the model weights themselves</strong>.
                        {' '}{model.name} at FP16 is {formatBytes(model.params * 2)} — and Llama-2-7B is 14 GB.
                    </p>
                    <p>
                        What if we could represent those weights with fewer bits?
                    </p>
                    <p className="chapter-next-question">
                        → How do we shrink the model itself?
                    </p>
                </div>
            </section>
        </div>
    );
}

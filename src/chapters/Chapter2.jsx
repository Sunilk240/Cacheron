import { useState } from 'react';
import { MODELS, GPUS, kvCacheSize, kvPerToken, formatBytes } from '../data/modelConfig';
import ExplanationPanel from '../components/ExplanationPanel';
import MemoryBar from '../components/MemoryBar';
import GoDeeper from '../components/GoDeeper';
import './Chapter2.css';

// --- MHA → GQA → MQA Toggle ---
function AttentionModeToggle({ model }) {
    const [mode, setMode] = useState('current'); // 'mha', 'current', 'mqa'

    const modes = {
        mha: { label: 'MHA', kvHeads: model.Hq, desc: 'Multi-Head Attention — every Q head has its own KV head', ratio: '1:1' },
        current: { label: model.attnType, kvHeads: model.Hkv, desc: `Current configuration — ${model.Hq} Q heads share ${model.Hkv} KV heads`, ratio: `${model.Hq / model.Hkv}:1` },
        mqa: { label: 'MQA', kvHeads: 1, desc: 'Multi-Query Attention — all Q heads share a single KV head', ratio: `${model.Hq}:1` },
    };

    const currentMode = modes[mode];
    const mhaKV = kvPerToken(model, model.Hq, 2);
    const currentKV = kvPerToken(model, currentMode.kvHeads, 2);
    const savingsVsMHA = ((1 - currentKV / mhaKV) * 100).toFixed(0);

    const qHeads = model.Hq;
    const kvHeads = currentMode.kvHeads;
    const groupSize = Math.ceil(qHeads / kvHeads);

    return (
        <div className="attn-mode-section">
            <div className="attn-mode-toggle">
                {Object.entries(modes).map(([key, m]) => (
                    <button
                        key={key}
                        className={`attn-mode-btn ${mode === key ? 'active' : ''}`}
                        onClick={() => setMode(key)}
                    >
                        {m.label}
                    </button>
                ))}
            </div>

            <div className="attn-mode-content glass-card">
                <div className="attn-mode-visual">
                    {/* Q heads row */}
                    <div className="attn-heads-section">
                        <span className="attn-heads-label mono">Q heads ({qHeads})</span>
                        <div className="attn-heads-row">
                            {Array.from({ length: Math.min(qHeads, 32) }, (_, i) => {
                                const groupIdx = Math.floor(i / groupSize);
                                const hue = (groupIdx * 360 / kvHeads) % 360;
                                return (
                                    <div
                                        key={i}
                                        className="attn-head q-head-viz"
                                        style={{ borderColor: `hsl(${hue}, 60%, 55%)`, backgroundColor: `hsla(${hue}, 60%, 55%, 0.1)` }}
                                        title={`Q head ${i} → KV head ${groupIdx}`}
                                    />
                                );
                            })}
                            {qHeads > 32 && <span className="attn-heads-more">+{qHeads - 32}</span>}
                        </div>
                    </div>

                    {/* Arrows */}
                    <div className="attn-group-arrows">
                        {Array.from({ length: Math.min(kvHeads, 16) }, (_, i) => {
                            const hue = (i * 360 / kvHeads) % 360;
                            return (
                                <div key={i} className="attn-arrow-group" style={{ color: `hsl(${hue}, 60%, 55%)` }}>
                                    {'↓'.repeat(Math.min(groupSize, 4))}
                                </div>
                            );
                        })}
                    </div>

                    {/* KV heads row */}
                    <div className="attn-heads-section">
                        <span className="attn-heads-label mono">KV heads ({kvHeads})</span>
                        <div className="attn-heads-row">
                            {Array.from({ length: Math.min(kvHeads, 32) }, (_, i) => {
                                const hue = (i * 360 / kvHeads) % 360;
                                return (
                                    <div
                                        key={i}
                                        className="attn-head kv-head-viz"
                                        style={{ borderColor: `hsl(${hue}, 60%, 55%)`, backgroundColor: `hsla(${hue}, 60%, 55%, 0.15)` }}
                                        title={`KV head ${i}`}
                                    />
                                );
                            })}
                        </div>
                    </div>
                </div>

                <div className="attn-mode-info">
                    <h4>{currentMode.label} — {currentMode.ratio} sharing</h4>
                    <p>{currentMode.desc}</p>
                    <div className="attn-mode-stats">
                        <div className="attn-stat">
                            <span className="attn-stat-label">KV per token</span>
                            <span className="attn-stat-value mono">{formatBytes(currentKV)}</span>
                        </div>
                        <div className="attn-stat">
                            <span className="attn-stat-label">vs MHA savings</span>
                            <span className={`attn-stat-value mono ${Number(savingsVsMHA) > 0 ? 'savings' : ''}`}>
                                {savingsVsMHA}%
                            </span>
                        </div>
                        <div className="attn-stat">
                            <span className="attn-stat-label">KV @2048 tokens</span>
                            <span className="attn-stat-value mono">
                                {formatBytes(kvCacheSize({ ...model, Hkv: kvHeads }, 2048, 1, 2))}
                            </span>
                        </div>
                    </div>
                </div>
            </div>

            {/* Side-by-side memory bars */}
            <div className="attn-compare-bars glass-card">
                <h4>Memory Comparison at 2048 tokens</h4>
                <MemoryBar
                    value={kvCacheSize({ ...model, Hkv: model.Hq }, 2048, 1, 2)}
                    max={kvCacheSize({ ...model, Hkv: model.Hq }, 2048, 1, 2)}
                    label="MHA (1:1)"
                    color="warm"
                />
                <MemoryBar
                    value={kvCacheSize(model, 2048, 1, 2)}
                    max={kvCacheSize({ ...model, Hkv: model.Hq }, 2048, 1, 2)}
                    label={`${model.attnType} (current)`}
                    color="accent"
                />
                <MemoryBar
                    value={kvCacheSize({ ...model, Hkv: 1 }, 2048, 1, 2)}
                    max={kvCacheSize({ ...model, Hkv: model.Hq }, 2048, 1, 2)}
                    label="MQA (N:1)"
                    color="success"
                />
            </div>
        </div>
    );
}

// --- PagedAttention Visualization ---
function PagedAttentionViz() {
    const [showPaged, setShowPaged] = useState(false);

    return (
        <div className="paged-section">
            <div className="paged-toggle">
                <button
                    className={`paged-toggle-btn ${!showPaged ? 'active' : ''}`}
                    onClick={() => setShowPaged(false)}
                >
                    Contiguous Allocation
                </button>
                <button
                    className={`paged-toggle-btn ${showPaged ? 'active' : ''}`}
                    onClick={() => setShowPaged(true)}
                >
                    PagedAttention
                </button>
            </div>

            <div className="paged-content glass-card">
                {!showPaged ? (
                    <div className="paged-viz animate-in">
                        <h4>Contiguous Allocation — The Problem</h4>
                        <div className="mem-blocks-container">
                            <div className="mem-block-row">
                                <span className="mem-block-label">Seq 1 (len=5)</span>
                                <div className="mem-blocks">
                                    {[1, 2, 3, 4, 5].map(i => <div key={i} className="mem-block used seq1">T{i}</div>)}
                                    {[1, 2, 3].map(i => <div key={`w${i}`} className="mem-block wasted">∅</div>)}
                                </div>
                                <span className="mem-waste-label">37% wasted</span>
                            </div>
                            <div className="mem-block-row">
                                <span className="mem-block-label">Seq 2 (len=3)</span>
                                <div className="mem-blocks">
                                    {[1, 2, 3].map(i => <div key={i} className="mem-block used seq2">T{i}</div>)}
                                    {[1, 2, 3, 4, 5].map(i => <div key={`w${i}`} className="mem-block wasted">∅</div>)}
                                </div>
                                <span className="mem-waste-label">63% wasted</span>
                            </div>
                            <div className="mem-block-row">
                                <span className="mem-block-label">Seq 3 (len=7)</span>
                                <div className="mem-blocks">
                                    {[1, 2, 3, 4, 5, 6, 7].map(i => <div key={i} className="mem-block used seq3">T{i}</div>)}
                                    {[1].map(i => <div key={`w${i}`} className="mem-block wasted">∅</div>)}
                                </div>
                                <span className="mem-waste-label">12% wasted</span>
                            </div>
                        </div>
                        <div className="paged-callout warn">
                            Pre-allocated blocks must accommodate the <strong>maximum</strong> sequence length.
                            On average, <strong>60–80% of allocated memory is wasted</strong> on padding tokens that will never be used.
                        </div>
                    </div>
                ) : (
                    <div className="paged-viz animate-in">
                        <h4>PagedAttention — Virtual Memory for KV Cache</h4>
                        <div className="paged-grid">
                            <div className="paged-physical">
                                <span className="paged-col-label">Physical Memory (Block Pool)</span>
                                <div className="mem-blocks paged-blocks">
                                    <div className="mem-block used seq1">S1-B0</div>
                                    <div className="mem-block used seq2">S2-B0</div>
                                    <div className="mem-block used seq3">S3-B0</div>
                                    <div className="mem-block used seq1">S1-B1</div>
                                    <div className="mem-block used seq3">S3-B1</div>
                                    <div className="mem-block free">Free</div>
                                    <div className="mem-block free">Free</div>
                                    <div className="mem-block free">Free</div>
                                </div>
                            </div>
                            <div className="paged-table">
                                <span className="paged-col-label">Block Table (Mapping)</span>
                                <table className="block-table">
                                    <thead>
                                        <tr><th>Seq</th><th>Block 0</th><th>Block 1</th></tr>
                                    </thead>
                                    <tbody>
                                        <tr className="seq1-row"><td>Seq 1</td><td>→ Phys 0</td><td>→ Phys 3</td></tr>
                                        <tr className="seq2-row"><td>Seq 2</td><td>→ Phys 1</td><td>—</td></tr>
                                        <tr className="seq3-row"><td>Seq 3</td><td>→ Phys 2</td><td>→ Phys 4</td></tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        <div className="paged-callout success">
                            Blocks are allocated <strong>on demand</strong> as sequences grow. No pre-allocation waste.
                            Internal fragmentation is limited to the last block per sequence (≤ block_size tokens).
                            <strong> Near-zero waste.</strong>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

// --- GQA Comparison across all 5 models ---
function GQAComparisonTable({ selectedModel, gpu }) {
    const budgetBytes = gpu.budget_mb * 1024 * 1024;

    return (
        <div className="gqa-compare glass-card">
            <h4>GQA Ratio Comparison — All Models</h4>
            <div className="gqa-compare-wrapper">
                <table className="model-compare-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>H<sub>q</sub></th>
                            <th>H<sub>kv</sub></th>
                            <th>Ratio</th>
                            <th>KV per token</th>
                            <th>KV @2048</th>
                            <th>vs MHA savings</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Object.entries(MODELS).map(([key, m]) => {
                            const kvPT = kvPerToken(m, null, 2);
                            const mhaKvPT = kvPerToken(m, m.Hq, 2);
                            const kv2048 = kvCacheSize(m, 2048, 1, 2);
                            const savings = ((1 - kvPT / mhaKvPT) * 100).toFixed(0);
                            const isSelected = key === selectedModel;
                            return (
                                <tr key={key} className={isSelected ? 'selected-row' : ''}>
                                    <td className="model-name-cell">{m.name}</td>
                                    <td className="mono">{m.Hq}</td>
                                    <td className="mono">{m.Hkv}</td>
                                    <td className="mono">{m.isMHA ? '1:1' : `${m.Hq / m.Hkv}:1`}</td>
                                    <td className="mono">{formatBytes(kvPT)}</td>
                                    <td className="mono">{formatBytes(kv2048)}</td>
                                    <td className={`mono ${Number(savings) > 0 ? 'savings' : 'no-savings'}`}>
                                        {savings}%
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}


// --- Main Chapter 2 ---
export default function Chapter2({ selectedModel, selectedGPU }) {
    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    return (
        <div className="chapter chapter2 animate-in">
            <div className="chapter-header">
                <h2 className="chapter-title">How do we stop memory from exploding?</h2>
                <p className="chapter-hook">
                    The KV cache grows with every token. We need to shrink what's stored, eliminate wasted space,
                    and decide what to keep versus what to discard.
                </p>
            </div>

            {/* Section 1: Q doesn't need caching */}
            <section className="chapter-section">
                <ExplanationPanel title="Why only K and V are cached (not Q)" variant="why">
                    <p>
                        During autoregressive decode, the model generates one new token at a time. The new token's
                        <strong> Query</strong> needs to attend to <strong>all previous tokens' Keys and Values</strong>.
                    </p>
                    <p>
                        But the Query is only needed for the <em>current</em> step. Once attention is computed, the
                        Query is discarded — it's never used again. Past tokens don't need the new token's Query either,
                        because attention is <strong>causal</strong>: token 50 can only attend to tokens 1–49, never to token 51.
                    </p>
                    <p>
                        So we cache K and V (needed by all future tokens), but not Q (needed only by the current token).
                    </p>
                </ExplanationPanel>
            </section>

            {/* Section 2: MHA → GQA → MQA */}
            <section className="chapter-section">
                <ExplanationPanel title="Shrinking the KV Cache: MHA → GQA → MQA" variant="what">
                    <p>
                        The most direct way to reduce KV cache size: <strong>use fewer KV heads</strong>.
                        In standard Multi-Head Attention (MHA), every Query head has its own Key and Value head.
                        But experiments show that multiple Query heads can <em>share</em> a single KV head without significant quality loss.
                    </p>
                    <p>
                        <strong>GQA (Grouped-Query Attention)</strong>: groups of Query heads share a single KV head.
                        <strong> MQA (Multi-Query Attention)</strong>: all Query heads share just one KV head.
                        The tradeoff is quality vs. memory — and GQA hits a sweet spot.
                    </p>
                </ExplanationPanel>
                <AttentionModeToggle model={model} />
            </section>

            {/* Section 3: GQA across models */}
            <section className="chapter-section">
                <ExplanationPanel title="How real models use GQA" variant="what">
                    <p>
                        Notice how Llama-2-7B uses <strong>Pure MHA</strong> (32 Q heads, 32 KV heads) — it was released
                        before GQA became standard practice. Compare it to Llama-3.2-3B which uses <strong>GQA 3:1</strong> (24 Q heads, 8 KV heads),
                        saving 67% of KV cache memory with negligible quality loss.
                    </p>
                </ExplanationPanel>
                <GQAComparisonTable selectedModel={selectedModel} gpu={gpu} />
            </section>

            {/* Section 4: PagedAttention */}
            <section className="chapter-section">
                <ExplanationPanel title="PagedAttention: Virtual Memory for KV Cache" variant="what">
                    <p>
                        Even after reducing the number of KV heads, there's another source of waste: <strong>pre-allocation</strong>.
                        When serving multiple sequences in a batch, KV cache memory is typically pre-allocated for the
                        maximum possible sequence length. But most sequences don't reach the maximum, so 60–80% of
                        allocated memory sits unused.
                    </p>
                    <p>
                        <strong>PagedAttention</strong> (from vLLM) borrows a concept from operating systems: virtual memory paging.
                        Instead of allocating one contiguous block per sequence, it divides KV cache into fixed-size <em>blocks</em> (pages)
                        and allocates them on demand through a <em>block table</em>.
                    </p>
                </ExplanationPanel>
                <PagedAttentionViz />
            </section>

            {/* Go Deeper: advanced KV optimization */}
            <GoDeeper title="Go Deeper — Emerging KV Cache Optimizations">
                <ExplanationPanel title="Beyond GQA and PagedAttention" variant="math">
                    <p><strong>KV Cache Quantization:</strong> Store the K and V vectors at reduced precision (INT8 or INT4)
                        instead of FP16. Since KV values are less sensitive to quantization error than model weights,
                        you can compress the cache 2–4× with minimal quality impact.</p>
                    <p><strong>Token Eviction:</strong> Not all tokens are equally important for future attention.
                        Eviction policies (e.g., StreamingLLM's attention sink + sliding window) keep only the most recent
                        tokens plus a few "anchor" tokens, enabling infinite-length generation with bounded memory.</p>
                    <p><strong>KV Cache Offloading:</strong> Spill inactive KV cache blocks from GPU memory to CPU RAM or SSD,
                        loading them back when needed. Trades latency for capacity — useful for very long contexts.</p>
                </ExplanationPanel>
            </GoDeeper>

            {/* Hook to Chapter 3 */}
            <section className="chapter-section">
                <div className="chapter-next-hook glass-card">
                    <p>
                        We've reduced <em>what</em> we store by using fewer KV heads (GQA) and smarter allocation (PagedAttention).
                        But even with a smaller cache, reading attention data from memory is still slow —
                        because the standard attention algorithm writes intermediate results to slow HBM memory unnecessarily.
                    </p>
                    <p className="chapter-next-question">
                        → How do we make the memory reads faster?
                    </p>
                </div>
            </section>
        </div>
    );
}

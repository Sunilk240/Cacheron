import { useState, useMemo } from 'react';
import { MODELS, GPUS, kvCacheSize, kvPerToken, formatBytes } from '../data/modelConfig';
import './Chapter2.css';

// ============================================================
// 1. MHA → GQA → MQA ANIMATED TOGGLE
//    Show head grid morphing as you switch modes, live memory comparison
// ============================================================

function AttentionModeToggle({ model, gpu }) {
    const [mode, setMode] = useState('actual'); // 'mha', 'gqa', 'mqa', 'actual'

    const modes = {
        mha: { label: 'MHA', kvHeads: model.Hq, desc: 'Multi-Head Attention: every Q head gets its own dedicated K,V head. Maximum expressiveness, but the largest KV cache.' },
        actual: { label: model.attnType.split(' ')[0], kvHeads: model.Hkv, desc: `${model.name}'s actual configuration: ${model.Hq} Q heads share ${model.Hkv} KV heads. This is what the model actually uses.` },
        mqa: { label: 'MQA', kvHeads: 1, desc: 'Multi-Query Attention: all Q heads share a single K,V pair. Smallest cache, but can reduce model quality.' },
    };

    // Add GQA only if actual isn't MHA or MQA
    if (model.Hkv !== model.Hq && model.Hkv !== 1) {
        // actual IS the GQA mode
    }

    const current = modes[mode] || modes.actual;
    const kvH = current.kvHeads;
    const qPerKV = Math.ceil(model.Hq / kvH);

    // Cache sizes at 4096 tokens
    const tokens = 4096;
    const mhaKV = kvPerToken(model, model.Hq) * tokens;
    const currentKV = kvPerToken(model, kvH) * tokens;
    const savings = mhaKV > 0 ? ((1 - currentKV / mhaKV) * 100).toFixed(0) : 0;

    return (
        <section className="chapter-section">
            <h3 className="section-title">Reducing KV Heads — MHA vs GQA vs MQA</h3>
            <p className="section-desc">
                The biggest lever for shrinking the KV cache is reducing the number of KV heads.
                In the original Transformer, every query head had its own key and value head — that's
                <strong> Multi-Head Attention (MHA)</strong>. But do we really need that many KV heads?
                Research showed that sharing KV heads across groups of query heads barely hurts quality
                while dramatically cutting memory. That's <strong>Grouped-Query Attention (GQA)</strong>.
                Take it to the extreme and share one KV pair across all heads — that's <strong>MQA</strong>.
            </p>
            <p className="section-desc">
                Toggle between modes to see how the head layout and cache size change:
            </p>

            <div className="attn-mode-section glass-card">
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

                <div className="attn-mode-info">
                    <div className="attn-mode-name">{current.label} — {model.Hq} Q : {kvH} KV</div>
                    <div className="attn-mode-explanation">{current.desc}</div>
                </div>

                {/* Head grid */}
                <div className="attn-mode-heads">
                    {Array.from({ length: kvH }, (_, g) => (
                        <div key={g} className="mode-head-group">
                            <div className="q-row">
                                {Array.from({ length: qPerKV }, (_, q) => (
                                    <div key={q} className="mode-head-dot q-h" title={`Q${g * qPerKV + q + 1}`}>Q</div>
                                ))}
                            </div>
                            <div className="mode-head-dot kv-h" title={`KV${g + 1}`}>K,V</div>
                            <span className="mode-head-label">g{g + 1}</span>
                        </div>
                    ))}
                </div>

                {/* Stats */}
                <div className="attn-mode-stats">
                    <div className="mode-stat">
                        <div className="mode-stat-value">{kvH}</div>
                        <div className="mode-stat-label">KV Heads</div>
                    </div>
                    <div className="mode-stat">
                        <div className="mode-stat-value">{qPerKV}:1</div>
                        <div className="mode-stat-label">Q per KV</div>
                    </div>
                    <div className="mode-stat">
                        <div className="mode-stat-value">{formatBytes(currentKV)}</div>
                        <div className="mode-stat-label">KV @ {tokens.toLocaleString()} tokens</div>
                    </div>
                    <div className="mode-stat">
                        <div className="mode-stat-value savings">{savings}%</div>
                        <div className="mode-stat-label">Savings vs MHA</div>
                    </div>
                </div>

                {/* Memory comparison bars */}
                <div className="mode-memory-bars">
                    <div className="mode-mem-row">
                        <span className="mode-mem-label mha-ref">MHA</span>
                        <div className="mode-mem-track">
                            <div className="mode-mem-fill mha-fill" style={{ width: '100%' }}>
                                <span className="mode-mem-fill-text">{formatBytes(mhaKV)}</span>
                            </div>
                        </div>
                    </div>
                    <div className="mode-mem-row">
                        <span className="mode-mem-label active-mode">{current.label}</span>
                        <div className="mode-mem-track">
                            <div className="mode-mem-fill current-fill" style={{ width: `${(currentKV / mhaKV) * 100}%` }}>
                                <span className="mode-mem-fill-text">{formatBytes(currentKV)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 2. PagedAttention VISUALIZATION
//    Before (contiguous, wasteful) vs After (paged, efficient)
// ============================================================

function PagedAttentionViz() {
    const [showPaged, setShowPaged] = useState(false);

    // Simulate 3 requests with different lengths
    const requests = [
        { id: 'A', tokens: 5, maxAlloc: 8, color: 'used-a' },
        { id: 'B', tokens: 3, maxAlloc: 8, color: 'used-b' },
        { id: 'C', tokens: 6, maxAlloc: 8, color: 'used-c' },
    ];

    const totalSlots = 24;

    // Contiguous: each request reserves maxAlloc slots
    const contiguousBlocks = [];
    requests.forEach(req => {
        for (let i = 0; i < req.maxAlloc; i++) {
            contiguousBlocks.push({
                type: i < req.tokens ? req.color : 'wasted',
                label: i < req.tokens ? req.id : '×',
            });
        }
    });

    // Paged: only allocate what's used, non-contiguous is fine
    const pagedBlocks = [];
    const pagedOrder = [];
    requests.forEach(req => {
        for (let i = 0; i < req.tokens; i++) {
            pagedOrder.push({ type: req.color, label: req.id });
        }
    });
    for (let i = 0; i < totalSlots; i++) {
        if (i < pagedOrder.length) {
            pagedBlocks.push(pagedOrder[i]);
        } else {
            pagedBlocks.push({ type: 'free', label: '' });
        }
    }

    const usedPct = ((pagedOrder.length / totalSlots) * 100).toFixed(0);
    const wastedPct = (((totalSlots - requests.reduce((s, r) => s + r.tokens, 0)) / totalSlots) * 100).toFixed(0);
    const contiguousWaste = requests.reduce((s, r) => s + (r.maxAlloc - r.tokens), 0);

    return (
        <section className="chapter-section">
            <h3 className="section-title">PagedAttention — No More Wasted Memory</h3>
            <p className="section-desc">
                Even with fewer KV heads, there's another source of waste. Traditional KV caches allocate
                a <strong>contiguous block</strong> for each request, sized for the maximum possible sequence length.
                If a request only uses 5 out of 8 slots, the remaining 3 slots are wasted — they can't be
                used by other requests. Think of it like reserving entire rows in a theater for groups of different
                sizes.
            </p>
            <p className="section-desc">
                <strong>PagedAttention</strong> (from the vLLM project) borrows the idea of <em>virtual memory
                    paging</em> from operating systems: split the KV cache into fixed-size blocks and allocate them
                on demand, non-contiguously. A block table tracks where each request's blocks live. No wasted slots.
            </p>

            <div className="paged-section glass-card">
                <div style={{ display: 'flex', gap: 'var(--space-sm)', justifyContent: 'center', marginBottom: 'var(--space-lg)' }}>
                    <button className={`attn-mode-btn ${!showPaged ? 'active' : ''}`} onClick={() => setShowPaged(false)}>
                        Before (Contiguous)
                    </button>
                    <button className={`attn-mode-btn ${showPaged ? 'active' : ''}`} onClick={() => setShowPaged(true)}>
                        After (PagedAttention)
                    </button>
                </div>

                <div className="paged-comparison">
                    {/* Before */}
                    <div className={`paged-side ${!showPaged ? 'highlight' : ''}`} style={{ borderColor: !showPaged ? 'var(--accent-primary)' : 'var(--border-subtle)' }}>
                        <div className="paged-side-title">Contiguous Allocation</div>
                        <div className="paged-side-desc">
                            Each request reserves {requests[0].maxAlloc} slots (max sequence length), even if it uses fewer.
                            Slots marked <strong style={{ color: 'var(--accent-warm)' }}>×</strong> are wasted.
                        </div>
                        <div className="paged-memory-blocks">
                            {contiguousBlocks.map((block, i) => (
                                <div key={i} className={`paged-block ${block.type}`}>{block.label}</div>
                            ))}
                        </div>
                        <div className="paged-efficiency poor">
                            ⚠️ {contiguousWaste} of {totalSlots} slots wasted ({((contiguousWaste / totalSlots) * 100).toFixed(0)}% waste)
                        </div>
                    </div>

                    {/* After */}
                    <div className={`paged-side ${showPaged ? 'highlight' : ''}`} style={{ borderColor: showPaged ? 'var(--accent-secondary)' : 'var(--border-subtle)' }}>
                        <div className="paged-side-title">PagedAttention</div>
                        <div className="paged-side-desc">
                            Blocks allocated on-demand. No pre-reservation. Free blocks available for new requests.
                        </div>
                        <div className="paged-memory-blocks">
                            {pagedBlocks.map((block, i) => (
                                <div key={i} className={`paged-block ${block.type}`}>{block.label}</div>
                            ))}
                        </div>
                        <div className="paged-efficiency good">
                            ✅ {pagedOrder.length} of {totalSlots} slots used — {totalSlots - pagedOrder.length} free for new requests (0% waste)
                        </div>
                    </div>
                </div>

                {/* Block table concept */}
                <div style={{ marginTop: 'var(--space-lg)' }}>
                    <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginBottom: 'var(--space-sm)' }}>
                        <strong>Block Table:</strong> Maps each request's logical token positions to physical memory blocks.
                    </div>
                    <table className="block-table">
                        <thead>
                            <tr>
                                <th>Request</th>
                                <th>Tokens Used</th>
                                <th>Blocks Allocated</th>
                                <th>Memory</th>
                            </tr>
                        </thead>
                        <tbody>
                            {requests.map(req => (
                                <tr key={req.id}>
                                    <td>
                                        <span className={`paged-block ${req.color}`} style={{ display: 'inline-flex', width: '22px', height: '18px' }}>{req.id}</span>
                                        {' Request {req.id}'}
                                    </td>
                                    <td>{req.tokens}</td>
                                    <td>{req.tokens} blocks (non-contiguous)</td>
                                    <td>{req.tokens} × block_size bytes</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 3. KV OPTIMIZATION TECHNIQUES
//    Cards for KV Quantization, Token Eviction, Cache Offloading
// ============================================================

function KVOptimizations() {
    return (
        <section className="chapter-section">
            <h3 className="section-title">Other Ways to Shrink the KV Cache</h3>
            <p className="section-desc">
                Beyond reducing KV heads and paging, there are three more techniques used in practice.
                These can be combined with GQA and PagedAttention for even greater memory savings.
            </p>

            <div className="kv-opt-grid">
                {/* KV Quantization */}
                <div className="kv-opt-card glass-card">
                    <div className="kv-opt-icon">🔢</div>
                    <div className="kv-opt-name">KV Cache Quantization</div>
                    <div className="kv-opt-desc">
                        Store K and V vectors in lower precision (INT8 or even INT4 instead of FP16).
                        Halves or quarters the KV cache size with minimal quality loss.
                    </div>
                    <div className="kv-opt-detail">
                        <span className="kv-opt-badge quant">INT8</span>
                        <span className="kv-opt-badge quant">INT4</span>
                        <br />
                        <strong>How it works:</strong> After computing K and V in FP16, quantize to INT8
                        before storing in cache. Dequantize when reading for attention computation.
                        <br />
                        <strong>Savings:</strong> INT8 → 2× smaller cache. INT4 → 4× smaller cache.
                        <br />
                        <strong>Trade-off:</strong> Slight accuracy loss from quantization noise, especially at INT4.
                    </div>
                </div>

                {/* Token Eviction */}
                <div className="kv-opt-card glass-card">
                    <div className="kv-opt-icon">🗑️</div>
                    <div className="kv-opt-name">Token Eviction</div>
                    <div className="kv-opt-desc">
                        Not all tokens are equally important. Evict KV entries for tokens the model
                        rarely attends to, keeping only the most relevant ones.
                    </div>
                    <div className="kv-opt-detail">
                        <span className="kv-opt-badge evict">H2O</span>
                        <span className="kv-opt-badge evict">Scissors</span>
                        <br />
                        <strong>How it works:</strong> Track cumulative attention scores. Tokens that consistently
                        receive low attention across heads and layers are candidates for eviction.
                        <br />
                        <strong>Savings:</strong> Keep only the top-k most-attended tokens — 50-80% cache reduction.
                        <br />
                        <strong>Trade-off:</strong> Risk of evicting a token that becomes important later (e.g. a fact
                        mentioned early in a long conversation).
                    </div>
                </div>

                {/* KV Offloading */}
                <div className="kv-opt-card glass-card">
                    <div className="kv-opt-icon">💾</div>
                    <div className="kv-opt-name">KV Offloading</div>
                    <div className="kv-opt-desc">
                        Move older KV cache entries from GPU memory to CPU RAM or even SSD. Bring them
                        back when needed for attention computation.
                    </div>
                    <div className="kv-opt-detail">
                        <span className="kv-opt-badge offload">GPU → CPU</span>
                        <span className="kv-opt-badge offload">CPU → SSD</span>
                        <br />
                        <strong>How it works:</strong> A sliding window of recent tokens stays in fast GPU memory.
                        Older entries are offloaded to CPU RAM and prefetched when attention needs them.
                        <br />
                        <strong>Savings:</strong> Only recent tokens consume GPU memory — enables very long contexts.
                        <br />
                        <strong>Trade-off:</strong> Latency increases when accessing offloaded entries. PCIe
                        bandwidth becomes the bottleneck.
                    </div>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 4. GQA COMPARISON TABLE across 5 models
// ============================================================

function GQAComparisonTable({ selectedModel }) {
    const modelKeys = Object.keys(MODELS);

    return (
        <section className="chapter-section">
            <h3 className="section-title">GQA Across All Models</h3>
            <p className="section-desc">
                Here's how the attention head configuration differs across all 5 models, and the
                resulting KV cache savings compared to full MHA.
            </p>

            <div className="glass-card" style={{ overflowX: 'auto' }}>
                <table className="model-compare-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Q Heads</th>
                            <th>KV Heads</th>
                            <th>Ratio</th>
                            <th>Type</th>
                            <th>KV @ 4K (Actual)</th>
                            <th>KV @ 4K (If MHA)</th>
                            <th>Savings</th>
                        </tr>
                    </thead>
                    <tbody>
                        {modelKeys.map(key => {
                            const m = MODELS[key];
                            const actualKV = kvPerToken(m) * 4096;
                            const mhaKV = kvPerToken(m, m.Hq) * 4096;
                            const saved = mhaKV > 0 ? ((1 - actualKV / mhaKV) * 100).toFixed(0) : 0;
                            const type = m.Hq === m.Hkv ? 'MHA' : m.Hkv === 1 ? 'MQA' : 'GQA';
                            const ratio = `${Math.ceil(m.Hq / m.Hkv)}:1`;
                            return (
                                <tr key={key} className={key === selectedModel ? 'selected-row' : ''}>
                                    <td style={{ fontFamily: 'var(--font-sans)', fontWeight: 'var(--fw-semibold)' }}>{m.name}</td>
                                    <td>{m.Hq}</td>
                                    <td>{m.Hkv}</td>
                                    <td>{ratio}</td>
                                    <td>
                                        <span className={type === 'MHA' ? 'mha-badge' : 'gqa-badge'}>{type}</span>
                                    </td>
                                    <td>{formatBytes(actualKV)}</td>
                                    <td style={{ color: 'var(--text-muted)' }}>{formatBytes(mhaKV)}</td>
                                    <td style={{ color: Number(saved) > 0 ? 'var(--accent-secondary)' : 'var(--text-muted)', fontWeight: 'var(--fw-bold)' }}>
                                        {saved}%
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </section>
    );
}


// ============================================================
// CHAPTER 2 — Main Component
// ============================================================

export default function Chapter2({ selectedModel, selectedGPU }) {
    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    return (
        <div className="chapter chapter2 animate-in">
            <div className="chapter-header">
                <h2 className="chapter-title">Can We Shrink the KV Cache?</h2>
                <p className="chapter-hook">
                    Chapter 1 showed that the KV cache grows linearly with every token, potentially consuming
                    gigabytes of memory. But there are several clever techniques to reduce this cost —
                    from sharing KV heads across query groups, to borrowing virtual memory concepts from
                    operating systems, to quantizing the cache itself.
                </p>
            </div>

            {/* Section 1: MHA → GQA → MQA */}
            <AttentionModeToggle model={model} gpu={gpu} />

            {/* Bridge */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    GQA reduces the <em>size</em> of the KV cache per token. But there's another source of
                    waste: how the cache is <em>managed</em> in memory across multiple requests.
                </p>
            </section>

            {/* Section 2: PagedAttention */}
            <PagedAttentionViz />

            {/* Bridge */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    We've reduced heads and eliminated fragmentation. Can we go even further?
                    Three more techniques push the boundaries of cache compression.
                </p>
            </section>

            {/* Section 3: KV Optimizations */}
            <KVOptimizations />

            {/* Section 4: GQA Comparison */}
            <GQAComparisonTable selectedModel={selectedModel} />

            {/* Hook to Chapter 3 */}
            <section className="chapter-section">
                <div className="chapter-next-hook glass-card">
                    <p>
                        We've shrunk the <strong>KV cache</strong> — fewer heads, paged allocation, compressed storage.
                        But the attention computation itself is still expensive: every token must attend to
                        every other token, reading K and V from slow GPU memory. Can we make the attention
                        <em> computation</em> itself faster?
                    </p>
                    <p className="chapter-next-question">
                        → Flash Attention: Tiling the computation to fit in fast SRAM
                    </p>
                </div>
            </section>
        </div>
    );
}

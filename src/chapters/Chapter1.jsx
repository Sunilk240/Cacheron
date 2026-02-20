import { useState, useEffect, useRef } from 'react';
import { MODELS, GPUS, kvCacheSize, modelWeightSize, formatBytes } from '../data/modelConfig';
import './Chapter1.css';

// ============================================================
// 1. QKV PROJECTION FLOW
//    Token → Embedding → multiply by W_Q, W_K, W_V → Q, K, V
//    K and V go to cache, Q goes to attention
// ============================================================

function QKVProjectionFlow({ model }) {
    const [selectedToken, setSelectedToken] = useState('Paris');
    const tokens = ['The', 'capital', 'of', 'France', 'is', 'Paris'];

    const qDim = `${model.Hq}×${model.dhead}`;
    const kvDim = `${model.Hkv}×${model.dhead}`;
    const qTotal = model.Hq * model.dhead;
    const kvTotal = model.Hkv * model.dhead;

    return (
        <section className="chapter-section">
            <h3 className="section-title">How Does a Token Become Q, K, and V?</h3>
            <p className="section-desc">
                Inside each transformer layer, every token goes through the same transformation. The token's
                embedding vector (a list of {model.dmodel} numbers) gets multiplied by three separate
                weight matrices — <strong style={{ color: 'var(--accent-warm)' }}>W<sub>Q</sub></strong>,{' '}
                <strong style={{ color: 'var(--text-accent)' }}>W<sub>K</sub></strong>, and{' '}
                <strong style={{ color: 'var(--accent-secondary)' }}>W<sub>V</sub></strong> — to produce
                three different vectors:
            </p>
            <ul className="section-desc" style={{ paddingLeft: '1.5em', marginTop: '-8px' }}>
                <li><strong style={{ color: 'var(--accent-warm)' }}>Query (Q)</strong>: "What am I looking for?"</li>
                <li><strong style={{ color: 'var(--text-accent)' }}>Key (K)</strong>: "What do I contain?"</li>
                <li><strong style={{ color: 'var(--accent-secondary)' }}>Value (V)</strong>: "What information do I carry?"</li>
            </ul>
            <p className="section-desc" style={{ marginTop: '4px' }}>
                Click any token below to see its journey through the three projections.
            </p>

            <div className="qkv-flow glass-card">
                {/* Token selector */}
                <div className="qkv-input-row" style={{ justifyContent: 'center', marginBottom: 'var(--space-md)' }}>
                    {tokens.map(tok => (
                        <span
                            key={tok}
                            className={`kv-tok ${selectedToken === tok ? 'current' : 'processed'}`}
                            onClick={() => setSelectedToken(tok)}
                            style={{ cursor: 'pointer' }}
                        >
                            {tok}
                        </span>
                    ))}
                </div>

                <div className="qkv-flow-diagram">
                    {/* Token → Embedding */}
                    <div className="qkv-input-row">
                        <div className="qkv-token-box">"{selectedToken}"</div>
                        <div className="qkv-embed-arrow">→ embed →</div>
                        <div className="qkv-embed-box">
                            <div className="qkv-embed-label">Embedding vector</div>
                            <div className="qkv-embed-dim">[1 × {model.dmodel}]</div>
                        </div>
                    </div>

                    {/* Split arrow */}
                    <div className="qkv-arrows-down">↓ multiplied by 3 weight matrices ↓</div>

                    {/* Three branches: Q, K, V */}
                    <div className="qkv-projection">
                        {/* Q branch */}
                        <div className="qkv-branch">
                            <div className="qkv-weight-box q-weight">
                                <span className="qkv-weight-name">W<sub>Q</sub></span>
                                <span className="qkv-weight-dim">[{model.dmodel} × {qTotal}]</span>
                            </div>
                            <div className="qkv-times">×</div>
                            <div className="qkv-result-box q-result">
                                Q
                                <span className="qkv-result-dim">[1 × {qTotal}]  ({qDim})</span>
                            </div>
                            <div className="qkv-cache-badge not-cached">
                                ✕ Not cached — recomputed each step
                            </div>
                        </div>

                        {/* K branch */}
                        <div className="qkv-branch">
                            <div className="qkv-weight-box k-weight">
                                <span className="qkv-weight-name">W<sub>K</sub></span>
                                <span className="qkv-weight-dim">[{model.dmodel} × {kvTotal}]</span>
                            </div>
                            <div className="qkv-times">×</div>
                            <div className="qkv-result-box k-result">
                                K
                                <span className="qkv-result-dim">[1 × {kvTotal}]  ({kvDim})</span>
                            </div>
                            <div className="qkv-cache-badge cached">
                                📦 Stored in KV cache
                            </div>
                        </div>

                        {/* V branch */}
                        <div className="qkv-branch">
                            <div className="qkv-weight-box v-weight">
                                <span className="qkv-weight-name">W<sub>V</sub></span>
                                <span className="qkv-weight-dim">[{model.dmodel} × {kvTotal}]</span>
                            </div>
                            <div className="qkv-times">×</div>
                            <div className="qkv-result-box v-result">
                                V
                                <span className="qkv-result-dim">[1 × {kvTotal}]  ({kvDim})</span>
                            </div>
                            <div className="qkv-cache-badge cached">
                                📦 Stored in KV cache
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 2. ATTENTION SCORE COMPUTATION
//    Q × K^T → scores → softmax → weights → × V → output
//    Animated step-by-step matrix multiply
// ============================================================

function AttentionScoreFlow({ model }) {
    const [activeStep, setActiveStep] = useState(0);
    const steps = [
        { label: 'Q × Kᵀ', desc: 'Compute raw attention scores: how much does each token attend to every other token.' },
        { label: '÷ √d', desc: `Scale by √${model.dhead} = ${Math.sqrt(model.dhead).toFixed(1)} to prevent values from becoming too large.` },
        { label: 'Softmax', desc: 'Normalize scores into probabilities that sum to 1 across each row.' },
        { label: '× V', desc: 'Weighted sum of Value vectors — tokens with high attention scores contribute more.' },
    ];

    // Mini example attention scores (4 tokens)
    const tokens = ['The', 'cat', 'sat', 'down'];
    const rawScores = [
        [1.2, 0.3, 0.1, 0.0],
        [0.5, 2.1, 0.2, 0.1],
        [0.3, 0.8, 1.9, 0.3],
        [0.2, 0.4, 0.6, 2.0],
    ];

    const scale = Math.sqrt(model.dhead);
    const scaledScores = rawScores.map(row => row.map(v => v / scale));

    // Softmax per row
    const softmaxScores = scaledScores.map(row => {
        const maxVal = Math.max(...row);
        const exps = row.map(v => Math.exp(v - maxVal));
        const sum = exps.reduce((a, b) => a + b, 0);
        return exps.map(e => e / sum);
    });

    const getDisplayScores = () => {
        if (activeStep === 0) return rawScores;
        if (activeStep === 1) return scaledScores;
        return softmaxScores;
    };

    const getColor = (val, step) => {
        if (step <= 1) {
            // Raw/scaled: blue gradient
            const norm = Math.min(val / 2.5, 1);
            return `rgba(124, 106, 255, ${0.15 + norm * 0.6})`;
        }
        // Softmax: green gradient
        const norm = Math.min(val / 0.6, 1);
        return `rgba(78, 205, 196, ${0.1 + norm * 0.65})`;
    };

    return (
        <section className="chapter-section">
            <h3 className="section-title">How Attention Is Actually Computed</h3>
            <p className="section-desc">
                Now that each token has a Q, K, and V vector, how does the model decide which tokens to pay
                attention to? It uses the <strong>Scaled Dot-Product Attention</strong> formula. This is the
                heart of the transformer — and it happens independently in each attention head, at every layer.
            </p>
            <p className="section-desc" style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-accent)', background: 'var(--bg-tertiary)', padding: '8px 12px', borderRadius: 'var(--radius-sm)', display: 'inline-block' }}>
                Attention(Q, K, V) = softmax(Q × Kᵀ / √d<sub>head</sub>) × V
            </p>
            <p className="section-desc">
                Click through each step below to see how 4 tokens compute their attention scores:
            </p>

            <div className="attn-score-section glass-card">
                {/* Step selector */}
                <div style={{ display: 'flex', gap: 'var(--space-sm)', marginBottom: 'var(--space-lg)', flexWrap: 'wrap' }}>
                    {steps.map((step, i) => (
                        <button
                            key={i}
                            className={`pd-toggle-btn ${activeStep === i ? 'active' : ''}`}
                            onClick={() => setActiveStep(i)}
                        >
                            {i + 1}. {step.label}
                        </button>
                    ))}
                </div>

                <p style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', marginBottom: 'var(--space-md)', lineHeight: 1.5 }}>
                    <strong style={{ color: 'var(--text-primary)' }}>{steps[activeStep].label}:</strong>{' '}
                    {steps[activeStep].desc}
                </p>

                {/* Attention heatmap */}
                <div className="attn-heatmap">
                    <div style={{ fontSize: '10px', color: 'var(--text-muted)', marginBottom: 'var(--space-xs)', textAlign: 'center' }}>
                        {activeStep <= 1 ? 'Raw scores (higher = more attention)' : activeStep === 2 ? 'After softmax (rows sum to 1)' : 'Final attention weights applied to V'}
                    </div>
                    <div className="attn-heatmap-grid">
                        {/* Column headers */}
                        <div className="attn-heatmap-row">
                            <div className="attn-heatmap-header" />
                            {tokens.map(tok => (
                                <div key={tok} className="attn-heatmap-header" style={{ minWidth: '44px', textAlign: 'center', color: 'var(--text-accent)' }}>
                                    {tok}
                                </div>
                            ))}
                        </div>
                        {/* Score rows */}
                        {getDisplayScores().map((row, i) => (
                            <div key={i} className="attn-heatmap-row">
                                <div className="attn-heatmap-header" style={{ color: 'var(--accent-warm)' }}>{tokens[i]}</div>
                                {row.map((val, j) => (
                                    <div
                                        key={j}
                                        className={`attn-heat-cell ${i === j ? 'highlight' : ''}`}
                                        style={{ background: getColor(val, activeStep) }}
                                        title={`${tokens[i]} → ${tokens[j]}: ${val.toFixed(3)}`}
                                    >
                                        {activeStep >= 2 ? (val * 100).toFixed(0) + '%' : val.toFixed(2)}
                                    </div>
                                ))}
                            </div>
                        ))}
                    </div>
                </div>

                {/* Step 3: show V multiplication concept */}
                {activeStep === 3 && (
                    <div style={{ marginTop: 'var(--space-md)', padding: 'var(--space-md)', background: 'rgba(78, 205, 196, 0.06)', borderRadius: 'var(--radius-md)', border: '1px solid rgba(78, 205, 196, 0.15)' }}>
                        <p style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                            <strong style={{ color: 'var(--accent-secondary)' }}>Final multiply by V:</strong>{' '}
                            Each row of attention weights is used as a weighted combination of the V vectors.
                            For example, "sat" gets{' '}
                            <span style={{ color: 'var(--text-accent)', fontFamily: 'var(--font-mono)' }}>
                                {(softmaxScores[2][0] * 100).toFixed(0)}%
                            </span> of V<sub>The</sub> +{' '}
                            <span style={{ color: 'var(--text-accent)', fontFamily: 'var(--font-mono)' }}>
                                {(softmaxScores[2][1] * 100).toFixed(0)}%
                            </span> of V<sub>cat</sub> +{' '}
                            <span style={{ color: 'var(--text-accent)', fontFamily: 'var(--font-mono)' }}>
                                {(softmaxScores[2][2] * 100).toFixed(0)}%
                            </span> of V<sub>sat</sub> +{' '}
                            <span style={{ color: 'var(--text-accent)', fontFamily: 'var(--font-mono)' }}>
                                {(softmaxScores[2][3] * 100).toFixed(0)}%
                            </span> of V<sub>down</sub>.
                            <br /><br />
                            This is how the model "mixes" information from different tokens — tokens with high attention scores
                            contribute more of their value to the output.
                        </p>
                    </div>
                )}
            </div>
        </section>
    );
}


// ============================================================
// 3. KV CACHE TOKEN-BY-TOKEN STEPPER
//    Step through generating tokens, see KV cache grow:
//    each token adds its K,V to cache, Q is discarded
// ============================================================

const SENTENCE = ["The", "capital", "of", "France", "is", "Paris", "."];

function KVCacheStepper({ model }) {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const timerRef = useRef(null);
    const total = SENTENCE.length;

    useEffect(() => {
        if (!isPlaying) return;
        timerRef.current = setInterval(() => {
            setStep(s => {
                if (s >= total) { setIsPlaying(false); return s; }
                return s + 1;
            });
        }, 900);
        return () => clearInterval(timerRef.current);
    }, [isPlaying, total]);

    const handlePlay = () => {
        if (step >= total) { setStep(0); }
        setIsPlaying(p => !p);
    };

    const handleStep = (dir) => {
        setIsPlaying(false);
        setStep(s => Math.max(0, Math.min(total, s + dir)));
    };

    const handleReset = () => { setIsPlaying(false); setStep(0); };

    // Per-token KV size
    const kvPerToken = 2 * model.L * model.Hkv * model.dhead * 2; // bytes, FP16
    const currentKV = step * kvPerToken;
    const kvAtMax = total * kvPerToken;

    return (
        <section className="chapter-section">
            <h3 className="section-title">Why the KV Cache Grows — Token by Token</h3>
            <p className="section-desc">
                Here's the key insight: when generating token <em>n</em>, the model needs
                to attend to <strong>all previous tokens</strong> (1 through <em>n-1</em>).
                Without caching, it would have to recompute K and V for every previous token at every step.
                Instead, the model <strong>stores each token's K and V</strong> in a growing cache.
                Only the new token's Q is computed fresh — it uses the cached K,V to compute attention.
            </p>
            <p className="section-desc">
                Step through below to see the cache grow. Each token adds {model.Hkv} K-vectors and
                {' '}{model.Hkv} V-vectors across all {model.L} layers.
            </p>

            <div className="kv-stepper glass-card">
                {/* Controls */}
                <div className="kv-stepper-controls">
                    <button className="kv-step-btn" onClick={handleReset}>↺</button>
                    <button className="kv-step-btn" onClick={() => handleStep(-1)} disabled={step === 0}>◀</button>
                    <button className="kv-step-btn play-btn" onClick={handlePlay}>
                        {isPlaying ? '⏸' : '▶'}
                    </button>
                    <button className="kv-step-btn" onClick={() => handleStep(1)} disabled={step >= total}>▶</button>
                    <span className="kv-step-info">Token {step}/{total}</span>
                </div>

                {/* Sentence with token states */}
                <div className="kv-sentence">
                    {SENTENCE.map((tok, i) => (
                        <span
                            key={i}
                            className={`kv-tok ${i < step ? (i === step - 1 ? 'current' : 'processed') : 'pending'}`}
                        >
                            {tok}
                        </span>
                    ))}
                </div>

                {/* Cache contents table */}
                <div style={{ overflowX: 'auto' }}>
                    <table className="kv-cache-table">
                        <thead>
                            <tr>
                                <th>Token</th>
                                <th style={{ color: 'var(--text-accent)' }}>K vector</th>
                                <th style={{ color: 'var(--accent-secondary)' }}>V vector</th>
                                <th>Shape (per layer)</th>
                                <th>Size</th>
                            </tr>
                        </thead>
                        <tbody>
                            {step === 0 && (
                                <tr>
                                    <td colSpan={5} style={{ textAlign: 'center', color: 'var(--text-muted)', fontStyle: 'italic', fontFamily: 'var(--font-sans)' }}>
                                        Press play or step forward to start generating tokens...
                                    </td>
                                </tr>
                            )}
                            {SENTENCE.slice(0, step).map((tok, i) => (
                                <tr key={i} className={i === step - 1 ? 'current-row' : ''}>
                                    <td>"{tok}"</td>
                                    <td className="kv-row-k">K<sub>{i + 1}</sub> [{model.Hkv}×{model.dhead}]</td>
                                    <td className="kv-row-v">V<sub>{i + 1}</sub> [{model.Hkv}×{model.dhead}]</td>
                                    <td>[2 × {model.Hkv} × {model.dhead}]</td>
                                    <td>{formatBytes(kvPerToken / model.L)} / layer</td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>

                {/* Accumulator bar */}
                <div className="kv-accumulator">
                    <span className="kv-accum-text">KV Cache:</span>
                    <div className="kv-accum-bar">
                        <div className="kv-accum-fill" style={{ width: `${kvAtMax > 0 ? (currentKV / kvAtMax) * 100 : 0}%` }} />
                    </div>
                    <span className="kv-accum-value">{formatBytes(currentKV)}</span>
                </div>

                <div className="kv-layer-note">
                    {step > 0 && (
                        <>
                            Cache stores K,V for <strong>{step} token{step > 1 ? 's' : ''}</strong> × <strong>{model.L} layers</strong> ={' '}
                            <strong>{step * model.L * 2}</strong> vectors ({step} × {model.L} × 2)
                            <br />
                            Total: <strong>{step}</strong> tokens × <strong>{formatBytes(kvPerToken)}</strong>/token = <strong>{formatBytes(currentKV)}</strong>
                        </>
                    )}
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 4. MULTI-HEAD ATTENTION — HEAD GRID
//    Show Q heads → KV heads grouping (MHA vs GQA vs MQA)
// ============================================================

function HeadGrid({ model }) {
    // Determine GQA grouping
    const qPerKV = model.Hq / model.Hkv;
    const attnType = model.Hq === model.Hkv ? 'MHA' : model.Hkv === 1 ? 'MQA' : 'GQA';

    return (
        <section className="chapter-section">
            <h3 className="section-title">Multi-Head Attention — Why Multiple Heads?</h3>
            <p className="section-desc">
                Instead of computing one big attention, the transformer splits Q, K, and V into
                {' '}<strong>multiple heads</strong>. Each head independently learns to focus on different
                aspects of the input — one head might track syntax, another might track meaning,
                another might track position. The outputs of all heads are concatenated and projected back.
            </p>
            <p className="section-desc">
                {model.name} uses <strong>{model.Hq} Query heads</strong> but only <strong>{model.Hkv} KV heads</strong>.
                {attnType === 'MHA' && (
                    <> This is <strong>Multi-Head Attention (MHA)</strong> — each Q head has its own dedicated K,V head pair. Full expressiveness, but the largest KV cache.</>
                )}
                {attnType === 'GQA' && (
                    <> This is <strong>Grouped-Query Attention (GQA)</strong> — every {qPerKV} Q heads share a single K,V head. This reduces the KV cache by {qPerKV}× while keeping most of the model quality.</>
                )}
                {attnType === 'MQA' && (
                    <> This is <strong>Multi-Query Attention (MQA)</strong> — all Q heads share a single K,V pair. Maximum cache savings, but can reduce quality.</>
                )}
            </p>

            <div className="head-grid-section glass-card">
                <div style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', textAlign: 'center', marginBottom: 'var(--space-md)' }}>
                    {model.name} — {attnType}: {model.Hq} Q heads → {model.Hkv} KV heads ({qPerKV} Q per KV group)
                </div>

                <div className="head-grid">
                    {Array.from({ length: model.Hkv }, (_, g) => (
                        <div key={g} className="head-group">
                            <div className="head-q-row">
                                {Array.from({ length: qPerKV }, (_, q) => (
                                    <div key={q} className="head-dot q-dot" title={`Q head ${g * qPerKV + q + 1}`}>
                                        Q
                                    </div>
                                ))}
                            </div>
                            <div className="head-dot kv-dot" title={`KV head ${g + 1}`}>
                                K,V
                            </div>
                            <span className="head-group-label">G{g + 1}</span>
                        </div>
                    ))}
                </div>

                <div className="head-grid-legend">
                    <div className="head-legend-item">
                        <div className="head-legend-dot q-legend" />
                        <span>Query head ({model.dhead}-dim)</span>
                    </div>
                    <div className="head-legend-item">
                        <div className="head-legend-dot kv-legend" />
                        <span>Shared KV head ({model.dhead}-dim)</span>
                    </div>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 5. KV CACHE SIZE FORMULA DERIVATION
//    Step-by-step with actual numbers
// ============================================================

function KVCacheFormula({ model }) {
    const kvPerToken = 2 * model.L * model.Hkv * model.dhead * 2;
    const kv1024 = kvCacheSize(model, 1024);
    const kv4096 = kvCacheSize(model, 4096);

    return (
        <section className="chapter-section">
            <h3 className="section-title">Deriving the KV Cache Size Formula</h3>
            <p className="section-desc">
                Now let's put it all together. How much memory does the KV cache actually consume?
                We can derive the exact formula from the architecture parameters we've been exploring.
            </p>

            <div className="formula-block glass-card">
                <div className="formula-line">
                    <span className="formula-step-num">1</span>
                    <span className="formula-annotation">Per token, per layer, we store K and V:</span>
                </div>
                <div className="formula-line" style={{ paddingLeft: '28px' }}>
                    <span className="formula-expr">
                        2 <span className="formula-annotation">(K+V)</span> × H<sub>kv</sub> × d<sub>head</sub> × bytes
                    </span>
                </div>
                <div className="formula-line" style={{ paddingLeft: '28px' }}>
                    <span className="formula-expr">
                        = 2 × {model.Hkv} × {model.dhead} × 2 <span className="formula-annotation">(FP16)</span>
                    </span>
                    <span className="formula-equals">=</span>
                    <span className="formula-expr" style={{ color: 'var(--text-accent)' }}>
                        {formatBytes(2 * model.Hkv * model.dhead * 2)} / token / layer
                    </span>
                </div>

                <div className="formula-line" style={{ marginTop: 'var(--space-sm)' }}>
                    <span className="formula-step-num">2</span>
                    <span className="formula-annotation">Multiply across all {model.L} layers:</span>
                </div>
                <div className="formula-line" style={{ paddingLeft: '28px' }}>
                    <span className="formula-expr">
                        {formatBytes(2 * model.Hkv * model.dhead * 2)} × {model.L} layers
                    </span>
                    <span className="formula-equals">=</span>
                    <span className="formula-expr" style={{ color: 'var(--text-accent)' }}>
                        {formatBytes(kvPerToken)} / token
                    </span>
                </div>

                <div className="formula-line" style={{ marginTop: 'var(--space-sm)' }}>
                    <span className="formula-step-num">3</span>
                    <span className="formula-annotation">For a sequence of N tokens:</span>
                </div>
                <div className="formula-result">
                    <div className="formula-line">
                        <span className="formula-expr">
                            KV Cache = N × 2 × L × H<sub>kv</sub> × d<sub>head</sub> × sizeof(FP16)
                        </span>
                    </div>
                </div>

                <div className="formula-line" style={{ marginTop: 'var(--space-md)' }}>
                    <span className="formula-step-num">✓</span>
                    <span className="formula-annotation">Examples for {model.name}:</span>
                </div>
                <div className="formula-line" style={{ paddingLeft: '28px' }}>
                    <span className="formula-expr">
                        1,024 tokens → {formatBytes(kv1024)}
                    </span>
                </div>
                <div className="formula-line" style={{ paddingLeft: '28px' }}>
                    <span className="formula-expr">
                        4,096 tokens → {formatBytes(kv4096)}
                    </span>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 6. 5-MODEL COMPARISON TABLE
// ============================================================

function ModelComparisonTable({ selectedModel }) {
    const modelKeys = Object.keys(MODELS);

    return (
        <section className="chapter-section">
            <h3 className="section-title">How Do Different Models Compare?</h3>
            <p className="section-desc">
                The KV cache size varies dramatically across models. More layers, more KV heads,
                and larger head dimensions all increase the cache. Models using GQA have a significant
                advantage because they share KV heads across multiple query heads.
            </p>

            <div className="glass-card" style={{ overflowX: 'auto' }}>
                <table className="model-compare-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Layers</th>
                            <th>Q Heads</th>
                            <th>KV Heads</th>
                            <th>d_head</th>
                            <th>Type</th>
                            <th>KV/token</th>
                            <th>KV @ 4K tokens</th>
                        </tr>
                    </thead>
                    <tbody>
                        {modelKeys.map(key => {
                            const m = MODELS[key];
                            const kvPT = 2 * m.L * m.Hkv * m.dhead * 2;
                            const kv4k = kvCacheSize(m, 4096);
                            const type = m.Hq === m.Hkv ? 'MHA' : m.Hkv === 1 ? 'MQA' : 'GQA';
                            return (
                                <tr key={key} className={key === selectedModel ? 'selected-row' : ''}>
                                    <td style={{ fontFamily: 'var(--font-sans)', fontWeight: 'var(--fw-semibold)' }}>{m.name}</td>
                                    <td>{m.L}</td>
                                    <td>{m.Hq}</td>
                                    <td>{m.Hkv}</td>
                                    <td>{m.dhead}</td>
                                    <td>
                                        <span className={type === 'MHA' ? 'mha-badge' : 'gqa-badge'}>{type}</span>
                                    </td>
                                    <td>{formatBytes(kvPT)}</td>
                                    <td>{formatBytes(kv4k)}</td>
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
// CHAPTER 1 — Main Component
// ============================================================

export default function Chapter1({ selectedModel, selectedGPU }) {
    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    return (
        <div className="chapter chapter1 animate-in">
            <div className="chapter-header">
                <h2 className="chapter-title">What Is Attention, and What Does It Store?</h2>
                <p className="chapter-hook">
                    The Prologue showed you that inference is memory-bound, and that the KV cache
                    is the dynamic piece that grows with every token. Now let's open up a single
                    transformer layer and see exactly how Q, K, and V are computed, how attention
                    scores are calculated, and why K and V — but not Q — must be cached.
                </p>
            </div>

            {/* Section 1: QKV Projection Flow */}
            <QKVProjectionFlow model={model} />

            {/* Transition */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    Now we have Q, K, and V for each token. The next question is: how does the model use
                    these three vectors to decide which tokens to pay attention to?
                </p>
            </section>

            {/* Section 2: Attention Score Computation */}
            <AttentionScoreFlow model={model} />

            {/* Transition */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    Each new token needs K and V from <em>all previous tokens</em> to compute attention.
                    Recomputing them every time would be wasteful — so we cache them. Let's see this in action.
                </p>
            </section>

            {/* Section 3: KV Cache Token Stepper */}
            <KVCacheStepper model={model} />

            {/* Section 4: Multi-Head Grid */}
            <HeadGrid model={model} />

            {/* Section 5: KV Cache Formula */}
            <KVCacheFormula model={model} />

            {/* Section 6: Model Comparison */}
            <ModelComparisonTable selectedModel={selectedModel} />

            {/* Hook to Chapter 2 */}
            <section className="chapter-section">
                <div className="chapter-next-hook glass-card">
                    <p>
                        The KV cache grows linearly with sequence length and can consume gigabytes of memory.
                        For {model.name} at 4,096 tokens, that's <strong>{formatBytes(kvCacheSize(model, 4096))}</strong> — on
                        top of the {formatBytes(modelWeightSize(model, 2))} model weights.
                    </p>
                    <p className="chapter-next-question">
                        → Can we shrink this cache? (GQA, PagedAttention, Quantization)
                    </p>
                </div>
            </section>
        </div>
    );
}

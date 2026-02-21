import { useState, useEffect, useRef } from 'react';
import { MODELS, GPUS, kvCacheSize, modelWeightSize, formatBytes } from '../data/modelConfig';
import SpeedControl from '../components/SpeedControl';
import AnimationCommentary from '../components/AnimationCommentary';
import SmallModelNote from '../components/SmallModelNote';
import './Chapter1.css';

// ============================================================
// 1. QKV PROJECTION FLOW — AUTO-PLAYING
//    Token → Embedding → multiply by W_Q, W_K, W_V → Q, K, V
//    K and V go to cache, Q goes to attention
//    Auto-plays through each token with commentary
// ============================================================

function QKVProjectionFlow({ model }) {
    const tokens = ['The', 'capital', 'of', 'France', 'is', 'Paris'];
    const [tokenIdx, setTokenIdx] = useState(0);
    const [phase, setPhase] = useState(0); // 0=embed, 1=project, 2=cache, 3=done
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState(1);
    const timerRef = useRef(null);

    const selectedToken = tokens[tokenIdx];
    const qDim = `${model.Hq}×${model.dhead}`;
    const kvDim = `${model.Hkv}×${model.dhead}`;
    const qTotal = model.Hq * model.dhead;
    const kvTotal = model.Hkv * model.dhead;

    // Commentary messages for each phase
    const getCommentary = () => {
        if (phase === 0) return `Token "${selectedToken}" enters the pipeline. It's looked up in the embedding table to produce a ${model.dmodel}-dimensional vector.`;
        if (phase === 1) return `The embedding is multiplied by three weight matrices: W_Q (${model.dmodel}×${qTotal}), W_K (${model.dmodel}×${kvTotal}), and W_V (${model.dmodel}×${kvTotal}). This produces the Query, Key, and Value vectors.`;
        if (phase === 2) return `Key and Value vectors are stored in the KV cache — they'll be needed by ALL future tokens. Query is used for this step's attention computation and then discarded.`;
        return `Token "${selectedToken}" processed! K,V cached. Moving to the next token...`;
    };

    // Auto-play logic
    useEffect(() => {
        if (!isPlaying) { clearInterval(timerRef.current); return; }
        const interval = 1200 / speed;
        timerRef.current = setInterval(() => {
            setPhase(prev => {
                if (prev >= 3) {
                    setTokenIdx(ti => {
                        if (ti >= tokens.length - 1) { setIsPlaying(false); return ti; }
                        return ti + 1;
                    });
                    return 0;
                }
                return prev + 1;
            });
        }, interval);
        return () => clearInterval(timerRef.current);
    }, [isPlaying, speed, tokens.length]);

    const handlePlay = () => {
        if (tokenIdx >= tokens.length - 1 && phase >= 3) { setTokenIdx(0); setPhase(0); }
        setIsPlaying(p => !p);
    };

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

            <div className="qkv-flow glass-card">
                {/* Controls */}
                <div className="qkv-controls">
                    <button className="stepper-btn play" onClick={handlePlay} title={isPlaying ? 'Pause' : 'Play'}>
                        {isPlaying ? '⏸' : '▶'}
                    </button>
                    <SpeedControl speed={speed} onSpeedChange={setSpeed} />
                    <span className="stepper-progress">Token {tokenIdx + 1}/{tokens.length}</span>
                </div>

                {/* Token selector */}
                <div className="qkv-input-row" style={{ justifyContent: 'center', marginBottom: 'var(--space-md)' }}>
                    {tokens.map((tok, i) => (
                        <span
                            key={tok}
                            className={`kv-tok ${i === tokenIdx ? 'current' : i < tokenIdx ? 'processed' : 'pending'}`}
                            onClick={() => { setIsPlaying(false); setTokenIdx(i); setPhase(0); }}
                            style={{ cursor: 'pointer' }}
                        >
                            {tok}
                        </span>
                    ))}
                </div>

                <div className="qkv-flow-diagram">
                    {/* Token → Embedding */}
                    <div className="qkv-input-row">
                        <div className={`qkv-token-box ${phase >= 0 ? 'stage-active' : ''}`}>"{selectedToken}"</div>
                        <div className={`qkv-embed-arrow ${phase >= 0 ? 'active' : ''}`}>→ embed →</div>
                        <div className={`qkv-embed-box ${phase >= 0 ? 'stage-active' : ''}`}>
                            <div className="qkv-embed-label">Embedding vector</div>
                            <div className="qkv-embed-dim">[1 × {model.dmodel}]</div>
                        </div>
                    </div>

                    {/* Split arrow */}
                    <div className={`qkv-arrows-down ${phase >= 1 ? 'active' : ''}`}>↓ multiplied by 3 weight matrices ↓</div>

                    {/* Three branches: Q, K, V */}
                    <div className="qkv-projection">
                        {/* Q branch */}
                        <div className={`qkv-branch ${phase < 1 ? 'dim' : ''}`}>
                            <div className={`qkv-weight-box q-weight ${phase === 1 ? 'computing' : ''}`}>
                                <span className="qkv-weight-name">W<sub>Q</sub></span>
                                <span className="qkv-weight-dim">[{model.dmodel} × {qTotal}]</span>
                            </div>
                            <div className="qkv-times">×</div>
                            <div className={`qkv-result-box q-result ${phase >= 1 ? 'visible' : ''}`}>
                                Q
                                <span className="qkv-result-dim">[1 × {qTotal}]  ({qDim})</span>
                            </div>
                            <div className={`qkv-cache-badge not-cached ${phase >= 2 ? 'visible' : ''}`}>
                                ✕ Not cached — recomputed each step
                            </div>
                        </div>

                        {/* K branch */}
                        <div className={`qkv-branch ${phase < 1 ? 'dim' : ''}`}>
                            <div className={`qkv-weight-box k-weight ${phase === 1 ? 'computing' : ''}`}>
                                <span className="qkv-weight-name">W<sub>K</sub></span>
                                <span className="qkv-weight-dim">[{model.dmodel} × {kvTotal}]</span>
                            </div>
                            <div className="qkv-times">×</div>
                            <div className={`qkv-result-box k-result ${phase >= 1 ? 'visible' : ''}`}>
                                K
                                <span className="qkv-result-dim">[1 × {kvTotal}]  ({kvDim})</span>
                            </div>
                            <div className={`qkv-cache-badge cached ${phase >= 2 ? 'visible' : ''}`}>
                                📦 Stored in KV cache
                            </div>
                        </div>

                        {/* V branch */}
                        <div className={`qkv-branch ${phase < 1 ? 'dim' : ''}`}>
                            <div className={`qkv-weight-box v-weight ${phase === 1 ? 'computing' : ''}`}>
                                <span className="qkv-weight-name">W<sub>V</sub></span>
                                <span className="qkv-weight-dim">[{model.dmodel} × {kvTotal}]</span>
                            </div>
                            <div className="qkv-times">×</div>
                            <div className={`qkv-result-box v-result ${phase >= 1 ? 'visible' : ''}`}>
                                V
                                <span className="qkv-result-dim">[1 × {kvTotal}]  ({kvDim})</span>
                            </div>
                            <div className={`qkv-cache-badge cached ${phase >= 2 ? 'visible' : ''}`}>
                                📦 Stored in KV cache
                            </div>
                        </div>
                    </div>
                </div>

                {/* Commentary */}
                <AnimationCommentary text={getCommentary()} icon="🔍" />
            </div>
        </section>
    );
}


// ============================================================
// 2. ATTENTION SCORE COMPUTATION
//    Q × K^T → scores → softmax → weights → × V → output
//    Animated step-by-step matrix multiply with auto-animate option
// ============================================================

function AttentionScoreFlow({ model }) {
    const [activeStep, setActiveStep] = useState(0);
    const [isAnimating, setIsAnimating] = useState(false);
    const [animRow, setAnimRow] = useState(-1);
    const [animCol, setAnimCol] = useState(-1);
    const timerRef = useRef(null);

    const steps = [
        { label: 'Q × Kᵀ', desc: 'Compute raw attention scores: how much does each token attend to every other token.' },
        { label: '÷ √d', desc: `Scale by √${model.dhead} = ${Math.sqrt(model.dhead).toFixed(1)} to prevent values from becoming too large.` },
        { label: 'Softmax', desc: 'Normalize scores into probabilities that sum to 1 across each row.' },
        { label: '× V', desc: 'Weighted sum of Value vectors — tokens with high attention scores contribute more.' },
    ];

    const tokens = ['The', 'cat', 'sat', 'down'];
    const rawScores = [
        [1.2, 0.3, 0.1, 0.0],
        [0.5, 2.1, 0.2, 0.1],
        [0.3, 0.8, 1.9, 0.3],
        [0.2, 0.4, 0.6, 2.0],
    ];

    const scale = Math.sqrt(model.dhead);
    const scaledScores = rawScores.map(row => row.map(v => v / scale));
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
            const norm = Math.min(val / 2.5, 1);
            return `rgba(124, 106, 255, ${0.15 + norm * 0.6})`;
        }
        const norm = Math.min(val / 0.6, 1);
        return `rgba(78, 205, 196, ${0.1 + norm * 0.65})`;
    };

    // Animate heatmap fill cell by cell
    const handleAnimate = () => {
        if (isAnimating) { clearInterval(timerRef.current); setIsAnimating(false); setAnimRow(-1); setAnimCol(-1); return; }
        setIsAnimating(true);
        setActiveStep(0);
        let r = 0, c = 0;
        timerRef.current = setInterval(() => {
            setAnimRow(r); setAnimCol(c);
            c++;
            if (c >= 4) { c = 0; r++; }
            if (r >= 4) { clearInterval(timerRef.current); setIsAnimating(false); setAnimRow(-1); setAnimCol(-1); }
        }, 200);
    };

    useEffect(() => () => clearInterval(timerRef.current), []);

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

            <div className="attn-score-section glass-card">
                <div style={{ display: 'flex', gap: 'var(--space-sm)', marginBottom: 'var(--space-lg)', flexWrap: 'wrap', alignItems: 'center' }}>
                    {steps.map((step, i) => (
                        <button
                            key={i}
                            className={`pd-toggle-btn ${activeStep === i ? 'active' : ''}`}
                            onClick={() => setActiveStep(i)}
                        >
                            {i + 1}. {step.label}
                        </button>
                    ))}
                    <button className="stepper-btn" onClick={handleAnimate} title={isAnimating ? 'Stop' : 'Animate cells'} style={{ marginLeft: 'auto' }}>
                        {isAnimating ? '⏹' : '🎬'}
                    </button>
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
                        <div className="attn-heatmap-row">
                            <div className="attn-heatmap-header" />
                            {tokens.map(tok => (
                                <div key={tok} className="attn-heatmap-header" style={{ minWidth: '44px', textAlign: 'center', color: 'var(--text-accent)' }}>
                                    {tok}
                                </div>
                            ))}
                        </div>
                        {getDisplayScores().map((row, i) => (
                            <div key={i} className="attn-heatmap-row">
                                <div className="attn-heatmap-header" style={{ color: 'var(--accent-warm)' }}>{tokens[i]}</div>
                                {row.map((val, j) => {
                                    const isAnimCell = isAnimating && (i < animRow || (i === animRow && j <= animCol));
                                    return (
                                        <div
                                            key={j}
                                            className={`attn-heat-cell ${i === j ? 'highlight' : ''} ${isAnimCell ? 'cell-animate' : ''}`}
                                            style={{ background: isAnimating && !isAnimCell ? 'var(--bg-tertiary)' : getColor(val, activeStep) }}
                                            title={`${tokens[i]} → ${tokens[j]}: ${val.toFixed(3)}`}
                                        >
                                            {activeStep >= 2 ? (val * 100).toFixed(0) + '%' : val.toFixed(2)}
                                        </div>
                                    );
                                })}
                            </div>
                        ))}
                    </div>
                </div>

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
//    LONGER PARAGRAPH (~60 tokens) with auto-play, speed control,
//    and running commentary
// ============================================================

const LONG_PARAGRAPH = [
    "Large", "language", "models", "process", "text", "by", "breaking", "it", "into", "tokens", ".",
    "Each", "token", "passes", "through", "dozens", "of", "transformer", "layers", ",",
    "where", "it", "attends", "to", "every", "previous", "token", "using", "cached",
    "Key", "and", "Value", "vectors", ".",
    "As", "the", "sequence", "grows", "longer", ",",
    "this", "cache", "grows", "linearly", "—",
    "eventually", "consuming", "more", "memory", "than", "the",
    "model", "weights", "themselves", "."
];

function KVCacheStepper({ model, gpu }) {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState(1);
    const timerRef = useRef(null);
    const total = LONG_PARAGRAPH.length;
    const containerRef = useRef(null);

    useEffect(() => {
        if (!isPlaying) return;
        const interval = 400 / speed;
        timerRef.current = setInterval(() => {
            setStep(s => {
                if (s >= total) { setIsPlaying(false); return s; }
                return s + 1;
            });
        }, interval);
        return () => clearInterval(timerRef.current);
    }, [isPlaying, total, speed]);

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
    const kvPerToken = 2 * model.L * model.Hkv * model.dhead * 2;
    const currentKV = step * kvPerToken;
    const kvAtMax = total * kvPerToken;

    // GPU overflow check
    const budgetBytes = gpu.budget_mb * 1024 * 1024;
    const weightBytes = modelWeightSize(model, 2);
    const totalMemory = weightBytes + currentKV;
    const isOverflow = totalMemory > budgetBytes;

    const getCommentary = () => {
        if (step === 0) return 'Press play to generate tokens one by one. Watch the KV cache grow as each token adds its K and V vectors.';
        const tok = LONG_PARAGRAPH[step - 1];
        return `Token ${step}/${total}: "${tok}" — added K,V vectors across all ${model.L} layers (${model.Hkv} KV heads × ${model.dhead} dims). Cache now holds ${step} tokens = ${formatBytes(currentKV)}.${isOverflow ? ' ⚠️ OVERFLOW: Total memory exceeds GPU budget!' : ''}`;
    };

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
                Watch the cache grow with a longer passage ({total} tokens) to see the memory impact clearly.
            </p>

            <SmallModelNote kvBytes={kvAtMax} />

            <div className="kv-stepper glass-card" ref={containerRef}>
                {/* Controls */}
                <div className="kv-stepper-controls">
                    <button className="kv-step-btn" onClick={handleReset}>↺</button>
                    <button className="kv-step-btn" onClick={() => handleStep(-1)} disabled={step === 0}>◀</button>
                    <button className="kv-step-btn play-btn" onClick={handlePlay}>
                        {isPlaying ? '⏸' : '▶'}
                    </button>
                    <button className="kv-step-btn" onClick={() => handleStep(1)} disabled={step >= total}>▶</button>
                    <SpeedControl speed={speed} onSpeedChange={setSpeed} />
                    <span className="kv-step-info">Token {step}/{total}</span>
                </div>

                {/* Sentence with token states */}
                <div className="kv-sentence">
                    {LONG_PARAGRAPH.map((tok, i) => (
                        <span
                            key={i}
                            className={`kv-tok ${i < step ? (i === step - 1 ? 'current' : 'processed') : 'pending'}`}
                        >
                            {tok}
                        </span>
                    ))}
                </div>

                {/* Accumulator bar with GPU limit marker */}
                <div className="kv-accumulator">
                    <span className="kv-accum-text">KV Cache:</span>
                    <div className="kv-accum-bar">
                        <div className={`kv-accum-fill ${isOverflow ? 'overflow' : ''}`} style={{ width: `${kvAtMax > 0 ? Math.min((currentKV / kvAtMax) * 100, 100) : 0}%` }} />
                    </div>
                    <span className={`kv-accum-value ${isOverflow ? 'overflow-text' : ''}`}>{formatBytes(currentKV)}</span>
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

                <AnimationCommentary text={getCommentary()} icon="📊" />
            </div>
        </section>
    );
}


// ============================================================
// 3b. KV CACHE OVERFLOW DEMO
//     Show exact token count where cache exceeds GPU memory.
//     Animated counter + memory bar hitting the limit.
// ============================================================

function KVOverflowDemo({ model, gpu }) {
    const kvPerToken = 2 * model.L * model.Hkv * model.dhead * 2;
    const weightBytes = modelWeightSize(model, 2);
    const budgetBytes = gpu.budget_mb * 1024 * 1024;
    const kvBudget = budgetBytes - weightBytes;
    const maxTokens = Math.max(0, Math.floor(kvBudget / kvPerToken));

    const [displayTokens, setDisplayTokens] = useState(0);
    const [isAnimating, setIsAnimating] = useState(false);
    const timerRef = useRef(null);

    const handleAnimate = () => {
        if (isAnimating) { clearInterval(timerRef.current); setIsAnimating(false); return; }
        setIsAnimating(true);
        setDisplayTokens(0);
        const target = Math.min(maxTokens + 100, maxTokens * 1.1);
        const increment = Math.max(1, Math.floor(target / 80));
        timerRef.current = setInterval(() => {
            setDisplayTokens(prev => {
                const next = prev + increment;
                if (next >= target) {
                    clearInterval(timerRef.current);
                    setIsAnimating(false);
                    return Math.ceil(target);
                }
                return next;
            });
        }, 30);
    };

    useEffect(() => () => clearInterval(timerRef.current), []);

    const currentKV = displayTokens * kvPerToken;
    const totalMemory = weightBytes + currentKV;
    const isOverflow = totalMemory > budgetBytes;
    const barPct = Math.min((totalMemory / (budgetBytes * 1.2)) * 100, 100);

    // If the model doesn't even fit, show a different message
    if (weightBytes >= budgetBytes) {
        return (
            <section className="chapter-section">
                <h3 className="section-title">KV Cache Overflow Point</h3>
                <div className="glass-card" style={{ textAlign: 'center', padding: 'var(--space-xl)' }}>
                    <p style={{ color: 'var(--accent-warm)', fontSize: 'var(--fs-md)', fontWeight: 'var(--fw-bold)' }}>
                        ❌ {model.name} ({formatBytes(weightBytes)}) doesn't even fit on {gpu.name} ({formatBytes(budgetBytes)}).
                    </p>
                    <p style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-sm)', marginTop: 'var(--space-sm)' }}>
                        Try a larger GPU or smaller model.
                    </p>
                </div>
            </section>
        );
    }

    return (
        <section className="chapter-section">
            <h3 className="section-title">KV Cache Overflow — When Does Memory Run Out?</h3>
            <p className="section-desc">
                With {model.name} on {gpu.name}, the model weights take {formatBytes(weightBytes)}, leaving{' '}
                <strong>{formatBytes(kvBudget)}</strong> for the KV cache. At {formatBytes(kvPerToken)} per token,
                that gives you at most <strong>{maxTokens.toLocaleString()} tokens</strong> before memory runs out.
            </p>

            <div className="glass-card" style={{ textAlign: 'center', padding: 'var(--space-lg)' }}>
                <button className="stepper-btn play" onClick={handleAnimate} style={{ marginBottom: 'var(--space-md)' }}>
                    {isAnimating ? '⏹ Stop' : '▶ Watch Memory Fill Up'}
                </button>

                <div className="overflow-counter">
                    <span className="overflow-token-count" style={{ color: isOverflow ? 'var(--accent-warm)' : 'var(--text-accent)' }}>
                        {displayTokens.toLocaleString()}
                    </span>
                    <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>tokens</span>
                </div>

                {/* Memory bar */}
                <div style={{ position: 'relative', height: '40px', background: 'var(--bg-tertiary)', borderRadius: 'var(--radius-md)', overflow: 'hidden', border: '1px solid var(--border-subtle)', margin: 'var(--space-md) 0' }}>
                    {/* Weights portion */}
                    <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${(weightBytes / (budgetBytes * 1.2)) * 100}%`, background: 'rgba(124, 106, 255, 0.25)', borderRight: '2px solid var(--accent-primary)' }}>
                        <span style={{ fontSize: '9px', padding: '2px 4px', color: 'var(--text-accent)', whiteSpace: 'nowrap' }}>Weights</span>
                    </div>
                    {/* KV portion */}
                    <div style={{ position: 'absolute', left: `${(weightBytes / (budgetBytes * 1.2)) * 100}%`, top: 0, height: '100%', width: `${Math.max(0, barPct - (weightBytes / (budgetBytes * 1.2)) * 100)}%`, background: isOverflow ? 'rgba(255, 107, 107, 0.3)' : 'rgba(78, 205, 196, 0.25)', transition: 'width 30ms, background 300ms' }}>
                        <span style={{ fontSize: '9px', padding: '2px 4px', color: isOverflow ? 'var(--accent-warm)' : 'var(--accent-secondary)', whiteSpace: 'nowrap' }}>KV Cache</span>
                    </div>
                    {/* Budget line */}
                    <div style={{ position: 'absolute', left: `${(budgetBytes / (budgetBytes * 1.2)) * 100}%`, top: 0, height: '100%', width: '2px', background: 'var(--text-muted)' }}>
                        <span style={{ position: 'absolute', top: '-18px', transform: 'translateX(-50%)', fontSize: '9px', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                            {gpu.budget_mb >= 1024 ? `${gpu.budget_mb / 1024} GB` : `${gpu.budget_mb} MB`}
                        </span>
                    </div>
                </div>

                {/* OOM banner */}
                {isOverflow && (
                    <div style={{
                        background: 'rgba(255, 107, 107, 0.12)',
                        border: '1px solid rgba(255, 107, 107, 0.3)',
                        borderRadius: 'var(--radius-md)',
                        padding: 'var(--space-sm) var(--space-md)',
                        color: 'var(--accent-warm)',
                        fontWeight: 'var(--fw-bold)',
                        fontSize: 'var(--fs-sm)',
                        animation: 'fadeInScale 300ms ease'
                    }}>
                        ⛔ Out of Memory! Total: {formatBytes(totalMemory)} / {formatBytes(budgetBytes)}
                    </div>
                )}

                <div style={{ marginTop: 'var(--space-md)', fontSize: 'var(--fs-xs)', color: 'var(--text-muted)' }}>
                    Maximum context length on {gpu.name}: <strong style={{ color: 'var(--text-accent)' }}>{maxTokens.toLocaleString()} tokens</strong>
                    {' '}≈ {Math.floor(maxTokens * 0.75).toLocaleString()} words
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
                    The introduction showed you that inference is memory-bound, and that the KV cache
                    is the dynamic piece that grows with every token. Now let's open up a single
                    transformer layer and see exactly how Q, K, and V are computed, how attention
                    scores are calculated, and why K and V — but not Q — must be cached.
                </p>
            </div>

            <QKVProjectionFlow model={model} />

            {/* Transition */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    Now we have Q, K, and V for each token. The next question is: how does the model use
                    these three vectors to decide which tokens to pay attention to?
                </p>
            </section>

            <AttentionScoreFlow model={model} />

            {/* Transition */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    Each new token needs K and V from <em>all previous tokens</em> to compute attention.
                    Recomputing them every time would be wasteful — so we cache them. Let's see this in action.
                </p>
            </section>

            <KVCacheStepper model={model} gpu={gpu} />

            {/* NEW: Overflow Demo */}
            <KVOverflowDemo model={model} gpu={gpu} />

            <HeadGrid model={model} />
            <KVCacheFormula model={model} />
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

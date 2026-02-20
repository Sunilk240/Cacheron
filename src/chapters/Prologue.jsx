import { useState, useEffect, useRef, useCallback } from 'react';
import { MODELS, GPUS, kvCacheSize, modelWeightSize, formatBytes } from '../data/modelConfig';
import './Prologue.css';

// ============================================================
// 1. TRANSFORMER ARCHITECTURE FLOW
// Animated block diagram: Embedding → Layers → LM Head → Token
// Data dots flow through, autoregressive loop, model-aware layers
// ============================================================

function TransformerFlow({ model }) {
    const [hoveredNode, setHoveredNode] = useState(null);
    const layerCount = model.L;
    // Show first 2 layers, dots, last layer
    const visibleLayers = layerCount > 4
        ? [0, 1, 'dots', layerCount - 1]
        : Array.from({ length: layerCount }, (_, i) => i);

    return (
        <section className="chapter-section">
            <h3 className="section-title">The Transformer Pipeline</h3>
            <p className="section-desc">
                Every large language model — from the smallest 135M-parameter model to GPT-4 — is built on the same
                architecture: the <strong>Transformer</strong>. It's a stack of identical layers, each containing a
                self-attention mechanism and a feed-forward network. Your prompt enters on the left as a sequence of
                tokens, flows through <strong>{layerCount} transformer layers</strong>, and a single next token
                exits on the right. That token then feeds back as input to generate the next one — this is called
                <strong>autoregressive generation</strong>, and it's why tokens appear one at a time.
            </p>
            <p className="section-desc">
                Hover over any node below to see {model.name}'s actual dimensions at that stage.
            </p>

            <div className="arch-flow glass-card">
                <div className="arch-flow-title">
                    {model.name} — {layerCount} layers, {model.dmodel}-dim
                </div>

                <div className="arch-pipeline">
                    {/* Flow track behind everything */}
                    <div className="arch-flow-track">
                        <div className="arch-data-dot" />
                        <div className="arch-data-dot" />
                        <div className="arch-data-dot" />
                    </div>

                    {/* Input */}
                    <div className="arch-node embed" onMouseEnter={() => setHoveredNode('embed')} onMouseLeave={() => setHoveredNode(null)}>
                        <span className="arch-node-label">Embedding</span>
                        <span className="arch-node-dim">{model.dmodel}-d</span>
                        {hoveredNode === 'embed' && (
                            <div className="arch-layer-detail">
                                Each token → {model.dmodel}-dim vector.<br />
                                Vocabulary lookup + position encoding.
                            </div>
                        )}
                    </div>

                    <div className="arch-arrow active" />

                    {/* Transformer Layers */}
                    <div className="arch-layers-group">
                        {visibleLayers.map((layerIdx, i) => {
                            if (layerIdx === 'dots') {
                                return (
                                    <div key="dots" style={{ display: 'flex', alignItems: 'center' }}>
                                        <div className="arch-arrow" />
                                        <div className="arch-layers-dots">
                                            <span /><span /><span />
                                            <span style={{ fontSize: '10px', color: 'var(--text-muted)', margin: '0 4px' }}>
                                                ×{layerCount - 3}
                                            </span>
                                        </div>
                                        <div className="arch-arrow" />
                                    </div>
                                );
                            }
                            return (
                                <div key={layerIdx} style={{ display: 'flex', alignItems: 'center' }}>
                                    {i > 0 && layerIdx !== 'dots' && visibleLayers[i - 1] !== 'dots' && <div className="arch-arrow" />}
                                    <div
                                        className={`arch-node layer`}
                                        onMouseEnter={() => setHoveredNode(`layer-${layerIdx}`)}
                                        onMouseLeave={() => setHoveredNode(null)}
                                    >
                                        <span className="arch-node-label">Layer {layerIdx + 1}</span>
                                        <span className="arch-node-dim">Attn+FFN</span>
                                        {hoveredNode === `layer-${layerIdx}` && (
                                            <div className="arch-layer-detail">
                                                <div className="detail-row">
                                                    <span className="detail-label">Attention heads:</span>
                                                    <span className="detail-value">{model.Hq} (Q) / {model.Hkv} (KV)</span>
                                                </div>
                                                <div className="detail-row">
                                                    <span className="detail-label">Head dim:</span>
                                                    <span className="detail-value">{model.dhead}</span>
                                                </div>
                                                <div className="detail-row">
                                                    <span className="detail-label">Attention:</span>
                                                    <span className="detail-value">{model.attnType}</span>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    <div className="arch-arrow active" />

                    {/* LM Head */}
                    <div className="arch-node lm-head" onMouseEnter={() => setHoveredNode('lm-head')} onMouseLeave={() => setHoveredNode(null)}>
                        <span className="arch-node-label">LM Head</span>
                        <span className="arch-node-dim">→ vocab</span>
                        {hoveredNode === 'lm-head' && (
                            <div className="arch-layer-detail">
                                Projects {model.dmodel}-dim → vocabulary logits.<br />
                                Softmax → probability of next token.
                            </div>
                        )}
                    </div>

                    <div className="arch-arrow active" />

                    {/* Output */}
                    <div className="arch-node token-out">
                        <span className="arch-node-label">Next Token</span>
                        <span className="arch-node-dim">↺ feeds back</span>
                    </div>
                </div>

                <div className="arch-loop">
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                        ↺ Each generated token feeds back as input — this loop repeats for every token in the response
                    </span>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 2. TOKEN GENERATION STEPPER
// Step through token generation, see pipeline stages per token,
// watch KV cache grow block by block
// ============================================================

const EXAMPLE_PROMPT = ["What", "is", "the", "capital", "of", "France", "?"];
const EXAMPLE_RESPONSE = ["The", "capital", "of", "France", "is", "Paris", "."];

function TokenStepper({ model }) {
    const [step, setStep] = useState(0); // 0 = prompt only, 1..7 = generating tokens
    const [isPlaying, setIsPlaying] = useState(false);
    const [pipelineStage, setPipelineStage] = useState(0); // 0-4 stages within each step
    const timerRef = useRef(null);
    const totalSteps = EXAMPLE_RESPONSE.length;

    // Pipeline stages for each token
    const stages = [
        { name: 'Embed', val: `${model.dmodel}-d` },
        { name: `${model.L} Layers`, val: 'Attn+FFN' },
        { name: 'LM Head', val: 'logits' },
        { name: 'Softmax', val: 'prob' },
        { name: 'Token', val: step > 0 ? EXAMPLE_RESPONSE[step - 1] : '...' },
    ];

    // Auto-play logic
    useEffect(() => {
        if (!isPlaying) return;
        timerRef.current = setInterval(() => {
            setPipelineStage(prev => {
                if (prev >= 4) {
                    // Move to next token
                    setStep(s => {
                        if (s >= totalSteps) {
                            setIsPlaying(false);
                            return s;
                        }
                        return s + 1;
                    });
                    return 0;
                }
                return prev + 1;
            });
        }, 500);
        return () => clearInterval(timerRef.current);
    }, [isPlaying, totalSteps]);

    const handlePlay = () => {
        if (step >= totalSteps) {
            setStep(0);
            setPipelineStage(0);
        }
        setIsPlaying(p => !p);
    };

    const handleStep = (dir) => {
        setIsPlaying(false);
        if (dir > 0 && step < totalSteps) {
            setStep(s => s + 1);
            setPipelineStage(4); // show completed
        } else if (dir < 0 && step > 0) {
            setStep(s => s - 1);
            setPipelineStage(0);
        }
    };

    const handleReset = () => {
        setIsPlaying(false);
        setStep(0);
        setPipelineStage(0);
    };

    // KV cache calculations
    const tokensInCache = EXAMPLE_PROMPT.length + step; // prompt + generated so far
    const kvPerToken = 2 * model.L * model.Hkv * model.dhead * 2; // bytes, FP16
    const currentKV = tokensInCache * kvPerToken;
    const maxTokensShown = EXAMPLE_PROMPT.length + totalSteps;

    return (
        <section className="chapter-section">
            <h3 className="section-title">Autoregressive Generation in Action</h3>
            <p className="section-desc">
                Now that you've seen the pipeline, let's watch it work. When {model.name} generates a response,
                each token must travel through every one of those {model.L} layers — and crucially, at each layer,
                the model saves intermediate results called <strong>Keys</strong> and <strong>Values</strong> into
                a structure called the <strong>KV cache</strong>. This cache lets the model remember what it has
                already computed, so it doesn't have to recompute everything from scratch for each new token.
            </p>
            <p className="section-desc">
                Press play or step through manually — watch the KV cache grow with every token generated.
            </p>

            <div className="token-stepper glass-card">
                {/* Controls */}
                <div className="stepper-controls">
                    <button className="stepper-btn" onClick={handleReset} title="Reset">↺</button>
                    <button className="stepper-btn" onClick={() => handleStep(-1)} disabled={step === 0} title="Previous">◀</button>
                    <button className={`stepper-btn play`} onClick={handlePlay} title={isPlaying ? 'Pause' : 'Play'}>
                        {isPlaying ? '⏸' : '▶'}
                    </button>
                    <button className="stepper-btn" onClick={() => handleStep(1)} disabled={step >= totalSteps} title="Next">▶</button>
                    <span className="stepper-progress">
                        Token {step}/{totalSteps}
                    </span>
                </div>

                {/* Prompt + generated tokens */}
                <div className="stepper-prompt">
                    {EXAMPLE_PROMPT.map((tok, i) => (
                        <span key={`p-${i}`} className="stepper-token input">{tok}</span>
                    ))}
                    {EXAMPLE_RESPONSE.slice(0, step).map((tok, i) => (
                        <span
                            key={`g-${i}`}
                            className={`stepper-token generated ${i === step - 1 ? 'current' : ''}`}
                        >
                            {tok}
                        </span>
                    ))}
                    {step < totalSteps && <span className="stepper-cursor" />}
                </div>

                {/* Pipeline visualization for current token */}
                {step > 0 && step <= totalSteps && (
                    <div className="stepper-pipeline">
                        {stages.map((stage, i) => (
                            <div key={stage.name} style={{ display: 'flex', alignItems: 'center' }}>
                                {i > 0 && <span className={`pipeline-arrow ${pipelineStage >= i ? 'active' : ''}`}>→</span>}
                                <div className={`pipeline-stage ${pipelineStage === i ? 'active' : pipelineStage > i ? 'done' : ''}`}>
                                    <span className="pipeline-stage-name">{stage.name}</span>
                                    <span className="pipeline-stage-val">{stage.val}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                )}

                {/* KV Cache growth */}
                <div className="stepper-kv">
                    <span className="stepper-kv-label">KV Cache:</span>
                    <div className="stepper-kv-blocks">
                        {Array.from({ length: maxTokensShown }, (_, i) => (
                            <div
                                key={i}
                                className={`kv-block ${i < tokensInCache ? (i === tokensInCache - 1 && step > 0 ? 'current' : 'filled') : 'empty'}`}
                                title={i < EXAMPLE_PROMPT.length ? `"${EXAMPLE_PROMPT[i]}"` : i < tokensInCache ? `"${EXAMPLE_RESPONSE[i - EXAMPLE_PROMPT.length]}"` : 'empty'}
                            />
                        ))}
                    </div>
                    <span className="stepper-kv-size">{formatBytes(currentKV)}</span>
                </div>

                <div className="stepper-layers">
                    Each token stores K and V vectors across all <strong>{model.L} layers</strong> ×{' '}
                    <strong>{model.Hkv} KV heads</strong> × <strong>{model.dhead} dims</strong> ={' '}
                    <span className="mono" style={{ color: 'var(--text-accent)' }}>{formatBytes(kvPerToken)}/token</span>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 3. PREFILL vs DECODE ANIMATION
// Side-by-side: parallel processing vs serial, GPU utilization
// ============================================================

function PrefillDecode({ model, gpu }) {
    const [activePhase, setActivePhase] = useState('both'); // 'prefill', 'decode', 'both'
    const [decodeStep, setDecodeStep] = useState(0);

    // Animate decode step
    useEffect(() => {
        if (activePhase !== 'decode' && activePhase !== 'both') return;
        const timer = setInterval(() => {
            setDecodeStep(s => (s + 1) % 7);
        }, 800);
        return () => clearInterval(timer);
    }, [activePhase]);

    const promptTokens = 7;
    const genTokens = 7;

    return (
        <section className="chapter-section">
            <h3 className="section-title">Why Is Generation So Slow? Prefill vs Decode</h3>
            <p className="section-desc">
                You might wonder: GPUs are powerful parallel processors — why does token generation feel slow?
                The answer is that inference actually has <strong>two very different phases</strong>.
                During the <strong>prefill phase</strong>, all your prompt tokens are processed at once in parallel —
                this is a large matrix-matrix multiplication that keeps the GPU saturated.
                But during the <strong>decode phase</strong>, the model generates one token at a time. Each token
                requires reading the full model weights ({formatBytes(modelWeightSize(model, 2))} for {model.name})
                from memory — but only does a tiny amount of compute per token. The GPU spends most of its time
                <em>waiting for data to arrive</em> from memory, not actually computing.
            </p>
            <p className="section-desc">
                This is the fundamental insight: <strong>LLM inference is memory-bound, not compute-bound.</strong>
            </p>

            <div className="prefill-decode glass-card">
                <div className="pd-toggle">
                    {['prefill', 'decode', 'both'].map(phase => (
                        <button
                            key={phase}
                            className={`pd-toggle-btn ${activePhase === phase ? 'active' : ''}`}
                            onClick={() => setActivePhase(phase)}
                        >
                            {phase === 'both' ? 'Both Phases' : phase.charAt(0).toUpperCase() + phase.slice(1) + ' Phase'}
                        </button>
                    ))}
                </div>

                <div className="pd-comparison">
                    {/* Prefill */}
                    {(activePhase === 'prefill' || activePhase === 'both') && (
                        <div className={`pd-phase ${activePhase === 'prefill' ? 'highlight' : ''}`}>
                            <div className="pd-phase-title">Prefill Phase</div>
                            <div className="pd-phase-desc">
                                All {promptTokens} prompt tokens processed <strong>in parallel</strong>.
                                Matrix-matrix multiply — GPU fully utilized.
                            </div>

                            <div className="pd-tokens">
                                {EXAMPLE_PROMPT.map((tok, i) => (
                                    <div key={i} className="pd-token parallel" style={{ animationDelay: `${i * 50}ms` }} title={tok}>
                                        {tok.substring(0, 3)}
                                    </div>
                                ))}
                            </div>

                            <div className="pd-gpu-bar">
                                <div className="pd-gpu-label">
                                    <span>GPU Compute Utilization</span>
                                    <span className="mono" style={{ color: 'var(--accent-primary)' }}>~92%</span>
                                </div>
                                <div className="pd-gpu-track">
                                    <div className="pd-gpu-fill prefill-fill" />
                                </div>
                            </div>

                            <div className="pd-bandwidth">
                                <span>Data flow:</span>
                                <div className="pd-pipe thick" />
                                <span className="pd-bottleneck-tag compute">Compute-bound</span>
                            </div>
                        </div>
                    )}

                    {/* Decode */}
                    {(activePhase === 'decode' || activePhase === 'both') && (
                        <div className={`pd-phase ${activePhase === 'decode' ? 'highlight' : ''}`}>
                            <div className="pd-phase-title">Decode Phase</div>
                            <div className="pd-phase-desc">
                                <strong>One token at a time</strong>. Each step: read {formatBytes(modelWeightSize(model, 2))} of
                                weights from memory for a single token. Matrix-vector multiply — GPU mostly idle, waiting for memory.
                            </div>

                            <div className="pd-tokens">
                                {EXAMPLE_RESPONSE.map((tok, i) => (
                                    <div
                                        key={i}
                                        className={`pd-token serial ${i === decodeStep ? 'active' : i > decodeStep ? 'waiting' : ''}`}
                                        title={tok}
                                    >
                                        {tok.substring(0, 3)}
                                    </div>
                                ))}
                            </div>

                            <div className="pd-gpu-bar">
                                <div className="pd-gpu-label">
                                    <span>GPU Compute Utilization</span>
                                    <span className="mono" style={{ color: 'var(--accent-warm)' }}>~15%</span>
                                </div>
                                <div className="pd-gpu-track">
                                    <div className={`pd-gpu-fill decode-fill ${decodeStep % 2 === 0 ? 'active-decode' : ''}`} />
                                </div>
                            </div>

                            <div className="pd-bandwidth">
                                <span>Data flow:</span>
                                <div className="pd-pipe thin" />
                                <span className="pd-bottleneck-tag memory">Memory-bound ⚠️</span>
                            </div>
                        </div>
                    )}
                </div>

                {/* Key insight */}
                <div style={{ marginTop: 'var(--space-md)', textAlign: 'center' }}>
                    <p style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                        <strong style={{ color: 'var(--accent-warm)' }}>The bottleneck:</strong>{' '}
                        Generating each token requires reading <strong>all {formatBytes(modelWeightSize(model, 2))}</strong> of
                        {model.name}'s weights from {gpu.name}'s memory ({gpu.bandwidth_gbs} GB/s bandwidth).
                        The GPU finishes computing long before the data arrives.
                        <br />
                        <strong>Minimum time per token</strong> ≈ {(modelWeightSize(model, 2) / (gpu.bandwidth_gbs * 1e9) * 1000).toFixed(2)} ms
                        ({(gpu.bandwidth_gbs * 1e9 / modelWeightSize(model, 2)).toFixed(0)} tokens/sec theoretical max).
                    </p>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 4. MEMORY BUDGET STACK
// Single stacked bar: weights (fixed) + KV cache (grows with slider)
// against GPU budget. Shows overflow when it exceeds budget.
// ============================================================

function MemoryStack({ model, gpu }) {
    const [tokens, setTokens] = useState(2048);

    const weightBytes = modelWeightSize(model, 2);
    const kvBytes = kvCacheSize(model, tokens);
    const totalBytes = weightBytes + kvBytes;
    const budgetBytes = gpu.budget_mb * 1024 * 1024;

    const maxBarBytes = Math.max(totalBytes, budgetBytes) * 1.1; // 10% padding
    const weightPct = (weightBytes / maxBarBytes) * 100;
    const kvPct = (kvBytes / maxBarBytes) * 100;
    const budgetPct = (budgetBytes / maxBarBytes) * 100;

    const doesFit = totalBytes <= budgetBytes;
    const utilizationPct = ((totalBytes / budgetBytes) * 100).toFixed(1);

    // Max tokens before overflow
    const kvBudget = budgetBytes - weightBytes;
    const kvPerToken = 2 * model.L * model.Hkv * model.dhead * 2; // FP16
    const maxTokens = Math.max(0, Math.floor(kvBudget / kvPerToken));

    return (
        <section className="chapter-section">
            <h3 className="section-title">The Memory Budget — Two Competing Tenants</h3>
            <p className="section-desc">
                So if inference is memory-bound, how much memory do we actually need? Two things compete for
                space in your device's memory: the <strong>model weights</strong> (fixed — {formatBytes(modelWeightSize(model, 2))}
                {' '}for {model.name} at FP16) and the <strong>KV cache</strong> (dynamic — it grows linearly with
                every token generated). On a small device like a Raspberry Pi with 1.5 GB, even a small model's
                KV cache can push total usage over the limit.
            </p>
            <p className="section-desc">
                Drag the slider below to see how the KV cache grows and when it causes a memory overflow
                on {gpu.name}.
            </p>

            <div className="mem-stack glass-card">
                <div className="mem-stack-title">
                    {gpu.name} — {gpu.budget_mb >= 1024 ? (gpu.budget_mb / 1024) + ' GB' : gpu.budget_mb + ' MB'} total
                </div>

                {/* Stacked bar */}
                <div className="mem-stack-bar">
                    {/* Weights segment */}
                    <div
                        className="mem-stack-segment weights"
                        style={{ width: `${weightPct}%` }}
                    >
                        <span className="mem-seg-label">Weights {formatBytes(weightBytes)}</span>
                    </div>

                    {/* KV cache segment */}
                    <div
                        className={`mem-stack-segment kv-cache ${!doesFit ? 'overflow-zone' : ''}`}
                        style={{ left: `${weightPct}%`, width: `${kvPct}%` }}
                    >
                        {kvPct > 8 && (
                            <span className="mem-seg-label">KV {formatBytes(kvBytes)}</span>
                        )}
                    </div>

                    {/* Budget line */}
                    <div className="mem-stack-budget" style={{ left: `${budgetPct}%` }}>
                        {gpu.budget_mb >= 1024 ? `${(gpu.budget_mb / 1024)} GB` : `${gpu.budget_mb} MB`} limit
                    </div>
                </div>

                {/* Token slider */}
                <div className="mem-slider-row">
                    <span className="mem-slider-label">Tokens:</span>
                    <input
                        type="range"
                        className="mem-slider"
                        min={1}
                        max={8192}
                        value={tokens}
                        onChange={e => setTokens(Number(e.target.value))}
                    />
                    <span className="mem-slider-value">{tokens.toLocaleString()}</span>
                </div>

                {/* Breakdown */}
                <div className="mem-breakdown">
                    <div className="mem-breakdown-item">
                        <span className="mem-swatch weights-swatch" />
                        <span className="mem-breakdown-label">Weights:</span>
                        <span className="mem-breakdown-value">{formatBytes(weightBytes)}</span>
                    </div>
                    <div className="mem-breakdown-item">
                        <span className="mem-swatch kv-swatch" />
                        <span className="mem-breakdown-label">KV Cache:</span>
                        <span className="mem-breakdown-value">{formatBytes(kvBytes)}</span>
                    </div>
                    <div className="mem-breakdown-item">
                        <span className="mem-swatch free-swatch" />
                        <span className="mem-breakdown-label">Free:</span>
                        <span className="mem-breakdown-value">
                            {doesFit ? formatBytes(budgetBytes - totalBytes) : '0 — overflow!'}
                        </span>
                    </div>
                </div>

                {/* Verdict */}
                <div className={`mem-verdict ${doesFit ? 'fits' : 'fails'}`}>
                    {doesFit
                        ? `✅ Fits — ${utilizationPct}% used (max ${maxTokens.toLocaleString()} tokens before overflow)`
                        : `❌ Overflow! ${formatBytes(totalBytes)} required, only ${formatBytes(budgetBytes)} available. Reduce to ≤${maxTokens.toLocaleString()} tokens.`
                    }
                </div>
            </div>
        </section>
    );
}


// ============================================================
// PROLOGUE — Main Component
// ============================================================

export default function Prologue({ selectedModel, selectedGPU }) {
    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    return (
        <div className="chapter prologue animate-in">
            <div className="chapter-header">
                <h2 className="chapter-title">What happens when you press Enter?</h2>
                <p className="chapter-hook">
                    You type a prompt, hit Enter, and tokens appear one by one.
                    Behind that simple interface lies a memory bottleneck that determines
                    how fast — and whether — a model can run on your device.
                </p>
            </div>

            <TransformerFlow model={model} />
            <TokenStepper model={model} />
            <PrefillDecode model={model} gpu={gpu} />
            <MemoryStack model={model} gpu={gpu} />

            {/* Hook to Chapter 1 */}
            <section className="chapter-section">
                <div className="chapter-next-hook glass-card">
                    <p>
                        The weights are fixed — {formatBytes(modelWeightSize(model, 2))} for {model.name}.
                        But the <strong>KV cache</strong> grows with every token. What is this KV cache?
                        Why does every token leave a permanent trace in memory?
                    </p>
                    <p className="chapter-next-question">
                        → What is attention, and what does it store?
                    </p>
                </div>
            </section>
        </div>
    );
}

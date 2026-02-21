import { useState, useEffect, useRef, useCallback } from 'react';
import { MODELS, GPUS, kvCacheSize, modelWeightSize, formatBytes } from '../data/modelConfig';
import SpeedControl from '../components/SpeedControl';
import AnimationCommentary from '../components/AnimationCommentary';
import SmallModelNote from '../components/SmallModelNote';
import './Prologue.css';

// ============================================================
// 1. TRANSFORMER ARCHITECTURE FLOW — AUTO-PLAYING
// Animated block diagram with data dots flowing through pipeline.
// Dot travels: Input → Embed → Layer 1..N → LM Head → Output
// Auto-loops, with running commentary for each stage.
// ============================================================

const PIPELINE_STAGES = [
    { node: 'input', label: 'Token Input', commentary: (m) => `A token from the prompt enters the pipeline. It starts as a simple token ID — just a number representing a word or subword.` },
    { node: 'embed', label: 'Embedding', commentary: (m) => `The token ID is looked up in an embedding table to produce a ${m.dmodel}-dimensional vector. This vector is the token's initial representation — a list of ${m.dmodel} numbers.` },
    { node: 'layer-0', label: 'Layer 1', commentary: (m) => `The embedding enters Layer 1. Inside: Self-Attention (${m.Hq} Q heads, ${m.Hkv} KV heads, ${m.attnType}) computes which tokens to attend to, then a Feed-Forward Network transforms the result.` },
    { node: 'layer-1', label: 'Layer 2', commentary: (m) => `Layer 2 receives the output from Layer 1 and applies the same attention + FFN pattern, but with different learned weights. Each layer refines the token's representation.` },
    { node: 'layers-rest', label: `Layers 3–N`, commentary: (m) => `The vector continues through the remaining ${m.L - 2} layers. Each layer has its own attention weights and FFN weights — totaling ${formatBytes(modelWeightSize(m, 2))} for ${m.name}.` },
    { node: 'lm-head', label: 'LM Head', commentary: (m) => `The final layer's output (a ${m.dmodel}-dim vector) is multiplied by the LM Head to produce logits — one score for every word in the vocabulary.` },
    { node: 'output', label: 'Next Token', commentary: (m) => `Softmax converts logits into probabilities. The model samples or picks the highest-probability token. This token then feeds back as input — that's autoregressive generation.` },
];

function TransformerFlow({ model }) {
    const [activeStage, setActiveStage] = useState(-1); // -1 = idle / waiting
    const [isPlaying, setIsPlaying] = useState(true);
    const [speed, setSpeed] = useState(1);
    const [commentaryLog, setCommentaryLog] = useState([]);
    const timerRef = useRef(null);
    const logEndRef = useRef(null);

    // Accumulate commentary as stages progress
    useEffect(() => {
        if (activeStage >= 0 && activeStage < PIPELINE_STAGES.length) {
            const entry = {
                stage: activeStage,
                label: PIPELINE_STAGES[activeStage].label,
                text: PIPELINE_STAGES[activeStage].commentary(model),
            };
            setCommentaryLog(prev => {
                // On loop restart (stage 0 after stage 6), clear log
                if (activeStage === 0 && prev.length > 0 && prev[prev.length - 1].stage === PIPELINE_STAGES.length - 1) {
                    return [entry];
                }
                // Avoid duplicates if stage didn't change
                if (prev.length > 0 && prev[prev.length - 1].stage === activeStage) return prev;
                return [...prev, entry];
            });
        }
    }, [activeStage, model]);

    // Auto-scroll to latest entry
    useEffect(() => {
        if (logEndRef.current) {
            logEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
    }, [commentaryLog]);

    // Auto-play through stages
    useEffect(() => {
        if (!isPlaying) { clearInterval(timerRef.current); return; }
        const interval = 2000 / speed;
        timerRef.current = setInterval(() => {
            setActiveStage(prev => {
                if (prev >= PIPELINE_STAGES.length - 1) return 0; // loop
                return prev + 1;
            });
        }, interval);
        return () => clearInterval(timerRef.current);
    }, [isPlaying, speed]);

    // Start playing on mount
    useEffect(() => { setActiveStage(0); }, []);

    const handlePlayPause = () => setIsPlaying(p => !p);

    const layerCount = model.L;

    return (
        <section className="chapter-section">
            <h3 className="section-title">The Transformer Pipeline</h3>
            <p className="section-desc">
                Every large language model — from the smallest 135M-parameter model to GPT-4 — is built on the same
                architecture: the <strong>Transformer</strong>. It's a stack of identical layers, each containing a
                self-attention mechanism and a feed-forward network. Your prompt enters on the left as a sequence of
                tokens, flows through <strong>{layerCount} transformer layers</strong>, and a single next token
                exits on the right. That token then feeds back as input — this is called
                <strong> autoregressive generation</strong>.
            </p>

            <div className="arch-flow glass-card">
                <div className="arch-flow-header">
                    <span className="arch-flow-title">{model.name} — {layerCount} layers, {model.dmodel}-dim</span>
                    <div className="arch-flow-controls">
                        <button className="stepper-btn play" onClick={handlePlayPause} title={isPlaying ? 'Pause' : 'Play'}>
                            {isPlaying ? '⏸' : '▶'}
                        </button>
                        <SpeedControl speed={speed} onSpeedChange={setSpeed} />
                    </div>
                </div>

                <div className="arch-pipeline">
                    {/* A flowing dot that follows activeStage */}
                    <div className="arch-flow-track">
                        <div className="arch-data-dot" />
                        <div className="arch-data-dot" style={{ animationDelay: '-3s' }} />
                        <div className="arch-data-dot" style={{ animationDelay: '-6s' }} />
                    </div>

                    {/* Input */}
                    <div className={`arch-node embed ${activeStage === 0 ? 'stage-active' : activeStage > 0 ? 'stage-done' : ''}`}>
                        <span className="arch-node-label">Input</span>
                        <span className="arch-node-dim">token</span>
                    </div>

                    <div className={`arch-arrow ${activeStage >= 1 ? 'active' : ''}`} />

                    {/* Embedding */}
                    <div className={`arch-node embed ${activeStage === 1 ? 'stage-active' : activeStage > 1 ? 'stage-done' : ''}`}>
                        <span className="arch-node-label">Embed</span>
                        <span className="arch-node-dim">{model.dmodel}-d</span>
                    </div>

                    <div className={`arch-arrow ${activeStage >= 2 ? 'active' : ''}`} />

                    {/* Layer group */}
                    <div className="arch-layers-group">
                        {/* Layer 1 */}
                        <div className={`arch-node layer ${activeStage === 2 ? 'stage-active' : activeStage > 2 ? 'stage-done' : ''}`}>
                            <span className="arch-node-label">Layer 1</span>
                            <span className="arch-node-dim">Attn+FFN</span>
                        </div>

                        <div className={`arch-arrow ${activeStage >= 3 ? 'active' : ''}`} />

                        {/* Layer 2 */}
                        <div className={`arch-node layer ${activeStage === 3 ? 'stage-active' : activeStage > 3 ? 'stage-done' : ''}`}>
                            <span className="arch-node-label">Layer 2</span>
                            <span className="arch-node-dim">Attn+FFN</span>
                        </div>

                        <div className={`arch-arrow ${activeStage >= 4 ? 'active' : ''}`} />

                        {/* Layers 3..N */}
                        <div className={`arch-node layer layers-rest ${activeStage === 4 ? 'stage-active' : activeStage > 4 ? 'stage-done' : ''}`}>
                            <span className="arch-node-label">Layers 3–{layerCount}</span>
                            <span className="arch-node-dim">×{layerCount - 2}</span>
                        </div>
                    </div>

                    <div className={`arch-arrow ${activeStage >= 5 ? 'active' : ''}`} />

                    {/* LM Head */}
                    <div className={`arch-node lm-head ${activeStage === 5 ? 'stage-active' : activeStage > 5 ? 'stage-done' : ''}`}>
                        <span className="arch-node-label">LM Head</span>
                        <span className="arch-node-dim">→ vocab</span>
                    </div>

                    <div className={`arch-arrow ${activeStage >= 6 ? 'active' : ''}`} />

                    {/* Output */}
                    <div className={`arch-node token-out ${activeStage === 6 ? 'stage-active' : ''}`}>
                        <span className="arch-node-label">Next Token</span>
                        <span className="arch-node-dim">↺ feeds back</span>
                    </div>
                </div>

                <div className="arch-loop">
                    <span style={{ fontSize: '10px', color: 'var(--text-muted)' }}>
                        ↺ Each generated token feeds back as input — this loop repeats for every token in the response
                    </span>
                </div>

                {/* Running commentary — stacking log */}
                <div className="pipeline-commentary-log">
                    {commentaryLog.length === 0 && (
                        <div className="pipeline-commentary-entry idle">
                            <span className="commentary-icon">🔍</span>
                            <span className="commentary-text">Press play to watch a token travel through the transformer pipeline.</span>
                        </div>
                    )}
                    {commentaryLog.map((entry, i) => (
                        <div
                            key={`${entry.stage}-${i}`}
                            className={`pipeline-commentary-entry ${i === commentaryLog.length - 1 ? 'latest' : 'past'}`}
                        >
                            <span className="pipeline-entry-badge">{entry.label}</span>
                            <span className="commentary-text">{entry.text}</span>
                        </div>
                    ))}
                    <div ref={logEndRef} />
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 2. TOKEN GENERATION STEPPER
// Step through token generation with auto-play, speed control,
// and running commentary per token.
// ============================================================

const EXAMPLE_PROMPT = ["What", "is", "the", "capital", "of", "France", "?"];
const EXAMPLE_RESPONSE = ["The", "capital", "of", "France", "is", "Paris", "."];

function TokenStepper({ model }) {
    const [step, setStep] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    const [speed, setSpeed] = useState(1);
    const [pipelineStage, setPipelineStage] = useState(0);
    const timerRef = useRef(null);
    const totalSteps = EXAMPLE_RESPONSE.length;

    const stages = [
        { name: 'Embed', val: `${model.dmodel}-d` },
        { name: `${model.L} Layers`, val: 'Attn+FFN' },
        { name: 'LM Head', val: 'logits' },
        { name: 'Softmax', val: 'prob' },
        { name: 'Token', val: step > 0 ? EXAMPLE_RESPONSE[step - 1] : '...' },
    ];

    // Commentary per token step
    const getCommentary = () => {
        if (step === 0) return 'Press play to start generating tokens. The prompt "What is the capital of France?" will be processed, then the model generates a response token-by-token.';
        const tok = EXAMPLE_RESPONSE[step - 1];
        const stageNames = ['embedding the token', `passing through ${model.L} layers`, 'producing logits via LM Head', 'computing softmax probabilities', `outputting "${tok}"`];
        const stgIdx = Math.min(pipelineStage, 4);
        return `Token ${step}/${totalSteps}: "${tok}" — currently ${stageNames[stgIdx]}. KV cache now stores K,V for ${EXAMPLE_PROMPT.length + step} tokens across all ${model.L} layers.`;
    };

    // Auto-play logic
    useEffect(() => {
        if (!isPlaying) return;
        const interval = 500 / speed;
        timerRef.current = setInterval(() => {
            setPipelineStage(prev => {
                if (prev >= 4) {
                    setStep(s => {
                        if (s >= totalSteps) { setIsPlaying(false); return s; }
                        return s + 1;
                    });
                    return 0;
                }
                return prev + 1;
            });
        }, interval);
        return () => clearInterval(timerRef.current);
    }, [isPlaying, totalSteps, speed]);

    const handlePlay = () => {
        if (step >= totalSteps) { setStep(0); setPipelineStage(0); }
        setIsPlaying(p => !p);
    };

    const handleStep = (dir) => {
        setIsPlaying(false);
        if (dir > 0 && step < totalSteps) { setStep(s => s + 1); setPipelineStage(4); }
        else if (dir < 0 && step > 0) { setStep(s => s - 1); setPipelineStage(0); }
    };

    const handleReset = () => { setIsPlaying(false); setStep(0); setPipelineStage(0); };

    // KV cache calculations
    const tokensInCache = EXAMPLE_PROMPT.length + step;
    const kvPerToken = 2 * model.L * model.Hkv * model.dhead * 2;
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

            <div className="token-stepper glass-card">
                {/* Controls */}
                <div className="stepper-controls">
                    <button className="stepper-btn" onClick={handleReset} title="Reset">↺</button>
                    <button className="stepper-btn" onClick={() => handleStep(-1)} disabled={step === 0} title="Previous">◀</button>
                    <button className="stepper-btn play" onClick={handlePlay} title={isPlaying ? 'Pause' : 'Play'}>
                        {isPlaying ? '⏸' : '▶'}
                    </button>
                    <button className="stepper-btn" onClick={() => handleStep(1)} disabled={step >= totalSteps} title="Next">▶</button>
                    <SpeedControl speed={speed} onSpeedChange={setSpeed} />
                    <span className="stepper-progress">Token {step}/{totalSteps}</span>
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

                {/* Commentary */}
                <AnimationCommentary text={getCommentary()} icon="📝" />
            </div>
        </section>
    );
}


// ============================================================
// 3. PREFILL vs DECODE ANIMATION
// Side-by-side: parallel processing vs serial, GPU utilization
// ============================================================

function PrefillDecode({ model, gpu }) {
    const [activePhase, setActivePhase] = useState('both');
    const [decodeStep, setDecodeStep] = useState(0);

    useEffect(() => {
        if (activePhase !== 'decode' && activePhase !== 'both') return;
        const timer = setInterval(() => { setDecodeStep(s => (s + 1) % 7); }, 800);
        return () => clearInterval(timer);
    }, [activePhase]);

    const promptTokens = 7;

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
                <em> waiting for data to arrive</em> from memory, not actually computing.
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

    const maxBarBytes = Math.max(totalBytes, budgetBytes) * 1.1;
    const weightPct = (weightBytes / maxBarBytes) * 100;
    const kvPct = (kvBytes / maxBarBytes) * 100;
    const budgetPct = (budgetBytes / maxBarBytes) * 100;

    const doesFit = totalBytes <= budgetBytes;
    const utilizationPct = ((totalBytes / budgetBytes) * 100).toFixed(1);

    // Max tokens before overflow
    const kvBudget = budgetBytes - weightBytes;
    const kvPerToken = 2 * model.L * model.Hkv * model.dhead * 2;
    const maxTokens = Math.max(0, Math.floor(kvBudget / kvPerToken));

    // Note when cache fits easily
    const fitsEasily = doesFit && parseFloat(utilizationPct) < 30;

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

            <SmallModelNote kvBytes={kvCacheSize(model, 2048)} />

            <div className="mem-stack glass-card">
                <div className="mem-stack-title">
                    {gpu.name} — {gpu.budget_mb >= 1024 ? (gpu.budget_mb / 1024) + ' GB' : gpu.budget_mb + ' MB'} total
                </div>

                {/* Stacked bar */}
                <div className="mem-stack-bar">
                    <div className="mem-stack-segment weights" style={{ width: `${weightPct}%` }}>
                        <span className="mem-seg-label">Weights {formatBytes(weightBytes)}</span>
                    </div>
                    <div
                        className={`mem-stack-segment kv-cache ${!doesFit ? 'overflow-zone' : ''}`}
                        style={{ left: `${weightPct}%`, width: `${kvPct}%` }}
                    >
                        {kvPct > 8 && <span className="mem-seg-label">KV {formatBytes(kvBytes)}</span>}
                    </div>
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

                {/* Note when cache fits easily */}
                {fitsEasily && (
                    <div style={{ marginTop: 'var(--space-sm)', padding: 'var(--space-xs) var(--space-md)', background: 'rgba(255, 215, 59, 0.06)', borderRadius: 'var(--radius-sm)', fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', textAlign: 'center' }}>
                        💡 With this GPU, {model.name} fits comfortably. Try selecting a <strong>larger model</strong> (Llama-2-7B) or a <strong>smaller GPU</strong> (Raspberry Pi 4) to see memory pressure.
                    </div>
                )}
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
            {/* Prerequisite notice */}
            <div className="prerequisite-notice">
                <span className="prerequisite-icon">📚</span>
                <span className="prerequisite-text">
                    <strong>Prerequisite:</strong> This guide assumes familiarity with Transformer architecture.
                    New to Transformers? Read{' '}
                    <a href="https://jalammar.github.io/illustrated-transformer/" target="_blank" rel="noopener noreferrer" className="prerequisite-link">
                        The Illustrated Transformer — Jay Alammar
                    </a>
                    {' '}first.
                </span>
            </div>

            <div className="chapter-header">
                <h2 className="chapter-title">The Inference Engine — What Happens When You Press Enter?</h2>
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

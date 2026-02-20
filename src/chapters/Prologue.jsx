import { useState, useEffect, useRef, useCallback } from 'react';
import { MODELS, GPUS, kvCacheSize, modelWeightSize, formatBytes } from '../data/modelConfig';
import ExplanationPanel from '../components/ExplanationPanel';
import MemoryBar from '../components/MemoryBar';
import GoDeeper from '../components/GoDeeper';
import './Prologue.css';

// --- Token-by-Token Animation ---
const PROMPT_TOKENS = ['What', ' is', ' the', ' capital', ' of', ' France', '?'];
const RESPONSE_TOKENS = ['The', ' capital', ' of', ' France', ' is', ' Paris', '.', ' It', ' is', ' known', ' as', ' the', ' City', ' of', ' Light', '.'];

function TokenAnimation() {
    const [phase, setPhase] = useState('idle');       // idle, prompt, thinking, generating, done
    const [visiblePrompt, setVisiblePrompt] = useState([]);
    const [visibleResponse, setVisibleResponse] = useState([]);
    const [tokenCounter, setTokenCounter] = useState(0);
    const timerRef = useRef(null);

    const reset = useCallback(() => {
        setPhase('idle');
        setVisiblePrompt([]);
        setVisibleResponse([]);
        setTokenCounter(0);
        if (timerRef.current) clearTimeout(timerRef.current);
    }, []);

    const startAnimation = useCallback(() => {
        reset();
        setPhase('prompt');

        // Show prompt tokens all at once (simulating user typing)
        setTimeout(() => {
            setVisiblePrompt(PROMPT_TOKENS);
            setPhase('thinking');

            // Brief "thinking" pause
            setTimeout(() => {
                setPhase('generating');
                let i = 0;
                const genNext = () => {
                    if (i < RESPONSE_TOKENS.length) {
                        setVisibleResponse(prev => [...prev, RESPONSE_TOKENS[i]]);
                        setTokenCounter(i + 1);
                        i++;
                        timerRef.current = setTimeout(genNext, 150 + Math.random() * 100);
                    } else {
                        setPhase('done');
                    }
                };
                genNext();
            }, 800);
        }, 300);
    }, [reset]);

    useEffect(() => {
        startAnimation();
        return () => { if (timerRef.current) clearTimeout(timerRef.current); };
    }, [startAnimation]);

    return (
        <div className="token-animation glass-card">
            <div className="token-anim-header">
                <span className="token-anim-title">LLM Inference</span>
                <button className="token-anim-replay" onClick={startAnimation} title="Replay">↻</button>
            </div>

            <div className="token-anim-chat">
                {/* Prompt */}
                {visiblePrompt.length > 0 && (
                    <div className="token-msg user-msg">
                        <span className="token-msg-label">User</span>
                        <div className="token-msg-content">
                            {visiblePrompt.map((t, i) => (
                                <span key={i} className="token prompt-token">{t}</span>
                            ))}
                        </div>
                    </div>
                )}

                {/* Thinking indicator */}
                {phase === 'thinking' && (
                    <div className="token-msg ai-msg thinking">
                        <span className="token-msg-label">Model</span>
                        <div className="token-msg-content">
                            <span className="thinking-dots">
                                <span>●</span><span>●</span><span>●</span>
                            </span>
                        </div>
                    </div>
                )}

                {/* Response */}
                {visibleResponse.length > 0 && (
                    <div className="token-msg ai-msg">
                        <span className="token-msg-label">Model</span>
                        <div className="token-msg-content">
                            {visibleResponse.map((t, i) => (
                                <span key={i} className="token response-token" style={{ animationDelay: `${i * 0.05}s` }}>{t}</span>
                            ))}
                            {phase === 'generating' && <span className="cursor-blink">│</span>}
                        </div>
                    </div>
                )}
            </div>

            {/* Counter */}
            <div className="token-anim-footer mono">
                {phase === 'prompt' && <span>Processing prompt...</span>}
                {phase === 'thinking' && <span>Prefill: {PROMPT_TOKENS.length} tokens processed in parallel</span>}
                {phase === 'generating' && <span>Decode: generating token {tokenCounter} of {RESPONSE_TOKENS.length}... (one at a time)</span>}
                {phase === 'done' && <span>Done: {RESPONSE_TOKENS.length} tokens generated, each required a full model forward pass</span>}
            </div>
        </div>
    );
}

// --- Prefill vs Decode Visualization ---
function PrefillDecodeViz() {
    const [activePhase, setActivePhase] = useState('prefill');

    return (
        <div className="prefill-decode-viz">
            <div className="pd-toggle">
                <button
                    className={`pd-toggle-btn ${activePhase === 'prefill' ? 'active' : ''}`}
                    onClick={() => setActivePhase('prefill')}
                >
                    Prefill Phase
                </button>
                <button
                    className={`pd-toggle-btn ${activePhase === 'decode' ? 'active' : ''}`}
                    onClick={() => setActivePhase('decode')}
                >
                    Decode Phase
                </button>
            </div>

            <div className="pd-content glass-card">
                {activePhase === 'prefill' ? (
                    <div className="pd-phase animate-in">
                        <div className="pd-visual">
                            <div className="pd-tokens-row">
                                {PROMPT_TOKENS.map((t, i) => (
                                    <span key={i} className="pd-token prefill-token">{t}</span>
                                ))}
                            </div>
                            <div className="pd-arrow">↓ All at once</div>
                            <div className="pd-gpu-bar">
                                <div className="pd-gpu-fill prefill-fill" style={{ width: '95%' }}>
                                    <span>GPU 95% utilized</span>
                                </div>
                            </div>
                        </div>
                        <div className="pd-desc">
                            <h4>Prefill — Compute-Bound</h4>
                            <p>All prompt tokens are processed <strong>in parallel</strong>. The GPU's compute units are fully busy
                                performing matrix multiplications. This is fast because the hardware is doing what it was designed for.</p>
                            <p className="pd-bottleneck">Bottleneck: <span className="tag compute">Compute</span></p>
                        </div>
                    </div>
                ) : (
                    <div className="pd-phase animate-in">
                        <div className="pd-visual">
                            <div className="pd-tokens-row">
                                <span className="pd-token decode-token active">▸ Token</span>
                                <span className="pd-token decode-token dim">next</span>
                                <span className="pd-token decode-token dim">next</span>
                                <span className="pd-token decode-token dim">…</span>
                            </div>
                            <div className="pd-arrow">↓ One at a time</div>
                            <div className="pd-gpu-bar">
                                <div className="pd-gpu-fill decode-fill" style={{ width: '15%' }}>
                                    <span>15%</span>
                                </div>
                                <span className="pd-gpu-idle">GPU idle — waiting for memory</span>
                            </div>
                        </div>
                        <div className="pd-desc">
                            <h4>Decode — Memory-Bound</h4>
                            <p>Each token is generated <strong>one at a time</strong>. For each token, the entire model's weights
                                must be read from memory. The GPU finishes the math quickly, then waits for the next chunk of data.</p>
                            <p className="pd-bottleneck">Bottleneck: <span className="tag memory">Memory Bandwidth</span></p>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

// --- Memory Bottleneck Diagram ---
function MemoryBottleneck({ model, gpu }) {
    const weightBytes = modelWeightSize(model, 2);
    const readTimeMs = (weightBytes / 1e9) / gpu.bandwidth_gbs * 1000;
    // Approximate compute time ~ weights / (TFLOPS * 1e12) * 1000
    // Rough estimate: compute is ~10-50x faster than memory read
    const computeTimeMs = readTimeMs * 0.08;

    const maxTime = Math.max(readTimeMs, computeTimeMs);

    return (
        <div className="bottleneck-chart glass-card">
            <h4 className="bottleneck-title">Time Per Token — {model.name} on {gpu.name}</h4>

            <div className="bottleneck-bars">
                <div className="bottleneck-row">
                    <span className="bottleneck-label">Memory Read</span>
                    <div className="bottleneck-bar-track">
                        <div
                            className="bottleneck-bar-fill memory-bar-color"
                            style={{ width: `${(readTimeMs / maxTime) * 100}%` }}
                        />
                    </div>
                    <span className="bottleneck-value mono">{readTimeMs.toFixed(1)} ms</span>
                </div>

                <div className="bottleneck-row">
                    <span className="bottleneck-label">Compute (MatMul)</span>
                    <div className="bottleneck-bar-track">
                        <div
                            className="bottleneck-bar-fill compute-bar-color"
                            style={{ width: `${(computeTimeMs / maxTime) * 100}%` }}
                        />
                    </div>
                    <span className="bottleneck-value mono">{computeTimeMs.toFixed(2)} ms</span>
                </div>
            </div>

            <div className="bottleneck-callout">
                <span className="bottleneck-ratio mono">{(readTimeMs / computeTimeMs).toFixed(0)}×</span>
                <span>Memory read takes <strong>{(readTimeMs / computeTimeMs).toFixed(0)}× longer</strong> than the actual computation.
                    The GPU spends most of its time waiting for data.</span>
            </div>
        </div>
    );
}

// --- Main Prologue Component ---
export default function Prologue({ selectedModel, selectedGPU }) {
    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    const weightsFp16 = modelWeightSize(model, 2);
    const kvFp16 = kvCacheSize(model, 2048, 1, 2);
    const budgetBytes = gpu.budget_mb * 1024 * 1024;

    return (
        <div className="chapter prologue animate-in">
            <div className="chapter-header">
                <h2 className="chapter-title">What happens when you press Enter?</h2>
                <p className="chapter-hook">
                    You type a prompt and hit Enter. Words start appearing, one at a time.
                    But what is actually happening inside the model?
                </p>
            </div>

            {/* Section 1: Token-by-Token Generation */}
            <section className="chapter-section">
                <ExplanationPanel title="Tokens come out one at a time" variant="what">
                    <p>
                        LLMs don't write entire paragraphs at once. They generate <strong>one token at a time</strong> — each
                        token requires running the entire model to predict just the next word. A 100-word response means
                        running the model roughly 130 times, each time producing a single token.
                    </p>
                </ExplanationPanel>
                <TokenAnimation />
            </section>

            {/* Section 2: Prefill vs Decode */}
            <section className="chapter-section">
                <ExplanationPanel title="Two phases: Prefill and Decode" variant="what">
                    <p>
                        Inference has two distinct phases. <strong>Prefill</strong> reads your entire prompt in parallel —
                        the GPU is busy, compute is the bottleneck. <strong>Decode</strong> generates tokens one by one —
                        for each token, the entire model must be read from memory, and the GPU sits idle most of the time.
                    </p>
                </ExplanationPanel>
                <PrefillDecodeViz />
            </section>

            {/* Section 3: The Memory Bottleneck */}
            <section className="chapter-section">
                <ExplanationPanel title="Memory is the bottleneck, not compute" variant="why">
                    <p>
                        During decode, every token generation reads <strong>all {formatBytes(weightsFp16)}</strong> of {model.name}'s
                        weights from {gpu.name}'s memory. At {gpu.bandwidth_gbs} GB/s, just reading the weights
                        takes <code>{((weightsFp16 / 1e9) / gpu.bandwidth_gbs * 1000).toFixed(1)} ms</code> per token.
                        The actual matrix multiplication finishes in a fraction of that time.
                    </p>
                    <p>
                        This is why inference speed is measured in <strong>tokens per second</strong>, not FLOPS.
                        A faster GPU doesn't help much if the memory bus can't feed it data fast enough.
                    </p>
                </ExplanationPanel>

                <MemoryBottleneck model={model} gpu={gpu} />

                <div className="memory-bars-group glass-card">
                    <h4 className="memory-bars-title">Memory Budget — {model.name} on {gpu.name}</h4>
                    <MemoryBar
                        value={weightsFp16}
                        max={budgetBytes}
                        label="Model Weights (FP16)"
                        sublabel="Fixed cost — doesn't change during generation"
                        color="accent"
                    />
                    <MemoryBar
                        value={kvFp16}
                        max={budgetBytes}
                        label="KV Cache @2,048 tokens"
                        sublabel="Variable cost — grows with every generated token"
                        color="warm"
                    />
                    <MemoryBar
                        value={weightsFp16 + kvFp16}
                        max={budgetBytes}
                        label="Total"
                        sublabel={weightsFp16 + kvFp16 > budgetBytes
                            ? `Exceeds ${gpu.name} capacity by ${formatBytes(weightsFp16 + kvFp16 - budgetBytes)}!`
                            : `Uses ${((weightsFp16 + kvFp16) / budgetBytes * 100).toFixed(0)}% of ${gpu.name}`
                        }
                        color="success"
                    />
                </div>
            </section>

            {/* Go Deeper: Arithmetic Intensity */}
            <GoDeeper title="Go Deeper — Arithmetic Intensity">
                <ExplanationPanel title="Why is decode memory-bound?" variant="math">
                    <p>
                        <strong>Arithmetic intensity</strong> is the ratio of compute operations to memory operations.
                        When this ratio is low, the processor finishes its math before the next batch of data arrives.
                    </p>
                    <p>
                        During decode, the model processes a <strong>single token</strong> per step. For a matrix multiply
                        of shape <code>[1 × d]</code> × <code>[d × d]</code>, you do <code>2d²</code> FLOPs but read <code>d² × 2 bytes</code> of
                        weights. The arithmetic intensity is:
                    </p>
                    <p>
                        <code>AI = 2d² / (d² × 2) = 1 FLOP/byte</code>
                    </p>
                    <p>
                        For {gpu.name} with {gpu.bandwidth_gbs} GB/s bandwidth, the memory can deliver{' '}
                        <code>{gpu.bandwidth_gbs} G values/s</code> (at FP16). The compute units can process
                        orders of magnitude more. <strong>Memory is the ceiling.</strong>
                    </p>
                </ExplanationPanel>
            </GoDeeper>

            {/* Hook to Chapter 1 */}
            <section className="chapter-section">
                <div className="chapter-next-hook glass-card">
                    <p>
                        Model weights are fixed — you chose them when you loaded the model. But there's a second memory cost
                        that <em>grows</em> with every token: the <strong>KV Cache</strong>. It stores the attention keys
                        and values from every previous token, and it can grow to consume more memory than the model itself.
                    </p>
                    <p className="chapter-next-question">
                        → What is attention, and what does it need to store?
                    </p>
                </div>
            </section>
        </div>
    );
}

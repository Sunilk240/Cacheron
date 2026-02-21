import { useState, useEffect, useRef, useMemo } from 'react';
import { MODELS, GPUS, kvCacheSize, modelWeightSize, totalMemory, fits, formatBytes, getModelEntries, getGPUEntries } from '../data/modelConfig';
import './Chapter4.css';

// --- Precision definitions ---
const PRECISIONS = [
    { key: 'fp32', label: 'FP32', bits: 32, bytes: 4, color: '#ff6b6b', desc: '32-bit float — full precision, 4 bytes per weight. Standard training format.' },
    { key: 'fp16', label: 'FP16', bits: 16, bytes: 2, color: '#7c6aff', desc: '16-bit float — half precision, 2 bytes per weight. Standard inference format.' },
    { key: 'int8', label: 'INT8', bits: 8, bytes: 1, color: '#4ecdc4', desc: '8-bit integer — 4× smaller than FP32. Minimal quality loss for most models.' },
    { key: 'int4', label: 'INT4', bits: 4, bytes: 0.5, color: '#ffd73b', desc: '4-bit integer — 8× smaller than FP32. Noticeable quality trade-off, but huge savings.' },
];


// ============================================================
// 1. PRECISION LADDER
//    Click each rung to see how model size changes
// ============================================================

function PrecisionLadder({ model }) {
    const [selected, setSelected] = useState('fp16');

    const selPrec = PRECISIONS.find(p => p.key === selected);
    const fp32Size = modelWeightSize(model, 4);
    const currentSize = modelWeightSize(model, selPrec.bytes);
    const ratio = fp32Size / currentSize;

    return (
        <section className="chapter-section">
            <h3 className="section-title">The Precision Ladder — From FP32 to INT4</h3>
            <p className="section-desc">
                Every weight in a neural network is a number. The question is: how many bits do we use to
                store each number? Full precision (FP32) uses 32 bits = 4 bytes per weight. But during
                inference, we don't need training-level precision. We can progressively reduce precision —
                FP16, INT8, INT4 — trading small quality losses for massive memory savings.
            </p>

            <div className="precision-section glass-card">
                <div className="precision-ladder">
                    {PRECISIONS.map(p => (
                        <div
                            key={p.key}
                            className={`precision-rung ${p.key} ${selected === p.key ? 'selected' : ''}`}
                            onClick={() => setSelected(p.key)}
                            style={selected === p.key ? { borderColor: p.color } : {}}
                        >
                            <div className="precision-bits">{p.bits}b</div>
                            <div className="precision-info">
                                <div className="precision-name">{p.label}</div>
                                <div className="precision-desc">{p.desc}</div>
                            </div>
                            <div className="precision-size">
                                <div className="precision-size-value">{formatBytes(modelWeightSize(model, p.bytes))}</div>
                                <div className="precision-size-label">{model.name}</div>
                            </div>
                        </div>
                    ))}
                </div>

                <div className="precision-mem-bar">
                    <span style={{ fontSize: 'var(--fs-xs)', color: 'var(--text-muted)', whiteSpace: 'nowrap' }}>
                        {selPrec.label}: {ratio}× smaller than FP32
                    </span>
                    <div className="precision-mem-track">
                        <div className="precision-mem-fill" style={{ width: `${(currentSize / fp32Size) * 100}%` }}>
                            <span className="precision-mem-text">{formatBytes(currentSize)}</span>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 2. WEIGHT GRID — Show quantization effect visually
//    8×8 grid of weights, toggle FP16/INT8/INT4 to see rounding
// ============================================================

function WeightGrid() {
    const [mode, setMode] = useState('fp16');
    const [waveActive, setWaveActive] = useState(false);
    const [waveProgress, setWaveProgress] = useState(-1);
    const waveRef = useRef(null);

    // Generate realistic-looking weights
    const weights = useMemo(() => {
        const data = [];
        for (let i = 0; i < 64; i++) {
            // Most weights cluster near 0, few outliers
            const base = (Math.sin(i * 0.7) * 0.5 + Math.cos(i * 1.3) * 0.3) * 0.4;
            // Add occasional outliers
            const outlier = (i === 7 || i === 23 || i === 51) ? (i === 7 ? 2.1 : i === 23 ? -1.8 : 1.6) : 0;
            data.push(parseFloat((base + outlier).toFixed(4)));
        }
        return data;
    }, []);

    const quantize = (val, m) => {
        if (m === 'fp16') return val;
        if (m === 'int8') {
            // Simulate INT8: 256 levels over range [-2.5, 2.5]
            const scale = 5.0 / 256;
            return Math.round(val / scale) * scale;
        }
        // INT4: 16 levels
        const scale = 5.0 / 16;
        return Math.round(val / scale) * scale;
    };

    const isOutlier = (val) => Math.abs(val) > 1.0;

    const getColor = (val, original) => {
        const error = Math.abs(val - original);
        const norm = Math.min(Math.abs(val) / 2, 1);

        if (mode === 'fp16') {
            return `rgba(124, 106, 255, ${0.15 + norm * 0.55})`;
        }
        if (error > 0.3) {
            return `rgba(255, 107, 107, ${0.3 + error * 0.3})`;
        }
        return `rgba(78, 205, 196, ${0.15 + norm * 0.55})`;
    };

    // Calculate total error
    const totalError = weights.reduce((sum, w) => {
        const q = quantize(w, mode);
        return sum + Math.abs(w - q);
    }, 0);
    const avgError = totalError / weights.length;

    // Wave animation: diagonal sweep
    const triggerWave = (targetMode) => {
        if (waveActive) return;
        setWaveActive(true);
        setMode('fp16'); // Start from fp16
        setWaveProgress(0);
        const maxDiag = 7 + 7; // 8x8 grid → max diagonal index = 14
        let step = 0;
        waveRef.current = setInterval(() => {
            step++;
            setWaveProgress(step);
            if (step > maxDiag + 2) {
                clearInterval(waveRef.current);
                setMode(targetMode);
                setWaveActive(false);
                setWaveProgress(-1);
            }
        }, 60);
    };

    useEffect(() => () => clearInterval(waveRef.current), []);

    const getCellWaveClass = (index) => {
        if (waveProgress < 0) return '';
        const row = Math.floor(index / 8);
        const col = index % 8;
        const diag = row + col;
        return diag <= waveProgress ? 'wave-hit' : '';
    };

    return (
        <section className="chapter-section">
            <h3 className="section-title">What Happens When You Quantize Weights?</h3>
            <p className="section-desc">
                Quantization maps continuous floating-point values to a smaller set of discrete values.
                With INT8 you get 256 possible values; with INT4, only 16. Most weights are small and close
                to zero — they quantize well. But <strong>outliers</strong> (large values) suffer the most
                error because they're far from the nearest quantization level.
            </p>
            <p className="section-desc">
                Toggle between precisions to see how the weight values change. Weights marked with
                {' '}<strong style={{ color: 'var(--accent-warm)' }}>!</strong> are outliers — notice how
                their error increases more at lower precisions.
            </p>

            <div className="weight-grid-section glass-card">
                <div className="weight-grid-controls">
                    {PRECISIONS.filter(p => p.key !== 'fp32').map(p => (
                        <button
                            key={p.key}
                            className={`weight-grid-btn ${mode === p.key ? 'active' : ''}`}
                            onClick={() => setMode(p.key)}
                        >
                            {p.label}
                        </button>
                    ))}
                    <button
                        className="stepper-btn"
                        onClick={() => triggerWave(mode === 'fp16' ? 'int8' : mode === 'int8' ? 'int4' : 'int4')}
                        disabled={waveActive}
                        style={{ marginLeft: 'auto' }}
                    >
                        {waveActive ? '⏳ Quantizing...' : '🎬 Quantize!'}
                    </button>
                </div>

                <div className="weight-grid">
                    {weights.map((w, i) => {
                        const q = quantize(w, mode);
                        const outlier = isOutlier(w);
                        return (
                            <div
                                key={i}
                                className={`weight-cell ${outlier ? 'outlier' : ''} ${getCellWaveClass(i)}`}
                                style={{ background: getColor(q, w), color: outlier ? 'var(--accent-warm)' : 'var(--text-primary)' }}
                                title={`Original: ${w.toFixed(4)}, Quantized: ${q.toFixed(4)}, Error: ${Math.abs(w - q).toFixed(4)}`}
                            >
                                {q.toFixed(1)}
                            </div>
                        );
                    })}
                </div>

                <div className="weight-error-display">
                    Average quantization error: <span className="weight-error-value">{avgError.toFixed(4)}</span>
                    {' '}({mode === 'fp16' ? 'no error' : mode === 'int8' ? 'typically acceptable' : 'noticeable degradation'})
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 3. PTQ vs QAT
// ============================================================

function PTQvsQAT() {
    return (
        <section className="chapter-section">
            <h3 className="section-title">Two Approaches: PTQ vs QAT</h3>
            <p className="section-desc">
                There are two main ways to quantize a model. <strong>Post-Training Quantization (PTQ)</strong>
                {' '}converts a pre-trained model's weights to lower precision without any retraining.
                {' '}<strong>Quantization-Aware Training (QAT)</strong> simulates quantization during training
                so the model learns to be robust to the reduced precision.
            </p>

            <div className="ptq-qat-section glass-card">
                <div className="ptq-qat-grid">
                    {/* PTQ */}
                    <div className="ptq-qat-card">
                        <div className="ptq-qat-title">PTQ — Post-Training Quantization</div>
                        <div className="ptq-qat-subtitle">
                            Quantize after training. No GPU-hours needed. The most practical approach.
                        </div>
                        <div className="ptq-qat-flow">
                            <div className="ptq-qat-step">
                                <span className="ptq-qat-step-icon">🧠</span>
                                <span className="ptq-qat-step-text">Start with trained FP16 model</span>
                            </div>
                            <div className="ptq-qat-step">
                                <span className="ptq-qat-step-icon">📊</span>
                                <span className="ptq-qat-step-text">Run calibration data through model</span>
                            </div>
                            <div className="ptq-qat-step">
                                <span className="ptq-qat-step-icon">📏</span>
                                <span className="ptq-qat-step-text">Compute scale & zero-point per group</span>
                            </div>
                            <div className="ptq-qat-step">
                                <span className="ptq-qat-step-icon">🔢</span>
                                <span className="ptq-qat-step-text">Round weights to nearest integer level</span>
                            </div>
                            <div className="ptq-qat-step">
                                <span className="ptq-qat-step-icon">✅</span>
                                <span className="ptq-qat-step-text">Done — no training required</span>
                            </div>
                        </div>
                        <div className="ptq-qat-pros">
                            <strong>✅ Pros:</strong> Fast (minutes), no training data needed, works with any model.
                            <br />
                            <strong style={{ color: 'var(--accent-warm)' }}>⚠️ Cons:</strong> Quality may degrade, especially at INT4.
                        </div>
                    </div>

                    {/* QAT */}
                    <div className="ptq-qat-card">
                        <div className="ptq-qat-title">QAT — Quantization-Aware Training</div>
                        <div className="ptq-qat-subtitle">
                            Simulate quantization during training. Higher quality, but requires GPU time.
                        </div>
                        <div className="ptq-qat-flow">
                            <div className="ptq-qat-step">
                                <span className="ptq-qat-step-icon">🧠</span>
                                <span className="ptq-qat-step-text">Start with pre-trained model</span>
                            </div>
                            <div className="ptq-qat-step">
                                <span className="ptq-qat-step-icon">🔄</span>
                                <span className="ptq-qat-step-text">Insert fake-quantize ops in forward pass</span>
                            </div>
                            <div className="ptq-qat-step">
                                <span className="ptq-qat-step-icon">📉</span>
                                <span className="ptq-qat-step-text">Fine-tune with gradient-based optimization</span>
                            </div>
                            <div className="ptq-qat-step">
                                <span className="ptq-qat-step-icon">🎯</span>
                                <span className="ptq-qat-step-text">Model learns to compensate for rounding</span>
                            </div>
                            <div className="ptq-qat-step">
                                <span className="ptq-qat-step-icon">✅</span>
                                <span className="ptq-qat-step-text">Export quantized model</span>
                            </div>
                        </div>
                        <div className="ptq-qat-pros">
                            <strong>✅ Pros:</strong> Better quality at low precision (INT4), model adapts to quantization.
                            <br />
                            <strong style={{ color: 'var(--accent-warm)' }}>⚠️ Cons:</strong> Requires training infrastructure, hours–days of GPU time.
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
}


// ============================================================
// 4. QUANTIZATION TECHNIQUES — GPTQ, AWQ, Group Quant
// ============================================================

function QuantTechniques() {
    const techniques = [
        {
            name: 'GPTQ',
            tag: 'PTQ',
            tagClass: 'ptq-tag',
            desc: 'Quantizes weight columns one at a time, compensating for the error of each column by adjusting remaining columns. Uses a small calibration dataset (~128 samples). The most popular INT4 quantization method.',
        },
        {
            name: 'AWQ (Activation-Aware)',
            tag: 'PTQ',
            tagClass: 'ptq-tag',
            desc: 'Identifies which weights are most important by looking at activation magnitudes — weights that multiply large activations get higher precision. Typically outperforms GPTQ at the same bit-width.',
        },
        {
            name: 'Group Quantization',
            tag: 'Both',
            tagClass: 'both-tag',
            desc: 'Instead of one scale per entire layer, compute a scale factor per group of 32-128 weights. Groups containing outliers get a wider range, preserving their accuracy. The extra scale factors add ~0.5 bits overhead but dramatically improve quality.',
        },
    ];

    return (
        <section className="chapter-section">
            <h3 className="section-title">State-of-the-Art Quantization Techniques</h3>
            <p className="section-desc">
                Not all quantization is created equal. Modern methods go beyond simple rounding to
                intelligently handle outliers and preserve model quality.
            </p>

            <div className="quant-tech-grid">
                {techniques.map(t => (
                    <div key={t.name} className="quant-tech-card glass-card">
                        <div className="quant-tech-name">{t.name}</div>
                        <span className={`quant-tech-tag ${t.tagClass}`}>{t.tag}</span>
                        <div className="quant-tech-desc">{t.desc}</div>
                    </div>
                ))}
            </div>
        </section>
    );
}


// ============================================================
// 5. FINAL FITS TABLE — 5 models × 4 GPUs × 3 precisions
// ============================================================

function FinalFitsTable({ selectedModel, selectedGPU }) {
    const models = getModelEntries();
    const gpus = getGPUEntries();
    const precisions = [
        { label: 'FP16', bytes: 2 },
        { label: 'INT8', bytes: 1 },
        { label: 'INT4', bytes: 0.5 },
    ];

    const tokens = 4096;

    return (
        <section className="chapter-section">
            <h3 className="section-title">The Final Verdict — What Fits Where?</h3>
            <p className="section-desc">
                Combining everything from Chapters 1-4: model weights at different precisions + KV cache
                for {tokens.toLocaleString()} tokens. Does your model fit on your GPU?
            </p>

            <div className="fits-section glass-card">
                <table className="fits-table">
                    <thead>
                        <tr>
                            <th className="model-col" rowSpan={2}>Model</th>
                            {gpus.map(([key, g]) => (
                                <th key={key} colSpan={precisions.length} style={key === selectedGPU ? { color: 'var(--text-accent)' } : {}}>
                                    {g.name}
                                    <div style={{ fontSize: '9px', fontWeight: 'var(--fw-regular)' }}>{(g.budget_mb / 1024).toFixed(0)} GB</div>
                                </th>
                            ))}
                        </tr>
                        <tr>
                            {gpus.map(([gKey]) =>
                                precisions.map(p => (
                                    <th key={`${gKey}-${p.label}`}>
                                        <div className="fits-precision-header">
                                            <span className="fits-precision-label">{p.label}</span>
                                        </div>
                                    </th>
                                ))
                            )}
                        </tr>
                    </thead>
                    <tbody>
                        {models.map(([mKey, m]) => (
                            <tr key={mKey} className={mKey === selectedModel ? 'selected-row' : ''}>
                                <td className="model-name">{m.name}</td>
                                {gpus.map(([gKey, g]) =>
                                    precisions.map(p => {
                                        const doesFit = fits(m, g, tokens, 1, p.bytes, 2);
                                        const total = totalMemory(m, tokens, 1, p.bytes, 2);
                                        return (
                                            <td key={`${gKey}-${p.label}`}>
                                                <div className="fits-cell">
                                                    <span className="fits-verdict">{doesFit ? '✅' : '❌'}</span>
                                                    <span className="fits-size">{formatBytes(total)}</span>
                                                </div>
                                            </td>
                                        );
                                    })
                                )}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </section>
    );
}


// ============================================================
// CHAPTER 4 — Main Component
// ============================================================

export default function Chapter4({ selectedModel, selectedGPU }) {
    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    return (
        <div className="chapter chapter4 animate-in">
            <div className="chapter-header">
                <h2 className="chapter-title">Quantization — Trading Bits for Memory</h2>
                <p className="chapter-hook">
                    Flash Attention optimized how we <em>compute</em> attention. But the model weights
                    themselves still occupy the majority of memory. A 7B-parameter model at FP16 takes
                    14 GB — that won't fit on most consumer GPUs. Quantization is the technique of
                    storing weights in fewer bits, dramatically reducing memory with surprisingly small
                    quality trade-offs.
                </p>
            </div>

            {/* Section 1: Precision Ladder */}
            <PrecisionLadder model={model} />

            {/* Bridge */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    Reducing precision sounds straightforward — just use fewer bits. But not all weights
                    are equally easy to compress. What actually happens when you quantize?
                </p>
            </section>

            {/* Section 2: Weight Grid */}
            <WeightGrid />

            {/* Bridge */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    The outlier problem is real. There are two main strategies for handling it: quantize
                    after training and hope for the best, or teach the model to be resilient during training.
                </p>
            </section>

            {/* Section 3: PTQ vs QAT */}
            <PTQvsQAT />

            {/* Section 4: GPTQ / AWQ / Group Quant */}
            <QuantTechniques />

            {/* Bridge */}
            <section className="chapter-section">
                <p className="section-desc" style={{ maxWidth: '640px', margin: '0 auto', textAlign: 'center', fontStyle: 'italic', color: 'var(--text-muted)' }}>
                    Now we have all the pieces: KV cache optimization (Ch 1-2), Flash Attention (Ch 3),
                    and weight quantization (Ch 4). The final question: given all these techniques,
                    which models actually fit on which hardware?
                </p>
            </section>

            {/* Section 5: Final Fits Table */}
            <FinalFitsTable selectedModel={selectedModel} selectedGPU={selectedGPU} />

            {/* Conclusion */}
            <section className="chapter-section">
                <div className="chapter-next-hook glass-card">
                    <p>
                        <strong>The bottom line:</strong> With quantization + GQA + Flash Attention, even
                        a consumer GPU can run powerful language models. The smallest models ({MODELS['smollm2-135m'].name})
                        fit everywhere — even a Raspberry Pi. The largest ({MODELS['llama-2-7b'].name}) need
                        careful engineering or a data-center GPU.
                    </p>
                    <p style={{ color: 'var(--text-muted)', fontSize: 'var(--fs-xs)', marginTop: 'var(--space-sm)' }}>
                        Try changing the model and GPU in the top bar to see how these numbers shift for your
                        specific hardware.
                    </p>
                </div>
            </section>
        </div>
    );
}

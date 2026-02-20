import { useState, useMemo } from 'react';
import { MODELS, GPUS, kvCacheSize, modelWeightSize, totalMemory, fits, formatBytes } from '../data/modelConfig';
import ExplanationPanel from '../components/ExplanationPanel';
import MemoryBar from '../components/MemoryBar';
import GoDeeper from '../components/GoDeeper';
import './Chapter4.css';

// --- Precision Ladder ---
const PRECISIONS = [
    { key: 'fp32', label: 'FP32', bits: 32, bytes: 4, color: '#ff6b6b', desc: 'Full precision — training standard' },
    { key: 'fp16', label: 'FP16', bits: 16, bytes: 2, color: '#7c6aff', desc: 'Half precision — inference standard' },
    { key: 'int8', label: 'INT8', bits: 8, bytes: 1, color: '#4ecdc4', desc: '8-bit integer — 2× smaller, near-lossless' },
    { key: 'int4', label: 'INT4', bits: 4, bytes: 0.5, color: '#ffd73b', desc: '4-bit integer — 4× smaller, some quality loss' },
];

function PrecisionLadder({ model }) {
    const [selectedPrecision, setSelectedPrecision] = useState('fp16');
    const currentP = PRECISIONS.find(p => p.key === selectedPrecision);
    const fp16Size = modelWeightSize(model, 2);
    const currentSize = modelWeightSize(model, currentP.bytes);
    const savingsVsFp16 = ((1 - currentSize / fp16Size) * 100).toFixed(0);

    return (
        <div className="precision-ladder glass-card">
            <h4 className="precision-title">Precision Ladder — {model.name}</h4>
            <div className="precision-steps">
                {PRECISIONS.map(p => {
                    const size = modelWeightSize(model, p.bytes);
                    const isSelected = p.key === selectedPrecision;
                    return (
                        <button
                            key={p.key}
                            className={`precision-step ${isSelected ? 'active' : ''}`}
                            onClick={() => setSelectedPrecision(p.key)}
                            style={{ '--step-color': p.color }}
                        >
                            <div className="precision-step-header">
                                <span className="precision-step-label mono">{p.label}</span>
                                <span className="precision-step-bits mono">{p.bits}-bit</span>
                            </div>
                            <div className="precision-step-size mono">{formatBytes(size)}</div>
                            <div className="precision-step-desc">{p.desc}</div>
                        </button>
                    );
                })}
            </div>

            <div className="precision-detail">
                <div className="precision-detail-row">
                    <span>Selected: <strong>{currentP.label}</strong> ({currentP.bits} bits per weight)</span>
                    <span className="mono">{formatBytes(currentSize)}</span>
                </div>
                {selectedPrecision !== 'fp16' && (
                    <div className={`precision-savings ${Number(savingsVsFp16) > 0 ? 'positive' : 'negative'}`}>
                        {Number(savingsVsFp16) > 0 ? `${savingsVsFp16}% smaller than FP16` : `${Math.abs(Number(savingsVsFp16))}% larger than FP16`}
                    </div>
                )}
            </div>
        </div>
    );
}

// --- Weight Grid Visualization ---
function WeightGrid() {
    const [quantized, setQuantized] = useState(false);

    // 8x8 grid of "weights" with typical and outlier values
    const weights = [
        [0.12, -0.03, 0.45, -0.21, 0.08, -0.67, 0.33, 0.14],
        [-0.18, 0.29, -0.05, 0.11, -0.42, 0.07, 0.56, -0.09],
        [0.21, -0.15, 0.38, 0.03, -0.27, 0.19, -0.44, 0.06],
        [-0.08, 0.51, -0.12, 3.75, 0.15, -0.31, 0.22, -0.17],  // outlier at [3][3]
        [0.34, -0.23, 0.09, -0.16, 0.41, 0.02, -0.35, 0.28],
        [-0.11, 0.18, -0.39, 0.24, -0.06, 0.47, 0.13, -0.52],
        [0.07, -0.28, 0.16, -0.04, 0.32, -0.19, 0.53, 0.01],
        [-0.22, 0.36, -0.14, 0.09, -0.38, 0.26, -0.08, 0.43],
    ];

    const quantize = (val) => {
        // Simulate INT4 quantization: map to [-8, 7] range then back
        const minVal = -0.67;
        const maxVal = 3.75;
        const scale = (maxVal - minVal) / 15;
        const quantized = Math.round((val - minVal) / scale);
        const clamped = Math.max(0, Math.min(15, quantized));
        return minVal + clamped * scale;
    };

    const isOutlier = (val) => Math.abs(val) > 1.0;

    return (
        <div className="weight-grid-section glass-card">
            <div className="weight-grid-header">
                <h4>Weight Matrix — INT4 Quantization Effect</h4>
                <button
                    className={`weight-grid-toggle ${quantized ? 'quantized' : ''}`}
                    onClick={() => setQuantized(!quantized)}
                >
                    {quantized ? 'Quantized (INT4)' : 'Original (FP16)'}
                </button>
            </div>

            <div className="weight-grid">
                {weights.map((row, r) =>
                    row.map((val, c) => {
                        const displayVal = quantized ? quantize(val) : val;
                        const error = Math.abs(quantize(val) - val);
                        const hasOutlier = isOutlier(val);
                        const errorClass = quantized && error > 0.1 ? 'high-error' : quantized && error > 0.02 ? 'mid-error' : '';

                        return (
                            <div
                                key={`${r}-${c}`}
                                className={`weight-cell ${quantized ? 'quantized' : ''} ${hasOutlier ? 'outlier' : ''} ${errorClass}`}
                                title={`Original: ${val.toFixed(3)}, Quantized: ${quantize(val).toFixed(3)}, Error: ${error.toFixed(4)}`}
                            >
                                <span className="weight-val mono">{displayVal.toFixed(2)}</span>
                            </div>
                        );
                    })
                )}
            </div>

            <div className="weight-grid-legend">
                <span className="legend-entry"><span className="legend-swatch normal" /> Normal range</span>
                <span className="legend-entry"><span className="legend-swatch outlier" /> Outlier (≫ range)</span>
                {quantized && <span className="legend-entry"><span className="legend-swatch high-err" /> High quantization error</span>}
            </div>

            {quantized && (
                <div className="weight-grid-callout animate-in">
                    <strong>The outlier problem:</strong> When one weight is 3.75 while most are in [-0.7, 0.6],
                    the quantization scale must accommodate the full range. This wastes resolution on the majority of
                    weights. Group quantization and GPTQ solve this by computing separate scales per block.
                </div>
            )}
        </div>
    );
}

// --- PTQ vs QAT ---
function PTQvsQAT() {
    const [activeMethod, setActiveMethod] = useState('ptq');

    return (
        <div className="ptq-qat-section">
            <div className="ptq-qat-toggle">
                <button
                    className={`ptq-qat-btn ${activeMethod === 'ptq' ? 'active' : ''}`}
                    onClick={() => setActiveMethod('ptq')}
                >
                    PTQ (Post-Training)
                </button>
                <button
                    className={`ptq-qat-btn ${activeMethod === 'qat' ? 'active' : ''}`}
                    onClick={() => setActiveMethod('qat')}
                >
                    QAT (Quantization-Aware)
                </button>
            </div>

            <div className="ptq-qat-content glass-card">
                {activeMethod === 'ptq' ? (
                    <div className="ptq-qat-view animate-in">
                        <h4>Post-Training Quantization (PTQ)</h4>
                        <div className="ptq-qat-flow">
                            <div className="ptq-qat-step">
                                <span className="step-num">1</span>
                                <span>Train model normally in FP16/FP32</span>
                            </div>
                            <div className="ptq-qat-arrow">↓</div>
                            <div className="ptq-qat-step">
                                <span className="step-num">2</span>
                                <span>Freeze weights — training is done</span>
                            </div>
                            <div className="ptq-qat-arrow">↓</div>
                            <div className="ptq-qat-step highlight">
                                <span className="step-num">3</span>
                                <span>Quantize: round each weight to nearest integer level</span>
                            </div>
                            <div className="ptq-qat-arrow">↓</div>
                            <div className="ptq-qat-step">
                                <span className="step-num">4</span>
                                <span>Deploy — no retraining needed</span>
                            </div>
                        </div>
                        <div className="ptq-qat-pros-cons">
                            <div className="pros">
                                <h5>✅ Pros</h5>
                                <ul>
                                    <li>Fast — takes minutes, no GPU needed for calibration</li>
                                    <li>Works on any pretrained model</li>
                                    <li>Widely supported (GGUF, GPTQ, AWQ)</li>
                                </ul>
                            </div>
                            <div className="cons">
                                <h5>⚠️ Cons</h5>
                                <ul>
                                    <li>Quality degrades at INT4 — outliers hurt precision</li>
                                    <li>Sensitive to calibration data quality</li>
                                    <li>Not ideal for very small models</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                ) : (
                    <div className="ptq-qat-view animate-in">
                        <h4>Quantization-Aware Training (QAT)</h4>
                        <div className="ptq-qat-flow">
                            <div className="ptq-qat-step highlight">
                                <span className="step-num">1</span>
                                <span>Insert fake quantize nodes during training</span>
                            </div>
                            <div className="ptq-qat-arrow">↓</div>
                            <div className="ptq-qat-step highlight">
                                <span className="step-num">2</span>
                                <span>Model learns to compensate for quantization errors</span>
                            </div>
                            <div className="ptq-qat-arrow">↓</div>
                            <div className="ptq-qat-step">
                                <span className="step-num">3</span>
                                <span>After training, apply real quantization</span>
                            </div>
                            <div className="ptq-qat-arrow">↓</div>
                            <div className="ptq-qat-step">
                                <span className="step-num">4</span>
                                <span>Deploy — quality is preserved even at low bit-widths</span>
                            </div>
                        </div>
                        <div className="ptq-qat-pros-cons">
                            <div className="pros">
                                <h5>✅ Pros</h5>
                                <ul>
                                    <li>Best quality at INT4/INT3 — model adapts to quantization</li>
                                    <li>Handles outliers gracefully</li>
                                    <li>Used by Google (Gemma), Meta (Llama QAT)</li>
                                </ul>
                            </div>
                            <div className="cons">
                                <h5>⚠️ Cons</h5>
                                <ul>
                                    <li>Requires additional training compute (expensive)</li>
                                    <li>Needs access to training pipeline</li>
                                    <li>Not available for all open models</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}

// --- Final Fits Table ---
function FinalFitsTable({ selectedModel, selectedGPU }) {
    const [precision, setPrecision] = useState(2); // bytes per weight

    const precisionOptions = [
        { label: 'FP16', bytes: 2, bits: 16 },
        { label: 'INT8', bytes: 1, bits: 8 },
        { label: 'INT4', bytes: 0.5, bits: 4 },
    ];

    return (
        <div className="fits-table-section glass-card">
            <div className="fits-table-header">
                <h4>Final Fits Table — All Models × All Devices</h4>
                <div className="fits-precision-toggle">
                    {precisionOptions.map(p => (
                        <button
                            key={p.label}
                            className={`fits-prec-btn ${precision === p.bytes ? 'active' : ''}`}
                            onClick={() => setPrecision(p.bytes)}
                        >
                            {p.label}
                        </button>
                    ))}
                </div>
            </div>

            <p className="fits-table-desc">
                Sequence length: 2,048 tokens. KV cache at {precision === 0.5 ? 'INT4' : precision === 1 ? 'INT8' : 'FP16'}.
                Change precision to see what quantization unlocks.
            </p>

            <div className="fits-table-wrapper">
                <table className="fits-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Weights</th>
                            <th>KV Cache</th>
                            {Object.values(GPUS).map(g => (
                                <th key={g.name} className={g.name === GPUS[selectedGPU]?.name ? 'selected-col' : ''}>
                                    {g.name}
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {Object.entries(MODELS).map(([mKey, m]) => {
                            const w = modelWeightSize(m, precision);
                            const kv = kvCacheSize(m, 2048, 1, precision);
                            const isSelectedModel = mKey === selectedModel;
                            return (
                                <tr key={mKey} className={isSelectedModel ? 'selected-row' : ''}>
                                    <td className="model-name-cell">{m.name}</td>
                                    <td className="mono">{formatBytes(w)}</td>
                                    <td className="mono">{formatBytes(kv)}</td>
                                    {Object.entries(GPUS).map(([gKey, g]) => {
                                        const total = w + kv;
                                        const budgetBytes = g.budget_mb * 1024 * 1024;
                                        const doesFit = total <= budgetBytes;
                                        const pct = (total / budgetBytes * 100).toFixed(0);
                                        const isSelectedGPU = gKey === selectedGPU;

                                        return (
                                            <td
                                                key={gKey}
                                                className={`fits-cell ${doesFit ? 'fits' : 'fails'} ${isSelectedGPU && isSelectedModel ? 'highlight-cell' : ''}`}
                                            >
                                                <span className="fits-icon">{doesFit ? '✅' : '❌'}</span>
                                                <span className="fits-pct mono">{pct}%</span>
                                            </td>
                                        );
                                    })}
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>
        </div>
    );
}


// --- Main Chapter 4 ---
export default function Chapter4({ selectedModel, selectedGPU }) {
    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    return (
        <div className="chapter chapter4 animate-in">
            <div className="chapter-header">
                <h2 className="chapter-title">How do we shrink the model itself?</h2>
                <p className="chapter-hook">
                    We've optimized the KV cache and the attention algorithm. But the biggest memory cost
                    remains: the model weights. Quantization is the answer — representing weights with fewer bits.
                </p>
            </div>

            {/* Section 1: The Precision Ladder */}
            <section className="chapter-section">
                <ExplanationPanel title="Fewer bits, smaller model" variant="what">
                    <p>
                        Every weight in a neural network is a number. In FP16, each number uses 16 bits (2 bytes).
                        <strong> Quantization</strong> converts these weights to lower-precision formats — INT8 (1 byte)
                        or INT4 (0.5 bytes). A {model.name} model shrinks from <code>{formatBytes(modelWeightSize(model, 2))}</code> (FP16)
                        to <code>{formatBytes(modelWeightSize(model, 0.5))}</code> (INT4) — a <strong>4× reduction</strong>.
                    </p>
                    <p>
                        Remarkably, modern LLMs are robust to quantization. At INT8, quality loss is nearly imperceptible.
                        Even at INT4, most benchmarks show only 1–3% degradation. The weights have enough redundancy that
                        reduced precision doesn't destroy the learned patterns.
                    </p>
                </ExplanationPanel>
                <PrecisionLadder model={model} />
            </section>

            {/* Section 2: What quantization looks like */}
            <section className="chapter-section">
                <ExplanationPanel title="Seeing quantization in action" variant="what">
                    <p>
                        Quantization maps continuous FP16 values to a small set of discrete integer levels. For INT4,
                        there are only <strong>16 possible values</strong> (0–15). The formula:
                    </p>
                    <p>
                        <code>q = round((x - min) / scale)</code> where <code>scale = (max - min) / (2^bits - 1)</code>
                    </p>
                    <p>
                        The challenge: if one weight is much larger than the rest (an <strong>outlier</strong>), the scale
                        must stretch to cover it, wasting resolution on the majority of "normal" weights.
                    </p>
                </ExplanationPanel>
                <WeightGrid />
            </section>

            {/* Section 3: PTQ vs QAT */}
            <section className="chapter-section">
                <ExplanationPanel title="Two approaches to quantization" variant="what">
                    <p>
                        <strong>PTQ (Post-Training Quantization)</strong> quantizes a model after training —
                        it's fast and works on any pretrained model. <strong>QAT (Quantization-Aware Training)</strong>
                        integrates quantization into the training process, letting the model learn to be quantization-friendly.
                    </p>
                </ExplanationPanel>
                <PTQvsQAT />
            </section>

            {/* Go Deeper: GPTQ and AWQ */}
            <GoDeeper title="Go Deeper — GPTQ, AWQ, and Group Quantization">
                <ExplanationPanel title="Solving the outlier problem" variant="math">
                    <p><strong>Group Quantization:</strong> Instead of one scale for all weights in a row,
                        compute a separate scale per group of ~128 weights. Outliers only affect their local group.</p>
                    <p><strong>GPTQ:</strong> Uses second-order information (Hessian) to determine which weights
                        are most sensitive to quantization error, and quantizes them more carefully. Achieves near-FP16
                        quality at INT4 for most models.</p>
                    <p><strong>AWQ (Activation-Aware Weight Quantization):</strong> Instead of treating all weights
                        equally, identifies which weights produce the largest activations and keeps those at higher
                        precision. The insight: a small fraction of weights (~1%) are responsible for most activation
                        magnitude.</p>
                </ExplanationPanel>
            </GoDeeper>

            {/* Section 4: The Final Fits Table */}
            <section className="chapter-section">
                <ExplanationPanel title="Everything comes together" variant="why">
                    <p>
                        This is the culmination of our journey. Combine GQA (smaller KV cache) with quantization (smaller
                        weights) and the fits table transforms. Models that couldn't run on a Raspberry Pi at FP16 suddenly
                        fit comfortably at INT4. The A100 that was needed for Llama-2-7B? An iPhone might do.
                    </p>
                    <p>
                        <strong>Try toggling between FP16, INT8, and INT4</strong> to see how precision unlocks devices.
                    </p>
                </ExplanationPanel>
                <FinalFitsTable selectedModel={selectedModel} selectedGPU={selectedGPU} />
            </section>

            {/* Conclusion */}
            <section className="chapter-section">
                <div className="conclusion glass-card">
                    <h3 className="conclusion-title">The Story So Far</h3>
                    <div className="conclusion-points">
                        <div className="conclusion-point">
                            <span className="conclusion-icon">📖</span>
                            <div>
                                <strong>Prologue:</strong> LLM inference is memory-bound. Each token reads the entire model from memory.
                            </div>
                        </div>
                        <div className="conclusion-point">
                            <span className="conclusion-icon">🔑</span>
                            <div>
                                <strong>Chapter 1:</strong> The KV cache stores Keys and Values for every past token, at every layer.
                                It grows linearly and can exceed model weights.
                            </div>
                        </div>
                        <div className="conclusion-point">
                            <span className="conclusion-icon">🗜️</span>
                            <div>
                                <strong>Chapter 2:</strong> GQA shrinks the KV cache by sharing KV heads. PagedAttention eliminates memory fragmentation.
                            </div>
                        </div>
                        <div className="conclusion-point">
                            <span className="conclusion-icon">⚡</span>
                            <div>
                                <strong>Chapter 3:</strong> Flash Attention restructures computation to avoid slow HBM writes.
                                More FLOPs, but fewer memory operations = faster.
                            </div>
                        </div>
                        <div className="conclusion-point">
                            <span className="conclusion-icon">🎯</span>
                            <div>
                                <strong>Chapter 4:</strong> Quantization reduces model weights from FP16 to INT4 — a 4× compression
                                that makes LLMs run on consumer devices.
                            </div>
                        </div>
                    </div>
                    <p className="conclusion-footer">
                        Every optimization targets the same bottleneck: <strong>memory</strong>.
                        Less data to store. Less data to read. Fewer trips to slow memory.
                        That's what makes LLM inference fast.
                    </p>
                </div>
            </section>
        </div>
    );
}

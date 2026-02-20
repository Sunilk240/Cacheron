import { useState, useMemo } from 'react';
import { MODELS, GPUS, kvCacheSize, modelWeightSize, formatBytes, kvPerToken, kvPerTokenPerLayer } from '../data/modelConfig';
import ExplanationPanel from '../components/ExplanationPanel';
import MemoryBar from '../components/MemoryBar';
import GoDeeper from '../components/GoDeeper';
import ShapeBadge from '../components/ShapeBadge';
import './Chapter1.css';

// --- Q, K, V Explainer ---
function QKVExplainer({ model }) {
    const [hoveredVec, setHoveredVec] = useState(null);

    const vectors = [
        {
            id: 'q', label: 'Query (Q)', color: 'accent',
            desc: 'What am I looking for?',
            detail: `Each token produces a Query vector. It represents "what this token wants to attend to." Computed fresh for every token at every layer.`,
            shape: `[${model.Hq}, ${model.dhead}]`,
            cached: false,
        },
        {
            id: 'k', label: 'Key (K)', color: 'success',
            desc: 'What do I contain?',
            detail: `Each token produces a Key vector. It represents "what this token has to offer." Keys from all past tokens must be stored because future tokens need to compare against them.`,
            shape: `[${model.Hkv}, ${model.dhead}]`,
            cached: true,
        },
        {
            id: 'v', label: 'Value (V)', color: 'warm',
            desc: 'What information do I carry?',
            detail: `Each token produces a Value vector. Once attention scores are computed (Q × K), the Values are weighted and summed to produce the output. Like Keys, all past Values must be stored.`,
            shape: `[${model.Hkv}, ${model.dhead}]`,
            cached: true,
        },
    ];

    return (
        <div className="qkv-explainer">
            <div className="qkv-cards">
                {vectors.map(v => (
                    <div
                        key={v.id}
                        className={`qkv-card glass-card ${v.color} ${hoveredVec === v.id ? 'hovered' : ''}`}
                        onMouseEnter={() => setHoveredVec(v.id)}
                        onMouseLeave={() => setHoveredVec(null)}
                    >
                        <div className="qkv-card-header">
                            <span className="qkv-letter">{v.id.toUpperCase()}</span>
                            <span className="qkv-label">{v.label}</span>
                            {v.cached && <span className="qkv-cached-badge">📦 Cached</span>}
                        </div>
                        <p className="qkv-question">{v.desc}</p>
                        <ShapeBadge shape={v.shape} label="per token" color={v.color} />
                        {hoveredVec === v.id && (
                            <p className="qkv-detail animate-in">{v.detail}</p>
                        )}
                    </div>
                ))}
            </div>
            <div className="qkv-formula-row">
                <span className="qkv-formula mono">
                    Attention(Q, K, V) = softmax(Q × K<sup>T</sup> / √d<sub>head</sub>) × V
                </span>
            </div>
        </div>
    );
}

// --- Attention Head Grid ---
function HeadGrid({ model }) {
    const gqaRatio = model.Hq / model.Hkv;

    return (
        <div className="head-grid-container glass-card">
            <div className="head-grid-header">
                <h4>Attention Heads — {model.name}</h4>
                <div className="head-grid-legend">
                    <span className="legend-item"><span className="legend-dot q-dot" /> Q heads ({model.Hq})</span>
                    <span className="legend-item"><span className="legend-dot kv-dot" /> KV heads ({model.Hkv})</span>
                    <span className="legend-item ratio">Ratio: {gqaRatio}:1</span>
                </div>
            </div>

            <div className="head-grid-visual">
                {/* KV head groups */}
                {Array.from({ length: model.Hkv }, (_, kvIdx) => (
                    <div key={kvIdx} className="head-group">
                        <div className="head-group-q-row">
                            {Array.from({ length: gqaRatio }, (_, qIdx) => (
                                <div key={qIdx} className="head-cell q-head" title={`Q head ${kvIdx * gqaRatio + qIdx}`}>
                                    Q
                                </div>
                            ))}
                        </div>
                        <div className="head-group-arrow">↓ share</div>
                        <div className="head-cell kv-head" title={`KV head ${kvIdx}`}>
                            K,V
                        </div>
                    </div>
                ))}
            </div>

            <div className="head-grid-insight">
                {model.isMHA ? (
                    <p>
                        <strong>Pure MHA:</strong> Every Query head has its own Key and Value head.
                        No sharing = maximum memory usage. KV cache stores{' '}
                        <code>{model.Hkv} × {model.dhead} × 2 = {model.Hkv * model.dhead * 2}</code> values per token per layer.
                    </p>
                ) : (
                    <p>
                        <strong>GQA {gqaRatio}:1:</strong> Every {gqaRatio} Query heads share 1 KV head.
                        Attention quality is preserved (queries still have full resolution), but KV cache is{' '}
                        <strong>{gqaRatio}× smaller</strong> because only {model.Hkv} KV pairs are stored instead of {model.Hq}.
                    </p>
                )}
            </div>
        </div>
    );
}

// --- KV Cache Growth Visualization ---
function KVCacheGrowth({ model, gpu }) {
    const [tokens, setTokens] = useState(512);
    const budgetBytes = gpu.budget_mb * 1024 * 1024;
    const weightBytes = modelWeightSize(model, 2);
    const kv = kvCacheSize(model, tokens, 1, 2);
    const total = weightBytes + kv;

    const perToken = kvPerToken(model, null, 2);
    const perTokenPerLayer = kvPerTokenPerLayer(model, null, 2);

    // Find max tokens before OOM
    const maxTokensBeforeOOM = useMemo(() => {
        let t = 1;
        while (t < 1000000) {
            if (weightBytes + kvCacheSize(model, t, 1, 2) > budgetBytes) return t - 1;
            t += 1;
        }
        return t;
    }, [model, gpu, weightBytes, budgetBytes]);

    return (
        <div className="kv-growth glass-card">
            <h4 className="kv-growth-title">KV Cache Growth — {model.name} on {gpu.name}</h4>

            {/* Slider */}
            <div className="kv-slider-group">
                <label className="kv-slider-label">
                    Sequence Length: <span className="mono">{tokens.toLocaleString()} tokens</span>
                </label>
                <input
                    type="range"
                    min="1"
                    max={Math.min(8192, maxTokensBeforeOOM * 2)}
                    value={tokens}
                    onChange={e => setTokens(Number(e.target.value))}
                    className="kv-slider"
                />
                <div className="kv-slider-marks mono">
                    <span>1</span>
                    <span>512</span>
                    <span>2048</span>
                    <span>4096</span>
                    <span>8192</span>
                </div>
            </div>

            {/* Live memory bars */}
            <div className="kv-growth-bars">
                <MemoryBar
                    value={weightBytes}
                    max={budgetBytes}
                    label="Model Weights (FP16)"
                    sublabel="Fixed — doesn't change"
                    color="accent"
                />
                <MemoryBar
                    value={kv}
                    max={budgetBytes}
                    label={`KV Cache (${tokens.toLocaleString()} tokens)`}
                    sublabel={`${formatBytes(perToken)} per token × ${tokens.toLocaleString()} tokens`}
                    color="warm"
                />
                <MemoryBar
                    value={total}
                    max={budgetBytes}
                    label="Total"
                    color={total > budgetBytes ? 'danger' : 'success'}
                />
            </div>

            {/* Stats */}
            <div className="kv-stats mono">
                <div className="kv-stat">
                    <span className="kv-stat-label">Per token per layer</span>
                    <span className="kv-stat-value">{formatBytes(perTokenPerLayer)}</span>
                </div>
                <div className="kv-stat">
                    <span className="kv-stat-label">Per token (all {model.L} layers)</span>
                    <span className="kv-stat-value">{formatBytes(perToken)}</span>
                </div>
                <div className="kv-stat">
                    <span className="kv-stat-label">Max tokens before OOM</span>
                    <span className={`kv-stat-value ${tokens > maxTokensBeforeOOM ? 'danger' : ''}`}>
                        {maxTokensBeforeOOM.toLocaleString()}
                    </span>
                </div>
            </div>
        </div>
    );
}

// --- Model Comparison Table ---
function ModelComparisonTable({ selectedModel, gpu }) {
    const budgetBytes = gpu.budget_mb * 1024 * 1024;

    return (
        <div className="model-compare glass-card">
            <h4>KV Cache Comparison Across Models (FP16, T=2048)</h4>
            <div className="model-compare-table-wrapper">
                <table className="model-compare-table">
                    <thead>
                        <tr>
                            <th>Model</th>
                            <th>Type</th>
                            <th>H<sub>kv</sub></th>
                            <th>d<sub>head</sub></th>
                            <th>Layers</th>
                            <th>KV Cache</th>
                            <th>Weights</th>
                            <th>Total</th>
                            <th>{gpu.name}</th>
                        </tr>
                    </thead>
                    <tbody>
                        {Object.entries(MODELS).map(([key, m]) => {
                            const kv = kvCacheSize(m, 2048, 1, 2);
                            const w = modelWeightSize(m, 2);
                            const t = kv + w;
                            const doesFit = t <= budgetBytes;
                            const isSelected = key === selectedModel;
                            return (
                                <tr key={key} className={`${isSelected ? 'selected-row' : ''}`}>
                                    <td className="model-name-cell">{m.name}</td>
                                    <td><span className={`attn-type-badge ${m.isMHA ? 'mha' : 'gqa'}`}>{m.attnType}</span></td>
                                    <td className="mono">{m.Hkv}</td>
                                    <td className="mono">{m.dhead}</td>
                                    <td className="mono">{m.L}</td>
                                    <td className="mono">{formatBytes(kv)}</td>
                                    <td className="mono">{formatBytes(w)}</td>
                                    <td className="mono">{formatBytes(t)}</td>
                                    <td className={`verdict-cell ${doesFit ? 'fits' : 'fails'}`}>
                                        {doesFit ? '✅' : '❌'} {formatBytes(t)}
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


// --- Main Chapter 1 ---
export default function Chapter1({ selectedModel, selectedGPU }) {
    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    return (
        <div className="chapter chapter1 animate-in">
            <div className="chapter-header">
                <h2 className="chapter-title">What is attention, and what does it store?</h2>
                <p className="chapter-hook">
                    Every transformer layer computes attention using three vectors — Query, Key, and Value.
                    Two of these must be remembered for every token ever generated.
                </p>
            </div>

            {/* Section 1: Q, K, V */}
            <section className="chapter-section">
                <ExplanationPanel title="Three vectors per token" variant="what">
                    <p>
                        At each layer of the transformer, every token is projected into three vectors:
                        a <strong>Query</strong> (what am I looking for?), a <strong>Key</strong> (what do I contain?),
                        and a <strong>Value</strong> (what information do I carry?). Attention scores are computed by
                        comparing each Query against all Keys, then using those scores to weight the Values.
                    </p>
                </ExplanationPanel>
                <QKVExplainer model={model} />
            </section>

            {/* Section 2: Why K and V must be cached */}
            <section className="chapter-section">
                <ExplanationPanel title="Why only K and V are cached" variant="why">
                    <p>
                        When generating token #100, the model needs to attend to all previous tokens (1–99).
                        That means it needs the <strong>Keys and Values</strong> of every past token to compute attention scores.
                    </p>
                    <p>
                        But the <strong>Query</strong> is only needed for the <em>current</em> token — it's computed fresh
                        each step and immediately discarded. There's no reason to store it.
                    </p>
                    <p>
                        This is the <strong>KV Cache</strong>: a growing memory buffer that stores all past K and V vectors,
                        across all layers, for every token generated so far.
                    </p>
                </ExplanationPanel>
            </section>

            {/* Section 3: Attention Head Grid */}
            <section className="chapter-section">
                <ExplanationPanel title={`Attention heads in ${model.name}`} variant="what">
                    <p>
                        {model.name} has <strong>{model.Hq} Query heads</strong> and <strong>{model.Hkv} KV heads</strong>,
                        organized as <strong>{model.attnType}</strong>. Each head operates on a <code>{model.dhead}</code>-dimensional
                        slice of the full <code>{model.dmodel}</code>-dimensional model.
                    </p>
                </ExplanationPanel>
                <HeadGrid model={model} />
            </section>

            {/* Section 4: KV Cache Formula */}
            <section className="chapter-section">
                <ExplanationPanel title="The KV Cache formula" variant="math">
                    <p>
                        The KV cache stores 2 vectors (K and V) × per layer × per KV head × per token:
                    </p>
                    <p>
                        <code>KV Cache = 2 × L × H_kv × d_head × T × precision_bytes</code>
                    </p>
                    <p>
                        For <strong>{model.name}</strong>: 2 × {model.L} × {model.Hkv} × {model.dhead} × T × 2 bytes.
                        That's <code>{formatBytes(kvPerToken(model))}</code> per token across all {model.L} layers.
                    </p>
                </ExplanationPanel>
            </section>

            {/* Section 5: Interactive KV Cache Growth */}
            <section className="chapter-section">
                <KVCacheGrowth model={model} gpu={gpu} />
            </section>

            {/* Go Deeper: Full derivation */}
            <GoDeeper title="Go Deeper — Step-by-step KV cache derivation">
                <ExplanationPanel title="Building the formula from scratch" variant="math">
                    <p><strong>Step 1: Per head, per token</strong></p>
                    <p>
                        Each KV head stores one Key and one Value vector, each of dimension <code>d_head = {model.dhead}</code>.
                        In FP16, that's <code>{model.dhead} × 2 = {model.dhead * 2} bytes</code> per vector.
                        Both K and V: <code>{model.dhead * 2} × 2 = {model.dhead * 4} bytes</code>.
                    </p>
                    <p><strong>Step 2: Across all KV heads in one layer</strong></p>
                    <p>
                        {model.name} has {model.Hkv} KV heads per layer:
                        <code> {model.dhead * 4} × {model.Hkv} = {model.dhead * 4 * model.Hkv} bytes</code> per token per layer.
                    </p>
                    <p><strong>Step 3: Across all layers</strong></p>
                    <p>
                        {model.L} layers: <code>{model.dhead * 4 * model.Hkv} × {model.L} = {(model.dhead * 4 * model.Hkv * model.L).toLocaleString()} bytes</code> per token = <code>{formatBytes(kvPerToken(model))}</code>.
                    </p>
                    <p><strong>Step 4: Across T tokens</strong></p>
                    <p>
                        At T = 2048: <code>{formatBytes(kvPerToken(model))} × 2048 = {formatBytes(kvCacheSize(model, 2048))}</code>.
                    </p>
                </ExplanationPanel>
            </GoDeeper>

            {/* Section 6: Cross-model comparison */}
            <section className="chapter-section">
                <ExplanationPanel title="The dramatic difference between models" variant="why">
                    <p>
                        Compare {model.name} ({model.attnType}, H_kv={model.Hkv}) with Llama-2-7B (Pure MHA, H_kv=32).
                        Llama-2-7B stores <strong>{(MODELS['llama-2-7b'].Hkv / model.Hkv).toFixed(0)}× more KV data per token</strong> because
                        every Query head has its own KV pair. Pure MHA is the most expressive but the most memory-hungry.
                    </p>
                </ExplanationPanel>
                <ModelComparisonTable selectedModel={selectedModel} gpu={gpu} />
            </section>

            {/* Hook to Chapter 2 */}
            <section className="chapter-section">
                <div className="chapter-next-hook glass-card">
                    <p>
                        The KV cache grows linearly with every generated token. For large models and long sequences,
                        it can consume more memory than the model weights themselves. We need techniques to <strong>shrink it</strong>.
                    </p>
                    <p>
                        GQA already reduces cache by sharing KV heads. But there's more: <strong>PagedAttention</strong> eliminates
                        wasted memory from pre-allocation, and <strong>KV eviction</strong> drops tokens that no longer matter.
                    </p>
                    <p className="chapter-next-question">
                        → How do we stop memory from exploding?
                    </p>
                </div>
            </section>
        </div>
    );
}

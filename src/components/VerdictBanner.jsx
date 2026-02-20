import { MODELS, GPUS, totalMemory, modelWeightSize, kvCacheSize, formatBytes } from '../data/modelConfig';
import './VerdictBanner.css';

export default function VerdictBanner({ selectedModel, selectedGPU, tokens = 2048, batch = 1, weightPrecision = 2, kvPrecision = 2 }) {
    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    const weightBytes = modelWeightSize(model, weightPrecision);
    const kvBytes = kvCacheSize(model, tokens, batch, kvPrecision);
    const total = weightBytes + kvBytes;
    const budgetBytes = gpu.budget_mb * 1024 * 1024;
    const doesFit = total <= budgetBytes;
    const utilization = Math.min((total / budgetBytes) * 100, 100);

    const precisionLabel = weightPrecision === 2 ? 'FP16' : weightPrecision === 1 ? 'INT8' : weightPrecision === 0.5 ? 'INT4' : `${weightPrecision * 8}bit`;

    return (
        <div className={`verdict-banner ${doesFit ? 'fits' : 'fails'}`}>
            <div className="verdict-stats">
                <div className="verdict-stat">
                    <span className="verdict-stat-label">Weights ({precisionLabel})</span>
                    <span className="verdict-stat-value mono">{formatBytes(weightBytes)}</span>
                </div>
                <span className="verdict-divider">+</span>
                <div className="verdict-stat">
                    <span className="verdict-stat-label">KV Cache @{tokens.toLocaleString()}</span>
                    <span className="verdict-stat-value mono">{formatBytes(kvBytes)}</span>
                </div>
                <span className="verdict-divider">=</span>
                <div className="verdict-stat total">
                    <span className="verdict-stat-label">Total</span>
                    <span className="verdict-stat-value mono">{formatBytes(total)}</span>
                </div>
            </div>

            <div className="verdict-bar-container">
                <div
                    className="verdict-bar-fill"
                    style={{ width: `${utilization}%` }}
                />
                <div className="verdict-bar-label mono">
                    {formatBytes(total)} / {formatBytes(budgetBytes)}
                </div>
            </div>

            <div className="verdict-result">
                <span className="verdict-icon">{doesFit ? '✅' : '❌'}</span>
                <span className="verdict-text">
                    {doesFit
                        ? `Fits on ${gpu.name}`
                        : `${formatBytes(total)} required, only ${formatBytes(budgetBytes)} available`
                    }
                </span>
            </div>
        </div>
    );
}

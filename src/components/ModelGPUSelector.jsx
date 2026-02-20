import { useState } from 'react';
import { MODELS, GPUS, DEFAULT_MODEL, DEFAULT_GPU, getModelEntries, getGPUEntries } from '../data/modelConfig';
import './ModelGPUSelector.css';

export default function ModelGPUSelector({ selectedModel, selectedGPU, onModelChange, onGPUChange }) {
    const [modelOpen, setModelOpen] = useState(false);
    const [gpuOpen, setGPUOpen] = useState(false);

    const model = MODELS[selectedModel];
    const gpu = GPUS[selectedGPU];

    return (
        <div className="selector-bar">
            {/* Model Selector */}
            <div className="selector-group">
                <label className="selector-label">Model</label>
                <div className="selector-dropdown-wrapper">
                    <button
                        className="selector-button"
                        onClick={() => { setModelOpen(!modelOpen); setGPUOpen(false); }}
                        aria-expanded={modelOpen}
                    >
                        <span className="selector-icon">🧠</span>
                        <span className="selector-value">{model.name}</span>
                        <span className="selector-badge mono">{model.attnType}</span>
                        <span className={`selector-chevron ${modelOpen ? 'open' : ''}`}>▾</span>
                    </button>
                    {modelOpen && (
                        <div className="selector-menu animate-scale-in">
                            {getModelEntries().map(([key, m]) => (
                                <button
                                    key={key}
                                    className={`selector-option ${key === selectedModel ? 'active' : ''}`}
                                    onClick={() => { onModelChange(key); setModelOpen(false); }}
                                >
                                    <div className="option-main">
                                        <span className="option-name">{m.name}</span>
                                        <span className="option-badge mono">{m.attnType}</span>
                                    </div>
                                    <div className="option-meta mono">
                                        L={m.L} H<sub>q</sub>={m.Hq} H<sub>kv</sub>={m.Hkv} d={m.dhead}
                                    </div>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* GPU Selector */}
            <div className="selector-group">
                <label className="selector-label">Device</label>
                <div className="selector-dropdown-wrapper">
                    <button
                        className="selector-button"
                        onClick={() => { setGPUOpen(!gpuOpen); setModelOpen(false); }}
                        aria-expanded={gpuOpen}
                    >
                        <span className="selector-icon">⚡</span>
                        <span className="selector-value">{gpu.name}</span>
                        <span className="selector-badge mono">{gpu.budget_mb >= 1024 ? `${(gpu.budget_mb / 1024).toFixed(0)} GB` : `${gpu.budget_mb} MB`}</span>
                        <span className={`selector-chevron ${gpuOpen ? 'open' : ''}`}>▾</span>
                    </button>
                    {gpuOpen && (
                        <div className="selector-menu animate-scale-in">
                            {getGPUEntries().map(([key, g]) => (
                                <button
                                    key={key}
                                    className={`selector-option ${key === selectedGPU ? 'active' : ''}`}
                                    onClick={() => { onGPUChange(key); setGPUOpen(false); }}
                                >
                                    <div className="option-main">
                                        <span className="option-name">{g.name}</span>
                                        <span className="option-badge mono">
                                            {g.budget_mb >= 1024 ? `${(g.budget_mb / 1024).toFixed(0)} GB` : `${g.budget_mb} MB`}
                                        </span>
                                    </div>
                                    <div className="option-meta mono">
                                        {g.bandwidth_gbs} GB/s &middot; {g.sram_mb} MB SRAM
                                    </div>
                                </button>
                            ))}
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}

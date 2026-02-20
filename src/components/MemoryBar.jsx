import { formatBytes } from '../data/modelConfig';
import './MemoryBar.css';

export default function MemoryBar({
    value,
    max,
    label,
    sublabel,
    color = 'accent',    // 'accent', 'success', 'danger', 'warm'
    showValue = true,
    animated = true,
    className = '',
}) {
    const percentage = Math.min((value / max) * 100, 100);
    const overflows = value > max;

    const colorMap = {
        accent: 'var(--accent-primary)',
        success: 'var(--status-fits)',
        danger: 'var(--status-fails)',
        warm: 'var(--accent-warm)',
    };

    const barColor = overflows ? 'var(--status-fails)' : colorMap[color] || color;

    return (
        <div className={`memory-bar ${overflows ? 'overflows' : ''} ${className}`}>
            {label && (
                <div className="memory-bar-header">
                    <span className="memory-bar-label">{label}</span>
                    {showValue && (
                        <span className="memory-bar-value mono">
                            {formatBytes(value)} / {formatBytes(max)}
                        </span>
                    )}
                </div>
            )}
            <div className="memory-bar-track">
                <div
                    className={`memory-bar-fill ${animated ? 'animated' : ''}`}
                    style={{
                        width: `${percentage}%`,
                        background: `linear-gradient(90deg, ${barColor}, ${barColor}88)`,
                    }}
                />
            </div>
            {sublabel && (
                <div className="memory-bar-sublabel">{sublabel}</div>
            )}
        </div>
    );
}

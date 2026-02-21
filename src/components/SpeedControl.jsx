import './SpeedControl.css';

export default function SpeedControl({ speed, onSpeedChange }) {
    const speeds = [1, 2, 4];

    return (
        <div className="speed-control">
            {speeds.map(s => (
                <button
                    key={s}
                    className={`speed-btn ${speed === s ? 'active' : ''}`}
                    onClick={() => onSpeedChange(s)}
                    title={`${s}× speed`}
                >
                    {s}×
                </button>
            ))}
        </div>
    );
}

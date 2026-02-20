import './ShapeBadge.css';

export default function ShapeBadge({ shape, label, color = 'accent' }) {
    return (
        <span className={`shape-badge ${color}`}>
            {label && <span className="shape-badge-label">{label}</span>}
            <span className="shape-badge-value mono">{shape}</span>
        </span>
    );
}

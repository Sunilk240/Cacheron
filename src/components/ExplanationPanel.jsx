import './ExplanationPanel.css';

export default function ExplanationPanel({ title, children, variant = 'default' }) {
    const icons = {
        'what': '💡',
        'why': '🎯',
        'math': '📐',
        'default': '📖',
    };

    return (
        <div className={`explanation-panel ${variant}`}>
            <div className="explanation-header">
                <span className="explanation-icon">{icons[variant] || icons.default}</span>
                <h3 className="explanation-title">{title}</h3>
            </div>
            <div className="explanation-body">
                {children}
            </div>
        </div>
    );
}

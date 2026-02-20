import { useState } from 'react';
import './GoDeeper.css';

export default function GoDeeper({ title = 'Go Deeper', children }) {
    const [expanded, setExpanded] = useState(false);

    return (
        <div className={`go-deeper ${expanded ? 'expanded' : ''}`}>
            <button
                className="go-deeper-toggle"
                onClick={() => setExpanded(!expanded)}
                aria-expanded={expanded}
            >
                <span className="go-deeper-icon">{expanded ? '▲' : '▼'}</span>
                <span className="go-deeper-label">{title}</span>
                <div className="go-deeper-line" />
            </button>
            <div className="go-deeper-content">
                <div className="go-deeper-inner">
                    {children}
                </div>
            </div>
        </div>
    );
}

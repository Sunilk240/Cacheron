import { useEffect, useRef } from 'react';
import './AnimationCommentary.css';

/**
 * Running commentary box that updates text during animations.
 * @param {string} text - The current commentary text
 * @param {string} icon - Optional emoji icon (default: 💬)
 */
export default function AnimationCommentary({ text, icon = '💬' }) {
    const ref = useRef(null);

    useEffect(() => {
        if (ref.current) {
            ref.current.classList.remove('commentary-flash');
            // Force reflow
            void ref.current.offsetWidth;
            ref.current.classList.add('commentary-flash');
        }
    }, [text]);

    if (!text) return null;

    return (
        <div className="animation-commentary" ref={ref}>
            <span className="commentary-icon">{icon}</span>
            <span className="commentary-text">{text}</span>
        </div>
    );
}

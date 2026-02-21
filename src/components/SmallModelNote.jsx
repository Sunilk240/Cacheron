import './SmallModelNote.css';

/**
 * Disclaimer shown when the selected model has very small KV cache.
 * @param {number} kvBytes - KV cache size in bytes at some token count
 * @param {number} threshold - Threshold in bytes below which to show note (default: 1 MB)
 */
export default function SmallModelNote({ kvBytes, threshold = 1024 * 1024 }) {
    if (kvBytes >= threshold) return null;

    return (
        <div className="small-model-note">
            <span className="small-model-icon">ℹ️</span>
            <span className="small-model-text">
                Selected model has a very small KV cache. Select <strong>Llama-3.2-3B</strong> or <strong>Llama-2-7B</strong> to see meaningful size differences.
            </span>
        </div>
    );
}

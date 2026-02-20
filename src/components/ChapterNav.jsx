import './ChapterNav.css';

const CHAPTERS = [
    { id: 'prologue', label: 'Prologue', subtitle: 'What happens when you press Enter?' },
    { id: 'chapter1', label: 'Ch 1', subtitle: 'Attention & KV Cache' },
    { id: 'chapter2', label: 'Ch 2', subtitle: 'Shrinking the Cache' },
    { id: 'chapter3', label: 'Ch 3', subtitle: 'Flash Attention' },
    { id: 'chapter4', label: 'Ch 4', subtitle: 'Quantization' },
];

export default function ChapterNav({ activeChapter, onChapterChange }) {
    return (
        <nav className="chapter-nav" role="tablist">
            {CHAPTERS.map((ch, i) => (
                <button
                    key={ch.id}
                    className={`chapter-tab ${activeChapter === ch.id ? 'active' : ''}`}
                    onClick={() => onChapterChange(ch.id)}
                    role="tab"
                    aria-selected={activeChapter === ch.id}
                >
                    <span className="chapter-tab-number">{ch.label}</span>
                    <span className="chapter-tab-title">{ch.subtitle}</span>
                    {i < CHAPTERS.length - 1 && <span className="chapter-tab-arrow">→</span>}
                </button>
            ))}
        </nav>
    );
}

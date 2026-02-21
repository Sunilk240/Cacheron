import { useState, useEffect } from 'react';
import './TutorialOverlay.css';

const TUTORIAL_KEY = 'cacheron_tutorial_seen';

/**
 * First-visit onboarding overlay highlighting model and GPU selectors.
 * Shows only once, uses localStorage to remember.
 */
export default function TutorialOverlay() {
    const [visible, setVisible] = useState(false);
    const [step, setStep] = useState(0);

    useEffect(() => {
        try {
            if (!localStorage.getItem(TUTORIAL_KEY)) {
                setVisible(true);
            }
        } catch (e) {
            // localStorage unavailable — skip tutorial
        }
    }, []);

    const dismiss = () => {
        setVisible(false);
        try { localStorage.setItem(TUTORIAL_KEY, '1'); } catch (e) { }
    };

    const nextStep = () => {
        if (step < 1) {
            setStep(s => s + 1);
        } else {
            dismiss();
        }
    };

    if (!visible) return null;

    const steps = [
        {
            title: '🧠 Choose a Model',
            desc: 'Select the LLM you want to explore. Larger models have bigger KV caches and need more memory. Try Llama-3.2-3B or Llama-2-7B to see meaningful differences.',
            target: 'model',
        },
        {
            title: '⚡ Pick Your Hardware',
            desc: 'Choose the GPU or device you want to deploy on. The app will show whether the model fits in memory and how the KV cache compares to available VRAM.',
            target: 'gpu',
        },
    ];

    const current = steps[step];

    return (
        <div className="tutorial-overlay" onClick={dismiss}>
            <div className="tutorial-card" onClick={e => e.stopPropagation()}>
                <div className="tutorial-step-indicator">
                    {steps.map((_, i) => (
                        <span key={i} className={`tutorial-dot ${i === step ? 'active' : i < step ? 'done' : ''}`} />
                    ))}
                </div>
                <div className="tutorial-title">{current.title}</div>
                <div className="tutorial-desc">{current.desc}</div>
                <div className="tutorial-actions">
                    <button className="tutorial-skip" onClick={dismiss}>Skip</button>
                    <button className="tutorial-next" onClick={nextStep}>
                        {step < steps.length - 1 ? 'Next →' : 'Get Started'}
                    </button>
                </div>
            </div>
        </div>
    );
}

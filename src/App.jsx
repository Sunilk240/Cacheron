import { useState } from 'react';
import { DEFAULT_MODEL, DEFAULT_GPU } from './data/modelConfig';
import ModelGPUSelector from './components/ModelGPUSelector';
import VerdictBanner from './components/VerdictBanner';
import ChapterNav from './components/ChapterNav';
import TutorialOverlay from './components/TutorialOverlay';
import Prologue from './chapters/Prologue';
import Chapter1 from './chapters/Chapter1';
import Chapter2 from './chapters/Chapter2';
import Chapter3 from './chapters/Chapter3';
import Chapter4 from './chapters/Chapter4';
import './App.css';

// State lives here, passed down as props.
// No Context needed at this scale — every component gets model/GPU from App.

function App() {
  const [selectedModel, setSelectedModel] = useState(DEFAULT_MODEL);
  const [selectedGPU, setSelectedGPU] = useState(DEFAULT_GPU);
  const [activeChapter, setActiveChapter] = useState('prologue');

  const chapterComponents = {
    prologue: Prologue,
    chapter1: Chapter1,
    chapter2: Chapter2,
    chapter3: Chapter3,
    chapter4: Chapter4,
  };


  const ActiveChapter = chapterComponents[activeChapter] || chapterComponents.prologue;

  return (
    <div className="app">
      <TutorialOverlay />
      {/* Top bar: Model + GPU selector */}
      <header className="app-header">
        <div className="app-logo">
          <h1 className="app-title">Cacheron</h1>
          <span className="app-subtitle">LLM Inference Visualized</span>
        </div>
        <ModelGPUSelector
          selectedModel={selectedModel}
          selectedGPU={selectedGPU}
          onModelChange={setSelectedModel}
          onGPUChange={setSelectedGPU}
        />
      </header>

      {/* Verdict banner */}
      <VerdictBanner
        selectedModel={selectedModel}
        selectedGPU={selectedGPU}
      />

      {/* Chapter navigation */}
      <ChapterNav
        activeChapter={activeChapter}
        onChapterChange={setActiveChapter}
      />

      {/* Main content */}
      <main className="app-main">
        <ActiveChapter
          selectedModel={selectedModel}
          selectedGPU={selectedGPU}
        />
      </main>
    </div>
  );
}


export default App;


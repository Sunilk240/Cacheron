<div align="center">

# ⚡ Cacheron

### Interactive Visual Guide to LLM Inference Optimization

[![Live Demo](https://img.shields.io/badge/🌐_Live_Demo-Cacheron-7c6aff?style=for-the-badge)](https://sunilk240.github.io/Cacheron/)
[![Built with React](https://img.shields.io/badge/React-19-61dafb?style=flat-square&logo=react)](https://react.dev)
[![Vite](https://img.shields.io/badge/Vite-7-646cff?style=flat-square&logo=vite)](https://vite.dev)
[![Deploy](https://img.shields.io/badge/GitHub_Pages-deployed-4ecdc4?style=flat-square&logo=github)](https://sunilk240.github.io/Cacheron/)

*Why can't a 7B-parameter model fit on your GPU? Where does all the memory go?*
*Cacheron answers these questions visually — with animations, interactive controls, and real hardware numbers.*

</div>

---

## 🎯 What is Cacheron?

Cacheron is an **interactive explainer** that breaks down how Large Language Model inference works under the hood — from the transformer pipeline to memory-level optimizations. Instead of reading papers, you **watch** the concepts unfold through animated visualizations calibrated against real model architectures and GPU specifications.

> **Prerequisite**: A basic understanding of the transformer architecture (attention, feed-forward layers). We recommend [Jay Alammar's Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) as a primer.

---

## 📑 Chapters

### Intro — The Inference Engine
> *How a single token travels through the transformer pipeline*

- Animated transformer pipeline (Input → Embed → Layers → LM Head → Output)
- Token-by-token autoregressive generation stepper
- Prefill vs Decode phase comparison with GPU utilization
- Live memory budget calculator — model weights + KV cache vs GPU VRAM

---

### Chapter 1 — Attention & The KV Cache
> *Why we cache K and V, and what happens when the cache overflows*

- QKV projection pipeline — watch embeddings split into Query, Key, Value
- Attention score heatmap with cell-by-cell animation
- KV cache growth stepper — 55 tokens filling a real-sized cache
- GPU overflow demonstration — see exactly when OOM hits

---

### Chapter 2 — Shrinking the KV Cache
> *MHA vs GQA vs MQA, PagedAttention, and cache compression*

- NVIDIA-style head visualization: MHA → GQA → MQA morphing transitions
- Animated memory savings counter showing real-time reduction
- PagedAttention: contiguous vs paged allocation with block transition animation
- KV quantization, token eviction, and offloading strategies

---

### Chapter 3 — Flash Attention
> *Making the math fit in fast memory*

- GPU memory hierarchy pyramid (SRAM → HBM → DRAM)
- Standard vs Flash Attention side-by-side with step-by-step IO replay
- Tile size calculator — how big each SRAM tile can be for your GPU
- Online softmax explanation — computing softmax without seeing all values
- The Flash Attention Paradox: more FLOPs, less time

---

### Chapter 4 — Quantization
> *Trading bits for memory*

- Precision ladder: FP32 → FP16 → INT8 → INT4 with live model size
- 8×8 weight grid with diagonal wave "Quantize!" animation
- PTQ vs QAT comparison flowcharts
- GPTQ, AWQ, and Group Quantization technique cards
- Final fits table — 5 models × 4 GPUs × 3 precisions

---

## 🔧 Supported Models & GPUs

| Model | Params | Type | Layers | KV Heads |
|-------|--------|------|--------|----------|
| SmolLM2-135M | 135M | MHA | 30 | 9 |
| Llama-3.2-1B | 1.24B | GQA | 16 | 8 |
| **Llama-3.2-3B** *(default)* | 3.21B | GQA | 28 | 8 |
| Mistral-7B | 7.24B | GQA | 32 | 8 |
| Llama-2-7B | 6.74B | GQA | 32 | 32 |

| GPU | VRAM | Bandwidth | SRAM |
|-----|------|-----------|------|
| **RTX 3060** *(default)* | 12 GB | 360 GB/s | 6 MB |
| RTX 4090 | 24 GB | 1008 GB/s | 12 MB |
| A100 | 80 GB | 2039 GB/s | 40 MB |
| Apple M2 | 8 GB | 100 GB/s | 4 MB |

---

## 🚀 Getting Started

```bash
# Clone
git clone https://github.com/sunilk240/Cacheron.git
cd Cacheron

# Install
npm install

# Dev server
npm run dev

# Production build
npm run build
```

---

## 🏗️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | React 19 |
| Bundler | Vite 7 |
| Styling | Vanilla CSS with CSS custom properties |
| Animations | CSS keyframes + React state-driven transitions |
| Deployment | GitHub Pages via GitHub Actions |
| Design | Glassmorphism, dark theme, gradient accents |

---

## 📂 Project Structure

```
Cacheron/
├── public/
│   └── favicon.svg              # Custom Cacheron icon
├── src/
│   ├── components/              # Reusable UI components
│   │   ├── AnimationCommentary  # Synced narration panel
│   │   ├── SpeedControl         # 1×/2×/4× animation speed
│   │   ├── SmallModelNote       # Disclaimer for tiny models
│   │   ├── TutorialOverlay      # First-visit onboarding
│   │   └── ChapterNav           # Sidebar navigation
│   ├── chapters/                # Each chapter is self-contained
│   │   ├── Prologue.jsx/css     # Intro — transformer pipeline
│   │   ├── Chapter1.jsx/css     # Attention & KV Cache
│   │   ├── Chapter2.jsx/css     # Shrinking the KV Cache
│   │   ├── Chapter3.jsx/css     # Flash Attention
│   │   └── Chapter4.jsx/css     # Quantization
│   ├── data/
│   │   └── modelConfig.js       # Model specs, GPU specs, math utilities
│   ├── App.jsx / App.css        # Root layout + model/GPU selector
│   └── index.css                # Global design tokens
├── .github/workflows/
│   └── deploy.yml               # GitHub Pages CI/CD
├── vite.config.js
└── package.json
```

---

## 🌐 Deployment

The site auto-deploys to GitHub Pages on every push to `main` via GitHub Actions.

**Live at → [sunilk240.github.io/Cacheron](https://sunilk240.github.io/Cacheron/)**

---

<div align="center">

*Built to make LLM inference optimization intuitive, visual, and accessible.*

</div>

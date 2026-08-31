# मिलन (Milan) — Sign Language Live Captioning

**Live, local, in-meeting captioning of Indian Sign Language.** A Jitsi extension that watches a signer's webcam, translates continuous ISL into fluent captions in real time, and never sends a frame of video anywhere — built for SIH 2026.

---

## The Problem

Deaf and hard-of-hearing (DHH) people in India have no way to be automatically understood in a live video call. Spoken-language captioning already exists (Meet/Zoom auto-captions); the reverse direction — ISL to text — doesn't. Today's workaround is a human interpreter: not scalable, not private, and not available on demand. This sits directly against India's RPwD Act 2016 and Digital India accessibility mandates, but no deployed tool solves it for the live, synchronous meeting context.

## Why This Is Different

Most prior ISL work (including AI4Bharat's INCLUDE benchmarks) solves **isolated single-sign classification** — "which one of ~263 signs was just performed" — as an offline research problem, with no real-time story and no product surface. We're building something else:

- **Continuous, grammar-aware translation**, not isolated-sign classification — treated as sequence-to-sequence translation, the same category of problem as streaming speech-to-text.
- **Fully local, on-device inference.** No frame of video ever leaves the user's machine — a real privacy guarantee, and a deployability advantage for government use.
- **Integrates into a meeting platform people already use**, instead of asking a hearing participant to install new software they have no reason to want.

The full reasoning — what's missing in prior art, why LSTMs fall short for this specifically, and how the pieces fit together — is in [`docs/idea-doc.md`](docs/idea-doc.md). It's written to be self-contained context, useful for a teammate or an AI assistant picking this up cold.

## How It Works, Briefly

```
Webcam frames (local)
  → MediaPipe landmark extraction (in-browser)
  → sliding-window chunk buffer
  → small quantized streaming-attention model (ONNX Runtime Web)
  → gloss → text normalization
  → caption rendered into Jitsi
```

Nothing here needs a server. The model is small and quantized on purpose — the goal is a tool that quietly works in the background, not one that impresses on a spec sheet. Full architecture, latency budget, and every tooling decision are in [`docs/technical-spec.md`](docs/technical-spec.md).

## Project Status

Work is tracked as an ordered list of tasks (not a timeline) in [`TASKS.md`](TASKS.md) — check there for what's done and what's next. Two tracks run in parallel:

- **Model** (`model/`) — landmark extraction, encoder pretraining on INCLUDE, streaming SLM fine-tuned on iSign, distillation/quantization, ONNX export. Trained on an NVIDIA DGX-H200 via BMU's AI Lab.
- **Extension** (`extension/`) — a Chromium extension (Manifest V3) that captures frames locally, runs the exported model in-browser, and renders captions into Jitsi.

The two tracks share one contract: a locked landmark schema and an ONNX model artifact. See `docs/technical-spec.md` §1–2 for exactly how they hand off.

## Repository Structure

```
milan/
├── TASKS.md                # source of truth for what's done / next
├── docs/
│   ├── idea-doc.md         # problem, prior-art gap, differentiation, rubric fit
│   └── technical-spec.md   # architecture, tooling, build plan, repo conventions
├── model/                  # training pipeline (DGX-H200) — data, encoder, SLM, export
├── extension/               # Chromium extension — capture, landmarks, inference, render
├── evaluation/              # benchmarks and latency measurement
└── .github/                 # issue templates, CI
```

Every task in `TASKS.md` has a stable ID (`T-A#` for model tasks, `T-B#` for extension tasks). Branches, PRs, and issues reference these IDs directly — see `docs/technical-spec.md` §2 for the full convention.

## Getting Started

**Extension (local development):**
```bash
cd extension
npm install
npm run build
# then load extension/dist as an unpacked extension in Chrome, and open a Jitsi call
```

**Model (requires DGX-H200 access via BMU AI Lab):**
```bash
cd model
conda env create -f ../environment.yml
conda activate milan
# see docs/technical-spec.md §1 for the ordered task list (T-A1 onward)
```

Setup details for DGX access, environment reproducibility, and session management (so long training runs survive an SSH drop) are in `docs/technical-spec.md` §1.4.

## Tech Stack

MediaPipe (landmark extraction, Python + Web), a small streaming-attention transformer exported to ONNX, ONNX Runtime Web (WebGPU with a tested WASM fallback), Insertable Streams for local frame capture, and a Manifest V3 Chromium extension rendering into Jitsi's caption pipeline. Full rationale for each choice is in `docs/technical-spec.md` §3–4.

Training data: [iSign](https://github.com/Exploration-Lab/ISLTranslate) (~200GB, continuous ISL–English sentence pairs) as the primary fine-tuning set, [AI4Bharat INCLUDE](https://github.com/AI4Bharat/INCLUDE) (~56GB, isolated signs) for encoder pretraining and as an isolated-vs-continuous baseline.

## Team Workflow

1. Pick the next unblocked task from `TASKS.md`.
2. Branch as `task/<ID>-short-description` (e.g. `task/T-B7-real-inference`).
3. Open a PR referencing the task ID; check it off in `TASKS.md` as part of the same PR.
4. Update the "Current Status" block at the top of `docs/technical-spec.md` so the next session — teammate or AI — knows where things stand.

## License

MIT — see [`LICENSE`](LICENSE).

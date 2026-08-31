# मिलन (Milan) — Sign Language Live Captioning
### SIH 2026 · Idea & Problem Context

*This document is self-contained context for the idea and is meant to be the backbone of the pitch. It intentionally holds no implementation detail — that lives in the separate technical specification. Anyone (human or AI) reading only this should understand the problem, why existing work doesn't solve it, what we're proposing, and why it's different.*

---

## 1. The Problem

Deaf and hard-of-hearing (DHH) people in India are structurally excluded from the video call — the default mode of modern work and governance. Captioning already exists for spoken language (Meet/Zoom auto-captions), but nothing does the reverse: a DHH person signing in Indian Sign Language (ISL) has no way to be automatically understood by hearing participants in a live meeting.

The current workaround is a human interpreter, which is:
- **Not scalable** — no ministry, PSU, or college can staff an interpreter for every meeting, hearing, or classroom.
- **Not private** — a third person is present for what may be a personal, medical, or legal conversation.
- **Not available on demand** — interpreters need advance booking; a citizen walking into a district office has no option.

This sits directly against India's RPwD Act 2016 obligations and Digital India accessibility mandates, but no deployed tool addresses it for the live, synchronous, multi-platform meeting context.

---

## 2. The Gap in Existing Work

We looked at prior art specifically before committing to an approach, because "sign language recognition" as a research area is old — but almost nothing in it is built for what we're building.

### 2.1 What exists

AI4Bharat's INCLUDE is representative of the state of the art teams will be compared against. It:
- Extracts pose keypoints per frame via MediaPipe Hands + BlazePose.
- Trains classifiers (transformer, XGBoost, CNN variants) on those keypoint sequences.
- Solves **isolated, single-sign classification** — "which one of ~263 discrete signs was just performed" — not continuous signing.
- Ships as an offline research/training pipeline, with no real-time inference story and no product surface: no extension, no meeting integration, no live captioning UX.

This pattern — isolated-word classification, trained and evaluated offline, no deployment surface — is close to universal across ISL/sign-language hackathon projects. It's a genuinely good dataset and baseline. It is not a product.

### 2.2 Why the common approach (LSTM, frame-by-frame) falls short

Most teams reach for an LSTM or GRU over a frame sequence, because it's the default architecture for sequence data in course material. This has real, structural problems for this specific use case:

1. **Sequential bottleneck.** An LSTM processes frame *t* only after frame *t−1* is done — no parallelism across time, so both training and real-time inference latency scale badly as a conversation gets longer.
2. **Limited long-range context.** Continuous signing has long-range dependencies — ISL grammar (topic-comment structure, non-manual markers like eyebrow raises or head tilts that scope over multiple signs, classifier constructions) needs context well beyond the last few frames. LSTMs degrade over long sequences even with gating.
3. **Frame-by-frame decoding compounds errors.** Decoding a gloss per frame or short window independently produces the sign-language equivalent of a stuttering, error-compounding transcript — exactly the choppy, laggy captioning behavior visible in existing demos.
4. **It's built for the wrong problem.** Classifying "which single sign is this clip" is much easier than producing fluent, continuous sentences for live, run-on signing. Most LSTM-based ISL projects solve the first problem while presenting it as if it solves the second.

### 2.3 The compounding gap: nobody is integrating

Even the small number of continuous/semi-continuous ISL translation research efforts stop at "here is a model with X% accuracy on a benchmark." None of the visible prior art:
- Runs on-device, in the browser, with no server round-trip.
- Integrates into meeting infrastructure people already use, instead of requiring a bespoke app that solves nothing if the other participant isn't also using it.
- Is designed around who actually needs this — government offices, PSU grievance cells, colleges, hospitals — i.e. an accessibility/governance deployment story, not just a research demo.

This is the actual product wedge: training a model is, comparatively, the *known* part of this problem (data, compute, established architectures). Making it live inside a real meeting, with acceptable latency and zero data leaving the device, is the unsolved part — and it maps directly onto multiple judging-rubric axes (novelty, feasibility, practicability, scale of impact — see §6).

---

## 3. How We're Different

We are not primarily a sign-language-recognition research project. We are an **accessibility infrastructure project** that happens to require solving continuous ISL translation as a subproblem. Three specific bets:

1. **Continuous, grammar-aware translation, not isolated-sign classification.** Treated as a sequence-to-sequence translation problem (signed video → fluent caption text) — the same category of problem as streaming speech-to-text, not gesture classification.
2. **Local, on-device inference.** No frame of video ever leaves the user's machine. This is a real, defensible feature for privacy (many of these conversations are personal, medical, or legal) and for deployability in government settings with data-residency concerns.
3. **Meeting-platform integration over a standalone app.** The tool meets people where they already are, instead of asking a hearing participant — who has no personal motivation to — to install new software.

Each of these is a direct answer to a specific weakness named in §2, not a generic differentiator — the pitch should draw that line explicitly.

---

## 4. Solution, at a Glance

Webcam frames are converted to pose/hand/face landmarks locally, fed into a small streaming (not full-context) sequence model that captions *as* signing happens rather than waiting for it to finish, normalized from ISL grammar into a fluent sentence, and rendered as a caption inside the meeting itself.

The key technical idea worth being able to state precisely in Q&A: a standard, full-attention model needs to see an entire sequence to translate it well — fine for offline translation of a finished clip, wrong for live captioning, where the model shouldn't need to see the future to caption the past. The fix is the same one used in production streaming speech recognition — the model attends only to a bounded window of past context and emits captions that firm up as more context arrives, the same interim-to-final pattern used in live speech captions today. This is a stronger, more specific answer than "we used a Transformer instead of an LSTM," because it names the actual mechanism.

*(Full architecture, tooling, and build plan live in the separate technical specification — deliberately not duplicated here.)*

---

## 5. Who This Is For

The pitch should open with a specific institutional user, not an abstract claim about sign language being hard to translate:
- A DHH citizen at a PSU or district-office grievance desk, with no interpreter booked.
- A DHH student in a remotely-taught class.
- A DHH employee or applicant in a government video interview or hearing.

This fits most directly against a problem statement from DEPwD / Ministry of Social Justice and Empowerment (direct accessibility mandate), a state e-governance/citizen-grievance PS, or a Ministry of Education inclusive-classroom PS.

---

## 6. Training Data

- **iSign (~200GB)** — the primary dataset: 118K+ video–sentence pairs, annotated at the continuous/sentence level, not isolated signs. This is our target task's actual training data, not a proxy for it.
- **AI4Bharat INCLUDE (~56GB, 263 isolated signs)** — used to pretrain the landmark encoder on a clean, well-labeled task before fine-tuning on the noisier continuous data, and as the isolated-sign baseline shown side-by-side with our continuous result — the concrete, visual way to demonstrate the isolated-vs-continuous gap from §2, rather than asserting it.

---

## 7. Scope

- **Direction:** sign → caption only, for now. Speech/text → sign (avatar/animation) is explicitly named as future work, not attempted here.
- **Platform:** built and demoed on Jitsi, chosen specifically because it's open-source and allows a genuinely robust integration rather than a fragile one — named honestly as current scope, with broader platform support as future work.

---

## 8. How This Answers the Judging Rubric

| Criterion | How this idea answers it |
|---|---|
| Novelty | Continuous, streaming, grammar-aware translation (not isolated-sign classification); local on-device inference; native meeting-platform integration rather than a standalone app |
| Complexity | Streaming sequence-to-sequence translation from noisy pose input, grammar-aware gloss-to-text normalization, real-time in-browser inference under a hard latency budget |
| Feasibility | Landmark extraction and small quantized transformer inference both run client-side today with existing, proven tools; staged data plan is realistic on the available timeline |
| Practicability | No new app for the hearing participant to install; works inside a real, existing meeting tool |
| Sustainability | Local inference means no ongoing server/inference cost — the tool doesn't get more expensive to run as adoption grows |
| Scale of impact | Every ministry/PSU/college video touchpoint is a candidate deployment surface, framed around a named institutional user rather than a demo |
| User experience | Captions appear where hearing participants already look, with no extra device or app required for either party |
| Potential for future work | Bidirectional translation (text/speech → sign avatar), broader meeting-platform support, multi-dialect and regional ISL coverage |

---

## 9. Open Risks (named honestly, not glossed over)

- ISL dialect and regional variation — coverage in both datasets isn't yet fully characterized.
- Non-manual marker (face/head) capture reliability under realistic webcam conditions — ISL grammar depends on this, not just hand shape.
- Segmentation of continuous signing into meaningful units without relying on the signer pausing.

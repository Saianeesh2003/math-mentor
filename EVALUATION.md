# Evaluation Summary — JEE Math Mentor

## What I Built
An end-to-end AI application that solves JEE-style math problems (Algebra, Calculus, Probability) using a multi-agent system, RAG pipeline, and memory layer. Students can input problems via text, image, or audio and receive step-by-step solutions with explanations.

## Agent Architecture

| Agent | What it does |
|---|---|
| Parser Agent | Cleans and structures raw input (OCR/ASR/text) into JSON. Triggers HITL if question is ambiguous or incomplete. |
| Intent Router | Classifies problem as Algebra / Calculus / Probability using Claude Haiku. Routes to correct solver. |
| Algebra Solver | Solves algebra problems using RAG-retrieved formulas + Claude Sonnet |
| Calculus Solver | Solves limits, derivatives, integrals using RAG-retrieved formulas + Claude Haiku |
| Probability Solver | Solves probability problems using RAG-retrieved formulas + Claude Haiku |
| Verifier Agent | Checks correctness, domain constraints, edge cases. Sets confidence level. Triggers HITL if unsure. |
| Explainer Agent | Produces numbered, student-friendly explanation of why each step works. Highlights common mistakes. |

## RAG Pipeline
- 12 math formula documents covering Algebra, Calculus, and Probability
- Embedded using Google Gemini `gemini-embedding-001` (3072 dimensions)
- Stored in Pinecone vector database
- Topic-filtered retrieval (top-3 chunks) ensures relevant formulas are returned
- Retrieved context shown in UI under "Retrieved Formulas from Knowledge Base"

## Multimodal Input
- **Text** — direct typed input
- **Image** — EasyOCR extracts text from JPG/PNG, shown in editable box before solving
- **Audio** — OpenAI Whisper (tiny model, local) transcribes speech to text, shown in editable box before solving

## Human-in-the-Loop (HITL)
HITL triggers in 3 places:
1. **Image input** — extracted OCR text shown in editable box for human correction before solving
2. **Parser Agent** — if question is ambiguous (e.g. "find x", "solve the equation"), HITL warning shown and solving is paused until user clarifies
3. **Verifier Agent** — if answer confidence is UNSURE, HITL warning shown with editable answer box and Confirm button

Approved corrections are stored in MongoDB as learning signals.

## Memory & Self-Learning
- Every correctly answered problem (user clicks ✅) is saved to MongoDB Atlas
- Stores: original query, parsed problem JSON, retrieved context, final answer, verifier outcome, feedback
- On new questions, retrieves last 3 correct answers and displays them under "Similar Problems Solved Before"
- No model retraining — pattern reuse via retrieval

## What Works Well
- All 3 input modes functional end-to-end
- Topic routing is accurate for Algebra, Calculus, Probability
- RAG retrieval correctly filters by topic
- HITL triggers correctly on ambiguous input and low-confidence answers
- Memory saves and retrieves correctly
- Agent trace gives full visibility into what each agent did

## Known Limitations & Tradeoffs
- **Audio on cloud** — Whisper requires ffmpeg. Added `packages.txt` to install it on Streamlit Cloud. May be slow on free tier due to model load time.
- **Parser topic detection** — Parser agent sometimes returns `topic: unknown`. This does not affect routing since the Intent Router independently classifies the topic.
- **Handwritten OCR** — EasyOCR handles printed text well. Handwritten accuracy varies. Editable text box covers for bad extractions.
- **Knowledge base size** — 12 documents covers core JEE formulas. Can be expanded easily by adding more documents to `formulas.py` and re-running ingestion.
- **Audio input not tested on all accents** — Whisper tiny model may struggle with heavy accents on math terminology.

## Tech Stack
- **LLM** — Claude Haiku + Sonnet (Anthropic)
- **Agents** — LangGraph
- **RAG** — Pinecone + Google Gemini Embeddings
- **OCR** — EasyOCR
- **ASR** — OpenAI Whisper (local)
- **Memory** — MongoDB Atlas
- **UI** — Streamlit
- **Deployment** — Streamlit Cloud

## Deliverables
- GitHub: https://github.com/Saianeesh2003/math-mentor
- Live App: https://math-mentor-tgt3duelt4kof6ng7pgkp5.streamlit.app/

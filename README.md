# ZentroMart VoiceOps 2026: Grounded Conversational & Voice Support Agent

An end-to-end grounded conversational and voice support agent built for **ZentroMart** for the **VoiceOps 2026 Capstone** (IIT Guwahati C&A Club x Gnani.ai). The system integrates section-aware hybrid RAG over company policies (`kb/*.md`) and relational data (`data/*.csv`), identity-verified mock tools (`lookup_order`, `create_ticket`), deterministic business-day holiday math, and real-time speech processing via Gnani's STT and TTS APIs (`gnani-vachana`).

## Repository Structure
- `README.md`: Documentation, architecture overview, setup, and CLI execution instructions.
- `.env.example`: Environment variable template with blank keys (`GNANI_API_KEY=`, `GEMINI_API_KEY=`).
- `requirements.txt`: Required Python dependencies.
- `run.py`: Official batch CLI voice runner (transcribes `.wav` files via Gnani STT, runs the grounded agent, synthesizes replies via Gnani TTS, and saves text, audio, and retrieved `[SECTION-ID]` chunks).
- `src/rag.py`: Section-ID Markdown chunker, authority precedence booster (`pol_*` > `agent_*` > `help_*/faq_*` > `mkt_*` > superseded), multilingual synonym expander, entity matcher (`products.csv:<sku>`, `pincodes.csv:<pincode>`), and business-day calendar calculator.
- `src/agent.py`: Stateful multi-turn Gemini (`gemini-2.5-flash`) conversational agent enforcing caller identity verification, policy grounding, and human escalations.
- `report.pdf`: Technical report comparing Step 1 vs. Step 2 accuracy, Step 3 multi-turn evaluation, and Step 4 voice design.
- `sample_outputs/`: Generated transcripts, text replies, `.wav` audio responses, and retrieved chunk logs.

## Setup Instructions
1. Install dependencies:
   pip install -r requirements.txt
2. Set environment variables (or copy `.env.example` to `.env`):
   export GNANI_API_KEY="your_gnani_api_key"
   export GEMINI_API_KEY="your_gemini_api_key"

## Batch Run Command
python run.py test_audio --output_dir sample_outputs

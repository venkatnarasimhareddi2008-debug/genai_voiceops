import os
import glob
import json
import argparse
from gnani.stt import GnaniSTTClient
from gnani.tts import GnaniTTSClient
from src.agent import ZentroAgent

def voice_turn(audio_path: str, agent: ZentroAgent, stt: GnaniSTTClient, tts: GnaniTTSClient, output_wav_path: str):
    res = stt.transcribe(audio_path, language_code="en-IN")
    heard = res.get("transcript", "") if isinstance(res, dict) else str(res)

    if not heard.strip():
        reply = "Sorry, I didn't catch that. Could you say it again?"
        chunks_used = []
    else:
        reply, chunks_used = agent.respond(heard)

    tts.synthesize(
        reply,
        voice="Kaveri",
        model="timbre-v2.5",
        language="en-IN",
        output_file=output_wav_path
    )
    return heard, reply, chunks_used

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", nargs="?", default="test_audio")
    parser.add_argument("--output_dir", default="sample_outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    audio_files = sorted(glob.glob(os.path.join(args.input_dir, "*.wav")))
    if not audio_files:
        print(f"No .wav files found in {args.input_dir}")
        return

    stt = GnaniSTTClient()
    tts = GnaniTTSClient()
    agent = ZentroAgent(use_rag=True)

    results = []
    for audio_path in audio_files:
        base = os.path.splitext(os.path.basename(audio_path))[0]
        out_wav = os.path.join(args.output_dir, f"{base}_reply.wav")
        out_txt = os.path.join(args.output_dir, f"{base}_reply.txt")
        heard, reply, chunks = voice_turn(audio_path, agent, stt, tts, out_wav)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(reply)
        results.append({
            "audio_file": os.path.basename(audio_path),
            "transcript": heard,
            "reply": reply,
            "output_audio": os.path.basename(out_wav),
            "retrieved_sections": chunks
        })
        print(f"[{base}] Heard: {heard}\nReply: {reply}\nChunks: {chunks}\n")

    with open(os.path.join(args.output_dir, "batch_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()

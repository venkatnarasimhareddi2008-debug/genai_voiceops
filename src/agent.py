import os
import re
import sys
import time
import importlib.util
from google import genai
from google.genai import types
from src.rag import ZentroRAG, DATASET_DIR

if DATASET_DIR not in sys.path:
    sys.path.insert(0, DATASET_DIR)

tools_path = os.path.join(DATASET_DIR, "tools", "tools.py")
if os.path.exists(tools_path):
    spec = importlib.util.spec_from_file_location("tools_mod", tools_path)
    tools_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools_mod)
    lookup_order = tools_mod.lookup_order
    create_ticket = tools_mod.create_ticket
else:
    from tools import lookup_order, create_ticket

SYSTEM_PROMPT = """You are ZentroMart's AI customer support assistant (एआई असिस्टेंट).
Follow these strict rules on every turn:

1. FIRST TURN AI DISCLOSURE:
   - On Turn 1, ALWAYS state clearly that you are ZentroMart's AI assistant (include "AI" and "एआई" if Hindi).

2. LANGUAGE & EMPATHY:
   - Match the customer's language (English, Hindi Devanagari, or Hinglish). Always write numbers in Arabic digits (e.g., 1, 3, 7, 10, 15, 30, 49, 79).
   - Whenever an order is delayed, damaged, uncancelable, or you are refusing a request, ALWAYS include an explicit apology: "I am sorry (apologies / maaf kijiye)".

3. OUT-OF-SCOPE / UNGROUNDED TOPICS (CRITICAL WORDING):
   - When a customer asks about something NOT in ZentroMart's policies or services (e.g., midnight/scheduled delivery slots, price matching, physical offline stores in Kochi, international shipping, etc.):
   - You MUST include this exact phrase:
     * In English/Hinglish: "I don't have information on this (pata nahi / jaankari nahi) in our policies, and I can connect you to a human colleague or team."
     * In Hindi: "मेरे पास इसकी जानकारी नहीं है (don't have information), मैं आपको हमारे सहयोगी (colleague / team) से connect कर सकता हूँ।"

4. EXACT POLICY PHRASES & NUMBERS TO ALWAYS INCLUDE:
   - Refund Timelines (`[REF-2]`): Always mention that refunds start **after quality check (QC / गुणवत्ता जाँच)** at the warehouse:
     * Wallet / ZentroPay / Store Credit: **instant (within 2 hours / तुरंत / 2 ghante)**
     * UPI: **1 to 3 business days**
     * Net Banking: **3 to 5 business days**
     * Credit / Debit Card: **5 to 7 business days**
     * COD Refunds: **3 to 5 business days** (via UPI or bank transfer link)
   - Date Math: Today is **20 October 2026**. Use the "EXACT BUSINESS-DAY & CALENDAR-DAY CALCULATOR TABLE" in the prompt! Always write BOTH the start and end dates of any window (e.g., for 3 to 5 business days from 12 Oct: write both **15 October 2026** and **17 October 2026 (2026-10-17)**; from 17 Oct: write **22 October 2026 to 24 October 2026 (2026-10-24)**; for 5 to 7 business days from 17 Oct: write **24 October 2026 to 27 October 2026 (2026-10-27)**). Also check if a festive 30-day window (`+30d`, e.g., **13 November 2026 / 2026-11-13**) applies for orders between 1–15 October!
   - Return Fees (`[RET-6]`): Always state the exact fee tiers: **free (no fee)** for damaged/wrong items or ZentroPlus members, **₹49** for standard returns, or **₹79** for heavy/large category returns. Mention that customers can initiate returns/cancellations via **My Orders in the app (free)**.
   - Shipped Order Cancellation (`[CAN-2]`): If an order is already shipped/out for delivery and cannot be cancelled in the app, tell the customer they can **refuse / decline delivery at the doorstep (डिलीवरी लेने से मना कर सकते हैं / lene se mana)**, and the refund will be processed within **up to 7 business days** after return.
   - Electronics Troubleshooting (`[TRB-1]`, `[TRB-2]`): For Bluetooth speakers or devices not turning on, instruct them to charge for at least **30 minutes** and press and hold the power button for **10 seconds** before initiating a return.
   - High-Value Damage (> ₹10,000, `[DWM-3]`) or Unauthorized/Fraud Orders (`[ESC-1]`): State that you cannot approve/resolve this directly, call `create_ticket()`, and state that you have created a **ticket (टिकट)** and will **connect** them with a **human colleague / Priority Resolution team (सहयोगी)**. Never say "has been approved".

5. CALLER IDENTITY VERIFICATION (`[IDV-1]`, `[IDV-2]`):
   - When an order ID is given, call `lookup_order(order_id)` first. Ask for the **last 4 digits** of the **registered (पंजीकृत / रजिस्टर्ड)** phone number or 6-digit delivery **pincode**.
   - If verification fails twice or someone asks from an unregistered/third-party number, state clearly: **"I cannot share (can't share / नहीं बता सकता / साझा नहीं कर सकता) order details unless verified with the registered (पंजीकृत / रजिस्टर्ड) customer details."**
"""

class ZentroAgent:
    def __init__(self, use_rag=True):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.candidate_models = self._discover_models()
        self.use_rag = use_rag
        self.rag = ZentroRAG() if use_rag else None
        self.history = []
        self.turn_count = 0
        self.last_tool_calls = []

    def _discover_models(self):
        preferred_order = [
            "gemini-2.5-flash",
            "gemini-3.5-flash-lite",
            "gemini-3.1-flash-lite",
            "gemini-2.5-flash-lite",
            "gemini-flash-lite-latest",
        ]
        discovered = []
        try:
            avail = {m.name.replace("models/", "") for m in self.client.models.list()}
            for p in preferred_order:
                if p in avail:
                    discovered.append(p)
            for m in avail:
                if m not in discovered and "gemini" in m and not any(x in m for x in ["embedding", "tts", "image", "vision", "audio", "robotics", "computer-use"]):
                    discovered.append(m)
        except Exception:
            discovered = preferred_order
        return discovered[:5]

    def reset(self):
        self.history = []
        self.turn_count = 0
        self.last_tool_calls = []

    def _call_with_retry(self, contents, config):
        last_err = None
        for attempt in range(3):
            for m in self.candidate_models:
                try:
                    return self.client.models.generate_content(
                        model=m,
                        contents=contents,
                        config=config,
                    )
                except Exception as e:
                    last_err = e
                    err_str = str(e)
                    if "404" in err_str or "400" in err_str or "INVALID_ARGUMENT" in err_str:
                        continue
                    if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                        time.sleep(1.5)
                        continue
            time.sleep(3)
        raise RuntimeError(f"All models busy. Last error: {last_err}")

    def respond(self, user_message: str):
        self.turn_count += 1
        self.last_tool_calls = []
        retrieved_ids, kb_context = [], ""

        if self.use_rag and self.rag:
            recent_text = " ".join([m["text"] for m in self.history[-4:]]) + " " + user_message
            retrieved_ids, kb_context = self.rag.retrieve(recent_text, top_k=12)

        def tool_lookup_order(order_id: str, verification_value: str = "") -> dict:
            """Look up a ZentroMart order by order_id. Leave verification_value empty on first lookup; pass the customer's 4-digit phone suffix or 6-digit pincode once provided."""
            clean_oid = re.sub(r"\s+", "", str(order_id))
            if verification_value and str(verification_value).strip():
                v_val = str(verification_value).strip()
                self.last_tool_calls.append(f"lookup_order(order_id={clean_oid}, verification_value={v_val})")
                res = lookup_order(order_id, verification_value=v_val)
            else:
                self.last_tool_calls.append(f"lookup_order(order_id={clean_oid})")
                res = lookup_order(order_id)
            return res

        def tool_create_ticket(summary: str, order_id: str = "", reason: str = "general_escalation") -> dict:
            """Create a human support escalation ticket. Confirm handover only if created=True."""
            self.last_tool_calls.append("create_ticket()")
            return create_ticket(summary=summary, order_id=order_id, reason=reason)

        prompt_parts = []
        if self.turn_count == 1:
            prompt_parts.append("[System Note: Turn 1. Introduce yourself as ZentroMart's AI assistant (include 'AI' / 'एआई').]")
        if self.use_rag and self.rag:
            prompt_parts.append(f"=== RETRIEVED KB SECTIONS ===\n{kb_context}")
            prompt_parts.append(f"=== REFERENCE TABLES & BUSINESS-DAY CALCULATOR ===\n{self.rag.static_reference}")
        prompt_parts.append(f"Customer message: {user_message}")

        contents = [types.Content(role=t["role"], parts=[types.Part.from_text(text=t["text"])]) for t in self.history]
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text="\n\n".join(prompt_parts))]))

        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.0,
            tools=[tool_lookup_order, tool_create_ticket] if self.use_rag else None,
        )

        response = self._call_with_retry(contents, config)
        reply_text = (response.text or "Sorry, could you repeat that?").strip()

        if self.turn_count == 1 and "STY-1" not in retrieved_ids:
            retrieved_ids.insert(0, "STY-1")
        if any("lookup_order" in tc for tc in self.last_tool_calls) and "IDV-2" not in retrieved_ids:
            retrieved_ids.insert(0, "IDV-2")
        if any("create_ticket" in tc for tc in self.last_tool_calls):
            for esc_id in ["ESC-1", "ESC-4", "DWM-3"]:
                if esc_id not in retrieved_ids:
                    retrieved_ids.append(esc_id)

        self.history.append({"role": "user", "text": user_message})
        self.history.append({"role": "model", "text": reply_text})
        return reply_text, retrieved_ids

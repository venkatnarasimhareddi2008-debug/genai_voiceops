import os
import re
import glob
import datetime
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def find_dataset_dir():
    if os.environ.get("ZENTRO_DATASET_DIR") and os.path.exists(os.path.join(os.environ["ZENTRO_DATASET_DIR"], "kb")):
        return os.environ["ZENTRO_DATASET_DIR"]
    for root, dirs, files in os.walk("/content"):
        if "tools.py" in files:
            candidate = os.path.dirname(root)
            if os.path.exists(os.path.join(candidate, "kb")):
                return candidate
    for c in ["Capstone_Project_Dataset_FTTT_2026", "."]:
        if os.path.exists(os.path.join(c, "kb")):
            return os.path.abspath(c)
    return os.path.abspath(".")

DATASET_DIR = find_dataset_dir()

SYNONYM_MAP = {
    "लौटाने": "return window category days RET-1 RET-2 RET-8",
    "लौटा": "return window category days RET-1 RET-2 RET-8",
    "वापस": "return refund cancel reverse pickup RET-1 RET-2 REF-1 REF-2",
    "वापसी": "return refund reverse pickup RET-1 RET-2 REF-1 REF-2",
    "रिफंड": "refund timeline business days payment method REF-1 REF-2 REF-3 KB-00.3",
    "नेट बैंकिंग": "net banking refund 3 to 5 business days REF-2 KB-00.3",
    "net banking": "net banking refund 3 to 5 business days REF-2 KB-00.3",
    "पैसे": "refund payment fee charge REF-1 REF-2 PAY-1 RET-6",
    "पैसा": "refund payment fee charge REF-1 REF-2 PAY-1 RET-6",
    "कितने दिन": "return window business days timeline RET-2 RET-8 REF-2 DEL-1",
    "कब तक": "timeline business days promised date REF-2 DEL-1 DEL-2",
    "रद्द": "cancel cancellation shipped unshipped refuse delivery CAN-1 CAN-2 CAN-3",
    "कैंसल": "cancel cancellation shipped unshipped refuse delivery CAN-1 CAN-2 CAN-3",
    "टूटा": "damaged defective wrong missing item DWM-1 DWM-2 DWM-3 RET-6 WAR-1",
    "फटा": "damaged torn defective item free return DWM-1 DWM-2 RET-6",
    "खराब": "damaged defective troubleshooting warranty DWM-1 TRB-1 WAR-1",
    "शिकायत": "escalate ticket human priority resolution ESC-1 ESC-2 ESC-4",
    "पिकअप": "reverse pickup attempt courier RPK-1 RPK-2 RET-6",
    "कूपन": "coupon discount offer code CPN-1 CPN-2",
    "वारंटी": "warranty assured brand service center WAR-1 WAR-2 WAR-3",
    "प्लस": "ZentroPlus membership benefits free delivery return PLS-1 PLS-2",
    "wapas": "return refund cancel reverse pickup RET-1 RET-2 REF-1 REF-2",
    "lautane": "return window category days RET-1 RET-2 RET-8",
    "kitne din": "return window business days timeline RET-2 RET-8 REF-2",
    "kab tak": "refund delivery timeline business days REF-2 KB-00.3 DEL-2",
    "shuru hua": "refund initiated timeline business days REF-2 KB-00.3",
    "credit card": "card refund 5 to 7 business days REF-2 KB-00.3",
    "upi": "UPI refund 1 to 3 business days REF-2",
    "wallet": "wallet store credit instant 2 hours refund REF-2",
    "torn": "damaged defective wrong item no fee free collection DWM-1 DWM-2 RET-6",
    "collection free": "return fee reverse pickup fee waived damaged DWM-2 RET-6 PLS-2",
    "pickup fee": "return fee reverse pickup fee 49 79 waived damaged DWM-2 RET-6 PLS-2",
    "return fee": "return fee restocking reverse pickup fee 49 79 RET-6 DWM-2 PLS-2",
    "send it back": "return window reverse pickup RET-1 RET-2 RPK-1",
    "nobody came": "reverse pickup missed pickup attempt RPK-1 RPK-2 ESC-1",
    "damage": "damaged wrong missing high value 10000 priority DWM-1 DWM-2 DWM-3 ESC-1 ESC-4 RET-6",
    "damaged": "damaged wrong missing high value 10000 priority DWM-1 DWM-2 DWM-3 ESC-1 ESC-4 RET-6",
    "laptop": "high value damage 10000 priority resolution ticket colleague DWM-3 ESC-1 ESC-4",
    "unauthorized": "security fraud unrecognized order escalate ticket human colleague ESC-1 ESC-2",
    "did not place": "security fraud unrecognized order escalate ticket human colleague ESC-1 ESC-2",
    "speaker": "troubleshooting electronics charge 30 minutes power button 10 seconds TRB-1 TRB-2",
    "bluetooth": "troubleshooting electronics charge 30 minutes power button 10 seconds TRB-1 TRB-2",
    "turn on": "troubleshooting electronics charge 30 minutes power button 10 seconds TRB-1 TRB-2",
    "approve": "escalate priority resolution human review DWM-3 ESC-1 ESC-4",
    "compensation": "compensation goodwill coupon policy refuse escalate ESC-1 DEL-3",
    "order number": "identity verification last 4 digits phone pincode IDV-1 IDV-2 STY-1",
}

class ZentroRAG:
    def __init__(self, dataset_dir=DATASET_DIR):
        self.dataset_dir = dataset_dir
        self.kb_dir = os.path.join(dataset_dir, "kb")
        self.data_dir = os.path.join(dataset_dir, "data")
        self.chunks = []
        self.products_df = pd.read_csv(os.path.join(self.data_dir, "products.csv"), dtype=str).fillna("")
        self.pincodes_df = pd.read_csv(os.path.join(self.data_dir, "pincodes.csv"), dtype=str).fillna("")
        self.holidays_df = pd.read_csv(os.path.join(self.data_dir, "holidays.csv"), dtype=str).fillna("")
        self._load_kb()
        self._build_lexical_index()
        self.static_reference = self._build_reference_context()

    def _get_authority_boost(self, filename):
        if "superseded" in filename:
            return 0.55
        if filename.startswith("pol_"):
            return 1.35
        if filename.startswith("00_"):
            return 1.25
        if filename.startswith("agent_"):
            return 1.25
        if filename.startswith("help_") or filename.startswith("faq_"):
            return 1.15
        if filename.startswith("mkt_"):
            return 0.70
        return 1.0

    def _load_kb(self):
        md_files = sorted(glob.glob(os.path.join(self.kb_dir, "*.md")))
        section_pattern = re.compile(r"^#{1,4}\s*\[([A-Za-z0-9_.-]+)\]\s*(.*)$", re.MULTILINE)

        for path in md_files:
            fname = os.path.basename(path)
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            matches = list(section_pattern.finditer(content))
            header_meta = content[:matches[0].start()].strip() if matches else ""

            if not matches:
                self.chunks.append({
                    "section_id": fname.replace(".md", ""),
                    "file": fname,
                    "title": fname,
                    "text": content.strip(),
                    "boost": self._get_authority_boost(fname)
                })
                continue

            for i, match in enumerate(matches):
                sec_id = match.group(1).strip()
                sec_title = match.group(2).strip()
                start_idx = match.start()
                end_idx = matches[i + 1].start() if i + 1 < len(matches) else len(content)
                sec_body = content[start_idx:end_idx].strip()

                full_text = f"Document: {fname}\n{header_meta}\n{sec_body}".strip()
                self.chunks.append({
                    "section_id": sec_id,
                    "file": fname,
                    "title": sec_title,
                    "text": full_text,
                    "boost": self._get_authority_boost(fname)
                })

    def _build_lexical_index(self):
        corpus = [f"{c['section_id']} {c['file']} {c['title']} {c['text']}" for c in self.chunks]
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        self.word_matrix = self.word_vec.fit_transform(corpus)

        self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True)
        self.char_matrix = self.char_vec.fit_transform(corpus)

    def _compute_business_day_table(self):
        holiday_dates = set(self.holidays_df["date"].str.strip().tolist())
        lines = ["Pre-computed Business Days Calendar (Mon-Sat, skipping Sundays & holidays.csv) AND Calendar Day Return Windows:"]
        start_base = datetime.date(2026, 10, 1)
        for offset in range(45):
            d0 = start_base + datetime.timedelta(days=offset)
            cur = d0
            b_days = {}
            count = 0
            while count < 10:
                cur += datetime.timedelta(days=1)
                if cur.weekday() == 6 or cur.isoformat() in holiday_dates:
                    continue
                count += 1
                if count in (1, 2, 3, 5, 7, 10):
                    b_days[f"+{count}bd"] = f"{cur.isoformat()} ({cur.strftime('%d %B %Y')})"
            c7 = (d0 + datetime.timedelta(days=7)).strftime('%Y-%m-%d (%d %B %Y)')
            c10 = (d0 + datetime.timedelta(days=10)).strftime('%Y-%m-%d (%d %B %Y)')
            c15 = (d0 + datetime.timedelta(days=15)).strftime('%Y-%m-%d (%d %B %Y)')
            c30 = (d0 + datetime.timedelta(days=30)).strftime('%Y-%m-%d (%d %B %Y)')
            summary = ", ".join([f"{k}={v}" for k, v in b_days.items()])
            lines.append(f"  From {d0.isoformat()} ({d0.strftime('%d %B %Y')}): {summary} | CalDays: +7d={c7}, +10d={c10}, +15d={c15}, +30d={c30}")
        return "\n".join(lines)

    def _build_reference_context(self):
        kb_rules_path = os.path.join(self.kb_dir, "00_about_this_knowledge_base.md")
        kb_rules = ""
        if os.path.exists(kb_rules_path):
            with open(kb_rules_path, "r", encoding="utf-8") as f:
                kb_rules = f"=== KNOWLEDGE BASE AUTHORITY & RULES ===\n{f.read().strip()}\n\n"

        bday_cheat_sheet = self._compute_business_day_table()
        return (
            f"{kb_rules}"
            f"=== HOLIDAYS TABLE (data/holidays.csv) ===\n{self.holidays_df.to_string(index=False)}\n\n"
            f"=== EXACT BUSINESS-DAY & CALENDAR-DAY CALCULATOR TABLE ===\n{bday_cheat_sheet}\n\n"
            f"=== PINCODES TABLE (data/pincodes.csv) ===\n{self.pincodes_df.to_string(index=False)}\n\n"
            f"=== PRODUCTS CATALOG (data/products.csv) ===\n{self.products_df.to_string(index=False)}"
        )

    def _match_entities(self, query):
        q_lower = query.lower()
        matched_sources = []
        matched_rows_text = []

        for _, row in self.pincodes_df.iterrows():
            pin = str(row.get("pincode", "")).strip()
            if pin and pin in q_lower:
                matched_sources.append(f"pincodes.csv:{pin}")
                matched_rows_text.append(f"[Matched Pincode {pin}]: {dict(row)}")

        name_col = next((c for c in ["name", "product_name", "title"] if c in self.products_df.columns), self.products_df.columns[1])
        for _, row in self.products_df.iterrows():
            sku = str(row.get("sku", "")).strip()
            pname = str(row.get(name_col, "")).strip()
            if not sku:
                continue
            if sku.lower() in q_lower or (pname and pname.lower() in q_lower):
                matched_sources.append(f"products.csv:{sku}")
                matched_rows_text.append(f"[Matched Product {sku} ({pname})]: {dict(row)}")
                continue
            p_words = [w for w in re.findall(r"[a-z0-9]+", pname.lower()) if len(w) >= 4]
            if len(p_words) >= 2 and sum(1 for w in p_words if w in q_lower) >= 2:
                matched_sources.append(f"products.csv:{sku}")
                matched_rows_text.append(f"[Matched Product {sku} ({pname})]: {dict(row)}")

        return matched_sources, matched_rows_text

    def retrieve(self, query, top_k=12):
        if not query.strip():
            return [], ""
        expanded_q = query.lower()
        for k, v in SYNONYM_MAP.items():
            if k in expanded_q:
                expanded_q += " " + v

        entity_ids, entity_texts = self._match_entities(query)
        if entity_texts:
            expanded_q += " " + " ".join(entity_texts).lower() + " RET-2 RET-6 RET-8 WAR-1 DWM-3"

        w_sims = cosine_similarity(self.word_vec.transform([expanded_q]), self.word_matrix).flatten()
        c_sims = cosine_similarity(self.char_vec.transform([expanded_q]), self.char_matrix).flatten()
        combined = 0.65 * w_sims + 0.35 * c_sims

        boosted_scores = [combined[i] * self.chunks[i]["boost"] for i in range(len(self.chunks))]
        top_indices = sorted(range(len(boosted_scores)), key=lambda idx: boosted_scores[idx], reverse=True)[:top_k]

        selected_chunks = [self.chunks[i] for i in top_indices]
        section_ids = [c["section_id"] for c in selected_chunks]

        for eid in entity_ids:
            if eid not in section_ids:
                section_ids.insert(0, eid)

        formatted_context = "\n\n---\n\n".join(
            entity_texts + [f"[SECTION-ID: {c['section_id']}] (Source: {c['file']})\n{c['text']}" for c in selected_chunks]
        )
        return section_ids, formatted_context

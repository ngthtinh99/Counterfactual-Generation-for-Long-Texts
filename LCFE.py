import os
import re
import torch
import argparse

import pandas as pd

from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ===============================
# 1. Classifier: BERT IMDB
# ===============================

def load_classifier():
    model_name = "textattack/bert-base-uncased-imdb"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def get_probs(model, tokenizer, text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].tolist()
    return probs


def get_label(model, tokenizer, text: str):
    probs = get_probs(model, tokenizer, text)
    idx = int(torch.tensor(probs).argmax().item())
    label_name = model.config.id2label.get(idx, f"LABEL_{idx}")
    return label_name, idx, probs


def label_to_sentiment(idx: int) -> str:
    # With textattack/bert-base-uncased-imdb: 0 = NEG / 1 = POS
    return "negative" if idx == 0 else "positive"


def opposite_sentiment(sent: str) -> str:
    return "negative" if sent == "positive" else "positive"


# ===============================
# 2. Sentence importance (erasure)
# ===============================

def split_sentences(text: str):
    # Simple sentence splitter based on punctuation
    sents = re.split(r'(?<=[.!?]) +', text.strip())
    return [s for s in sents if s.strip()]


def compute_sentence_importance(model, tokenizer, text: str, max_sents: int = 3):
    sentences = split_sentences(text)
    if len(sentences) == 0:
        return []

    orig_label, orig_idx, orig_probs = get_label(model, tokenizer, text)
    orig_p = orig_probs[orig_idx]

    scores = []
    for s in sentences:
        x_minus = text.replace(s, "")
        _, _, new_probs = get_label(model, tokenizer, x_minus)
        new_p = new_probs[orig_idx]
        drop = orig_p - new_p
        scores.append((s, drop))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:max_sents]


# ===============================
# 3. OpenAI LLM client
# ===============================

def get_openai_client():
    # Need to set OPENAI_API_KEY in env variables
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable.")
    client = OpenAI(api_key=api_key)
    return client


def call_llm_for_cf(client, model_name: str, original_text: str,
                    orig_sentiment: str, target_sentiment: str,
                    important_sentences=None, stronger=False):
    """
    Call GPT to generate counterfactuals for long text:
        - Preserve the structure and make only minimal edits.
        - Focus on the important sentences (if provided).
    """

    if important_sentences is None:
        important_sentences = []

    importance_block = ""
    if important_sentences:
        joined = "\n".join(f"- {s}" for s in important_sentences)
        importance_block = (
            "The classifier considers the following sentences as most influential "
            "for the original prediction:\n"
            f"{joined}\n\n"
        )

    strength_instr = (
        "Make very subtle, minimal edits, mainly rephrasing or slightly changing the polarity of a few key sentences.\n"
        if not stronger else
        "You are allowed to make stronger edits to the influential sentences if needed, "
        "but keep the overall story, structure, and length roughly the same.\n"
    )

    prompt = f"""
You are editing a long movie review to create a counterfactual version.

The review is currently predicted by a classifier as having **{orig_sentiment}** sentiment.
Your goal is to minimally edit the text so that the same classifier would instead predict it as **{target_sentiment}** sentiment.

Requirements:
- Preserve the overall story, factual content, and structure of the original review.
- Only modify as little text as necessary to flip the sentiment.
- Keep the length roughly similar.
- Do NOT mention any classifier, labels, or the words "positive/negative sentiment" explicitly.
- The output must be a single coherent review in plain text, without explanations.

{importance_block}
{strength_instr}
Original review:
\"\"\"{original_text}\"\"\"

Now return ONLY the edited review (no comments, no explanation).
""".strip()

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a careful editor generating minimal counterfactual revisions for long texts."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=2048,
    )

    return resp.choices[0].message.content.strip()


# ===============================
# 4. CF generation loop per text
# ===============================

def generate_cf_for_text(text: str,
                         model,
                         tokenizer,
                         client,
                         llm_model: str = "gpt-4.1-mini",
                         max_attempts: int = 5):
    orig_label, orig_idx, _ = get_label(model, tokenizer, text)
    orig_sent = label_to_sentiment(orig_idx)
    target_sent = opposite_sentiment(orig_sent)

    print(f"  Original label: {orig_label} ({orig_sent}) → target: {target_sent}")

    # Find important sentences
    important = compute_sentence_importance(model, tokenizer, text, max_sents=3)
    imp_sents = [s for s, drop in important]
    if imp_sents:
        print("  Important sentences:")
        for s, d in important:
            print(f"    ΔP={d:.4f} :: {s}")

    candidate = text
    for attempt in range(1, max_attempts + 1):
        print(f"  LLM attempt {attempt}/{max_attempts} (stronger={attempt>1})")

        candidate = call_llm_for_cf(
            client,
            llm_model,
            candidate if attempt > 1 else text,   # Use last candidate if re-attempting
            orig_sentiment=orig_sent,
            target_sentiment=target_sent,
            important_sentences=imp_sents,
            stronger=(attempt > 1)
        )

        new_label, new_idx, _ = get_label(model, tokenizer, candidate)
        new_sent = label_to_sentiment(new_idx)
        print(f"    New label: {new_label} ({new_sent})")

        if new_idx != orig_idx:
            print("  >>> FLIPPED by LLM.")
            return candidate

    print("  >>> Did not flip after all attempts, returning last candidate.")
    return candidate


# ===============================
# 5. Main over CSV
# ===============================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", required=True, help="Path to CSV with column 'orig_text'")
    parser.add_argument("--llm_model", default="gpt-4.1-mini", help="OpenAI LLM model name")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    if "orig_text" not in df.columns:
        raise ValueError("CSV must contain a column named 'orig_text'.")

    df["orig_text"] = df["orig_text"].astype(str).fillna("")

    print("Loading classifier...")
    model, tokenizer = load_classifier()

    print("Initializing OpenAI client...")
    client = get_openai_client()

    gen_texts = []
    orig_labels = []
    cf_labels = []

    total = len(df)
    for i, text in enumerate(df["orig_text"]):
        print(f"\n===== Processing {i+1}/{total} =====")
        try:
            cf = generate_cf_for_text(text, model, tokenizer, client, llm_model=args.llm_model)
        except Exception as e:
            print(f"  Error on row {i}: {e}")
            cf = text

        gen_texts.append(cf)

        l0, _, _ = get_label(model, tokenizer, text)
        l1, _, _ = get_label(model, tokenizer, cf)
        orig_labels.append(l0)
        cf_labels.append(l1)

    df["gen_text"] = gen_texts
    df["orig_label"] = orig_labels
    df["cf_label"] = cf_labels

    out_path = "results_raw.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDone. Saved to {out_path}")


if __name__ == "__main__":
    main()

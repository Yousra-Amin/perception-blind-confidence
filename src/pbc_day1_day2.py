"""
Perception-blind confidence (498R) -- Day 1 + Day 2 starter script.
Standalone: no RoSE code. Model: Qwen2-VL-7B-Instruct (4-bit). Data: ChartQA test subset.

Usage (Colab/Kaggle T4):
  pip install -q transformers accelerate bitsandbytes datasets scikit-learn matplotlib pandas
  python pbc_day1_day2.py greedy  --n 500     # Day 1: answers + token log-probs
  python pbc_day1_day2.py sample  --n 500     # Day 2: 5 sampled answers (agreement)
  python pbc_day1_day2.py verbal  --n 500     # Day 2: verbalized confidence
  python pbc_day1_day2.py analyze             # Day 2: ECE, AUROC, risk-coverage

Every stage appends to a .jsonl file and skips finished ids, so you can resume
after a Colab/Kaggle disconnect.
"""
import argparse, json, math, os, random, re
from pathlib import Path

OUT = Path(os.environ.get("PBC_RESULTS", "results")); OUT.mkdir(parents=True, exist_ok=True)
# Tip: set PBC_RESULTS to a Google Drive / persistent folder so a disconnect never loses data.
MODEL_ID = "Qwen/Qwen2-VL-7B-Instruct"
SEED = 0

# ---------------------------------------------------------------- scoring
def to_float(s):
    s = str(s).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None

def relaxed_correct(pred, golds):
    """ChartQA relaxed accuracy: numeric within 5%, otherwise exact (case-insens.)."""
    for g in golds:
        pf, gf = to_float(pred), to_float(g)
        if pf is not None and gf is not None:
            ok = (pf == 0) if gf == 0 else abs(pf - gf) / abs(gf) <= 0.05
        else:
            ok = str(pred).strip().lower() == str(g).strip().lower()
        if ok:
            return True
    return False

def norm(s):
    f = to_float(s)
    return f"{f:.6g}" if f is not None else str(s).strip().lower()

# ---------------------------------------------------------------- data/model
def load_subset(n):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceM4/ChartQA", split="test")
    idx = list(range(len(ds)))
    random.Random(SEED).shuffle(idx)   # fixed order: --n 20 is always a prefix of --n 500
    return ds, idx[:n]

def load_model():
    import torch
    from transformers import (Qwen2VLForConditionalGeneration, AutoProcessor,
                              BitsAndBytesConfig)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.float16)  # T4: no bf16
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID, quantization_config=bnb, torch_dtype=torch.float16, device_map="auto")
    # lower max_pixels if you hit out-of-memory
    proc = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256*28*28, max_pixels=768*28*28)
    model.eval()
    return model, proc

def build_inputs(proc, image, text):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return proc(text=[prompt], images=[image.convert("RGB")], return_tensors="pt").to("cuda")

def done_ids(path):
    if not path.exists():
        return set()
    return {json.loads(l)["id"] for l in open(path)}

def summarize(path):
    recs = [json.loads(l) for l in open(path)]
    msg = f"\nDONE: {path.name} now has {len(recs)} records"
    if recs and "correct" in recs[0]:
        msg += f", accuracy={sum(r['correct'] for r in recs) / len(recs):.3f}"
    print(msg, flush=True)

def stop_ids(proc):
    tok = proc.tokenizer
    ids = {tok.eos_token_id}
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None:
        ids.add(im_end)
    return ids

# ---------------------------------------------------------------- Day 1
def run_greedy(n):
    import torch
    ds, idx = load_subset(n)
    model, proc = load_model()
    stops = stop_ids(proc)
    path = OUT / "greedy.jsonl"; done = done_ids(path)
    with open(path, "a") as f:
        for k, i in enumerate(idx):
            if i in done:
                continue
            ex = ds[i]
            q, golds = ex["query"], ex["label"]
            inp = build_inputs(proc, ex["image"], q + "\nAnswer the question using a single word or phrase.")
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=32, do_sample=False,
                                     repetition_penalty=1.0,  # keep probabilities clean
                                     return_dict_in_generate=True, output_logits=True)
            gen = out.sequences[0, inp["input_ids"].shape[1]:].tolist()
            lps, toks = [], []
            for t, tok in enumerate(gen):
                if tok in stops:
                    break
                lp = torch.log_softmax(out.logits[t][0].float(), -1)[tok].item()
                lps.append(lp); toks.append(tok)
            ans = proc.tokenizer.decode(toks, skip_special_tokens=True).strip()
            rec = dict(id=i, question=q, golds=golds, answer=ans,
                       correct=relaxed_correct(ans, golds),
                       human_or_machine=ex.get("human_or_machine"),
                       token_logprobs=lps)
            f.write(json.dumps(rec) + "\n"); f.flush()
            if k % 5 == 0:
                print(f"{k}/{len(idx)}  last: {ans!r} correct={rec['correct']}")
    summarize(path)

# ---------------------------------------------------------------- Day 2
def run_sample(n, k_samples=5):
    import torch
    ds, idx = load_subset(n)
    greedy = {json.loads(l)["id"]: json.loads(l) for l in open(OUT / "greedy.jsonl")}
    model, proc = load_model()
    path = OUT / "sample.jsonl"; done = done_ids(path)
    with open(path, "a") as f:
        for i in idx:
            if i in done or i not in greedy:
                continue
            ex = ds[i]
            inp = build_inputs(proc, ex["image"], ex["query"] + "\nAnswer the question using a single word or phrase.")
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=32, do_sample=True, temperature=0.7,
                                     top_p=1.0, top_k=0, repetition_penalty=1.0,
                                     num_return_sequences=k_samples)
            outs = proc.tokenizer.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                               skip_special_tokens=True)
            outs = [o.strip() for o in outs]
            g = norm(greedy[i]["answer"])
            agree = sum(norm(o) == g for o in outs) / k_samples
            f.write(json.dumps(dict(id=i, samples=outs, agreement=agree)) + "\n"); f.flush()
            print(".", end="", flush=True)
    summarize(path)

def run_verbal(n):
    import torch
    ds, idx = load_subset(n)
    greedy = {json.loads(l)["id"]: json.loads(l) for l in open(OUT / "greedy.jsonl")}
    model, proc = load_model()
    path = OUT / "verbal.jsonl"; done = done_ids(path)
    with open(path, "a") as f:
        for i in idx:
            if i in done or i not in greedy:
                continue
            ex = ds[i]
            text = (f"Question: {ex['query']}\nProposed answer: {greedy[i]['answer']}\n"
                    "How confident are you that the proposed answer is correct? "
                    "Reply with only a number from 0 to 100.")
            inp = build_inputs(proc, ex["image"], text)
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=6, do_sample=False, repetition_penalty=1.0)
            raw = proc.tokenizer.decode(out[0, inp["input_ids"].shape[1]:], skip_special_tokens=True)
            m = re.search(r"\d+(\.\d+)?", raw)
            conf = min(float(m.group()), 100.0) / 100 if m else None
            f.write(json.dumps(dict(id=i, raw=raw, verbal=conf)) + "\n"); f.flush()
            print(".", end="", flush=True)
    summarize(path)

# ---------------------------------------------------------------- metrics
def ece(conf, correct, bins=15):
    import numpy as np
    conf, correct = np.asarray(conf, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total

def risk_coverage(conf, correct):
    import numpy as np
    order = np.argsort(-np.asarray(conf, float), kind="stable")
    c = np.asarray(correct, float)[order]
    n = len(c)
    coverage = np.arange(1, n + 1) / n
    risk = 1 - np.cumsum(c) / np.arange(1, n + 1)
    return coverage, risk, risk.mean()  # mean risk over coverage = AURC

def analyze():
    import numpy as np, pandas as pd, matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score
    g = pd.DataFrame([json.loads(l) for l in open(OUT / "greedy.jsonl")])
    lp = g["token_logprobs"].apply(lambda x: x if len(x) else [-20.0])
    g["seq_prob"] = lp.apply(lambda x: math.exp(sum(x)))
    g["min_prob"] = lp.apply(lambda x: math.exp(min(x)))
    g["mean_prob"] = lp.apply(lambda x: math.exp(sum(x) / len(x)))
    for name, col in [("sample", "agreement"), ("verbal", "verbal")]:
        p = OUT / f"{name}.jsonl"
        if p.exists():
            d = pd.DataFrame([json.loads(l) for l in open(p)])[["id", col]]
            g = g.merge(d, on="id", how="left")
    if "verbal" in g:
        print(f"verbal parse failures: {g['verbal'].isna().sum()} / {len(g)} (filled with 0.5)")
        g["verbal"] = g["verbal"].fillna(0.5)
    scores = [c for c in ["seq_prob", "min_prob", "mean_prob", "agreement", "verbal"] if c in g]
    y = g["correct"].astype(int).values
    print(f"\nN={len(g)}  accuracy={y.mean():.3f}")
    g["split"] = g["human_or_machine"].map({0: "human", 1: "machine"}).fillna("unknown")
    print(g.groupby("split")["correct"].agg(["mean", "count"]).round(3).to_string())
    rows, fig, ax = [], plt.figure(figsize=(6, 4)), plt.gca()
    for s in scores:
        cov, risk, aurc = risk_coverage(g[s], y)
        ax.plot(cov, risk, label=s)
        auc = roc_auc_score(y, g[s]) if len(set(y)) > 1 else float("nan")
        rows.append(dict(score=s, ECE=ece(g[s], y), AUROC=auc, AURC=aurc))
    ax.set_xlabel("coverage"); ax.set_ylabel("risk (error rate)"); ax.legend()
    fig.tight_layout(); fig.savefig(OUT / "risk_coverage.png", dpi=150)
    table = pd.DataFrame(rows).round(3); print(table.to_string(index=False))
    table.to_csv(OUT / "baseline_table.csv", index=False)
    # reliability diagram for the sequence probability
    edges = np.linspace(0, 1, 11); mids, accs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (g["seq_prob"] > lo) & (g["seq_prob"] <= hi)
        if m.any():
            mids.append(g.loc[m, "seq_prob"].mean()); accs.append(y[m.values].mean())
    plt.figure(figsize=(4, 4)); plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.plot(mids, accs, "o-"); plt.xlabel("confidence (seq_prob)"); plt.ylabel("accuracy")
    plt.tight_layout(); plt.savefig(OUT / "reliability_seq_prob.png", dpi=150)
    g.to_csv(OUT / "merged.csv", index=False)
    print("\nSaved: baseline_table.csv, risk_coverage.png, reliability_seq_prob.png, merged.csv")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["greedy", "sample", "verbal", "analyze"])
    ap.add_argument("--n", type=int, default=500)
    a = ap.parse_args()
    {"greedy": lambda: run_greedy(a.n), "sample": lambda: run_sample(a.n),
     "verbal": lambda: run_verbal(a.n), "analyze": analyze}[a.stage]()

# Semurg' Delta V2

> **Lightweight neural architecture for Uzbek NLP**  
> Original architecture — not a BERT fine-tune

**Author:** Nurislom Abdumuminov | 2025–2026  
**Email:** exampil180@gmail.com  
**GitHub:** [nurislom-abdimuminov](https://github.com/nurislom-abdimuminov)

---

## What is Semurg' Delta?

Semurg' Delta V2 is an **original lightweight neural architecture** designed specifically for the Uzbek language. Large models like BERT require 110M parameters and cannot run on mobile devices. Semurg' Delta V2 achieves competitive results with only **6.67M parameters**.

**Core problem solved:** No high-quality lightweight AI model existed for Uzbek. BERT does not understand Uzbek morphology. We solved this with two innovations:

- **Delta Gate** — filters each word feature using softplus gating
- **Morfema Tokenizer** — splits Uzbek words into root + suffixes
- **Feature Density Pooling** — gives more weight to important tokens

---

## Benchmark Results

Tested on Kaggle GPU T4 x2 | 50,000 Uzum Market reviews | 5 epochs

| Model | Accuracy | Parameters |
|-------|----------|------------|
| Baseline (BERT tokenizer) | 90.61% | 30.6M |
| Semurg' Delta V1 (Morfema) | 90.95% | 6.67M |
| **Semurg' Delta V2 (Morfema + Density)** | **91.86%** | **6.67M** |
| Old record (8M, BERT tok) | 89.89% | 8M |
| BERT-base | 91.09% | 110M |

**Key results:**
- V2 is **16x lighter** than BERT, but **+0.77% higher** accuracy
- Feature Density Pooling: **+0.91%** over V1
- Over old record: **+1.97%**

---

## Architecture

### 1. Delta Gate (V1 core)

```python
delta = softplus(W * x)   # softplus = log(1 + exp(x))
gate  = exp(-delta)        # range [0, 1]
x_out = x * gate           # unnecessary features filtered
```

Identity initialization ensures stability: W starts as identity matrix. At training start, the model passes data unchanged.

### 2. Feature Density Pooling (V2 innovation)

```python
d   = gate.mean(dim=-1, keepdim=True)  # Feature Density
w   = d * mask                          # Density-weighted mask
out = (x * w).sum(1) / w.sum(1)        # Weighted average
```

Tokens with higher gate density carry more useful information — they receive more weight in pooling. This gave **+0.77%** over simple average pooling.

### 3. Morfema Tokenizer

```python
'kitobdan'      -> ['kitob', 'dan']
'o\'qituvchimiz' -> ['o\'qituvchi', 'imiz']
```

BERT tokenizer cannot recognize Uzbek suffixes as separate tokens. Morfema tokenizer correctly splits roots and suffixes, allowing the model to better understand Uzbek grammar.

---

## Model Architecture

```python
class SemurgDeltaV2(nn.Module):
    def __init__(self, vocab_size, dim=256, num_classes=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.W   = nn.Linear(dim, dim)
        self.cls = nn.Linear(dim, num_classes)
        nn.init.eye_(self.W.weight)
        nn.init.zeros_(self.W.bias)

    def forward(self, ids, mask):
        x = self.emb(ids)
        # Delta gate
        g = torch.exp(-torch.log(1 + torch.exp(self.W(x))))
        # Feature Density pooling
        d = g.mean(dim=-1, keepdim=True)
        x = x * g
        m = mask.unsqueeze(-1).float()
        w = d * m
        return self.cls((x * w).sum(1) / w.sum(1).clamp(min=1))
```

---

## Comparison with Mamba

Inspired by Mamba (Gu & Dao, 2023), but fundamentally different:

| Feature | Mamba | Semurg' Delta |
|---------|-------|---------------|
| Delta purpose | Controls time-step | Controls feature importance |
| Complexity | O(n) recurrent | O(n) no state |
| Recurrent state | Yes | No |
| Target task | Time series | Text classification |

---

## Roadmap

| Stage | Goal |
|-------|------|
| Now (V2) | Sentiment classification — 91.86% |
| V3 | Multi-class: topic, language level |
| V4 | Full encoder — NER, POS tagging |
| V5 | Encoder-Decoder — full Uzbek NLP suite |
| Future | Uzbek AI foundation model |

**Strategic goal:** First original neural architecture for the Uzbek language — lightweight enough for mobile, accurate enough to replace BERT in Uzbekistan's IT ecosystem.

---

## Dataset

- **Source:** `risqaliyevds/uzbek-sentiment-analysis` (Hugging Face)
- **Size:** 50,000 Uzum Market product reviews
- **Task:** Binary sentiment classification (positive / negative)

---

## References

Gu, A., & Dao, T. (2023). *Mamba: Linear-Time Sequence Modeling with Selective State Spaces.* arXiv:2312.00752

---

## License

MIT License — see [LICENSE](LICENSE)

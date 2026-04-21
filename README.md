# Semurg' Delta V2

> Lightweight neural architecture for Uzbek NLP

**Author:** Nurislom Abdumuminov | 2025-2026

---

## What is Semurg' Delta?

Semurg' Delta is an original lightweight neural architecture
designed specifically for the Uzbek language.

- **Parameters:** 6.67M (vs BERT 110M — 16x smaller)
- **Task:** Uzbek sentiment analysis, text classification
- **Innovation:** Feature Density pooling + softplus gating

## Results

| Model | Accuracy | Parameters |
|-------|----------|------------|
| BERT-base | 91.09% | 110M |
|Semurg' Delta V2 | 91.61% | 8M  |

## Architecture

- Morfema tokenizer (Uzbek morphology)
- Feature Density pooling
- Softplus gating mechanism
- No attention — deterministic formula

## Contact

- GitHub: nurislom-abdimuminov
- Email: exampil180@gmail.com

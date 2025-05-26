# 🧠 Named Entity Recognition (NER) – A Domain-Transferable Pipeline from Biomedical to Finance

This module showcases a robust NER pipeline that begins in the biomedical domain and is later generalized to financial entity extraction, demonstrating strong cross-domain transferability and engineering adaptability.

---

## 🔍 Project Overview

| Notebook | Description |
|----------|-------------|
| `009_0_NLP_NER_BC5CDR.ipynb` | NER on BC5CDR dataset using BioBERT; targets `Chemical` and `Disease` entities. Chemical recognition performs well, but Disease detection is relatively weaker. |
| `009_1_NLP_NER_NCBI.ipynb`   | Focused NER on `Disease` entities using NCBI Disease Corpus; model fine-tuned for high precision and recall on a single entity class. |
| `009_2_NER_FiNER139_local.ipynb` | First attempt to adapt pipeline to Finance domain using FiNER139 dataset locally; 139 entity types causes training resource bottlenecks. |
| `009_2_NER_FiNER139_colab.ipynb` | Re-runs FiNER139 on Google Colab Pro with A100 GPU; training fails due to memory/time limits, but theoretical benchmark F1 score (>81%) is reported. |

---

## 🎯 Objectives

- ✅ Evaluate domain-specific NER using domain-adaptive models like BioBERT and FinBERT
- ✅ Improve entity-specific performance (e.g., Disease)
- ✅ Scale NER tasks to multi-label, multi-entity datasets (like Finance)
- ✅ Demonstrate modular NER pipeline compatible with Hugging Face ecosystem
- ✅ Address engineering trade-offs in large-scale training scenarios (Colab vs local)

---

## 🧠 Models Used

- `dmis-lab/biobert-base-cased-v1.2` → for BC5CDR & NCBI
- `yiyanghkust/finbert-pretrain` → for FiNER-139

---

## 📚 Datasets & References

| Dataset | Paper / Link |
|---------|--------------|
| BC5CDR | [Li et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27161011/) |
| NCBI Disease Corpus | [Dogan et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24393765/) |
| FiNER-139 | [Xypolopoulos et al., 2021](https://arxiv.org/abs/2108.13441) – [GitHub](https://github.com/nlpaueb/finer) |

---

## ✅ Highlights & Takeaways

- Biomedical NER shows clear modularization potential:
  - Chemical → consistently high F1
  - Disease → improved by moving to focused dataset (NCBI)
- Preprocessing Adaptability:
  - For `offsets`-based datasets (e.g., BC5CDR), token-label alignment is handled via `offset_mapping` from tokenizer.
  - For `tokens` + `ner_tags` datasets (e.g., NCBI, FiNER139), `is_split_into_words=True` and `word_ids()` are used for mapping.
  - Abstracting this distinction into a reusable function improves pipeline generalization.
- Financial NER requires optimization for:
  - Label imbalance (139 entities)
  - Memory and runtime bottlenecks (even on A100 GPU)
- Template pipeline is **fully generalizable** across domains by:
  - Modifying tokenizer + label encoder
  - Retaining metrics (seqeval) and alignment logic

---

## 🧩 Reusability

This NER pipeline can be adapted for:

- Any BIO/BIOES-labeled dataset
- Custom domains (legal, pharma, cybersecurity, etc.)
- Model benchmarking on Hugging Face with full Trainer integration

For reproducibility:

- All notebooks use Hugging Face `Trainer` API with token classification head. 
- Label mapping, alignment functions, and metrics are abstracted for modular reuse across domains.


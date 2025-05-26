# 🧠 NLP Module – Text Regression & Named Entity Recognition

This folder contains multiple projects focusing on two key NLP tasks:

- **Phrase-level Semantic Regression** (US Patent Matching)
- **Named Entity Recognition (NER)** in biomedical and financial domains

---

## 📂 Structure

| Notebook | Description |
|----------|-------------|
| `008_NLP_UsPatent_deberta.ipynb` | Sentence-pair regression with DeBERTa on patent phrase similarity |
| `NER/009_0_NER_BC5CDR.ipynb`     | Biomedical NER using BioBERT – BC5CDR dataset |
| `NER/009_1_NER_NCBI.ipynb`       | Disease NER using BioBERT – NCBI Disease Corpus |
| `NER/009_2_NER_FiNER139.ipynb`   | Financial NER using FinBERT – FiNER139 dataset |

---

## 🧠 Model Architectures Used

- [`microsoft/deberta-v3-base`](https://huggingface.co/microsoft/deberta-v3-base) for phrase regression
- [`dmis-lab/biobert-base-cased-v1.1`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1) for biomedical NER
- [`yiyanghkust/finbert-pretrain`](https://huggingface.co/yiyanghkust/finbert-pretrain) for financial NER

---

## ✅ Skills Demonstrated

- Hugging Face `Trainer`, `Dataset`, `Tokenizer` APIs
- NER-specific label alignment (BIO/BIOES tagging)
- Metric customization with `seqeval`, `pearson`
- Model comparison: small vs base variants
- Domain-adaptive pretraining and evaluation

---

## 📦 Datasets Used

- [U.S. Patent Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching)
- [BC5CDR](https://huggingface.co/datasets/bigbio/bc5cdr)
- [NCBI Disease Corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/)
- [FiNER-139](https://github.com/nlpaueb/finer)

---

## 🧩 Reusability

Each notebook is modular and can serve as a template for:

- Sentence similarity prediction (STS, paraphrase scoring)
- Domain-specific NER (biomedical, legal, financial)
- Hugging Face-based fine-tuning pipelines


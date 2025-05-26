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
- [`dmis-lab/biobert-base-cased-v1.2`](https://huggingface.co/dmis-lab/biobert-base-cased-v1.1) for biomedical NER
- [`yiyanghkust/finbert-pretrain`](https://huggingface.co/yiyanghkust/finbert-pretrain) for financial NER

---

## ✅ Skills Demonstrated

- Hugging Face `Trainer`, `Dataset`, `Tokenizer` APIs
- NER-specific label alignment (BIO/BIOES tagging)
- Metric customization with `seqeval`, `pearson`
- Model comparison: small vs base variants
- Domain-adaptive pretraining and evaluation

---

## 📚 References

| Notebook | Source / Paper |
|----------|----------------|
| `008_NLP_UsPatent_deberta.ipynb` | Kaggle Competition: [U.S. Patent Phrase to Phrase Matching](https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching) <br> Model: [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) |
| `009_0_NER_BC5CDR.ipynb` | Dataset: [BC5CDR Corpus](https://www.biocreative.org/tasks/biocreative-v/track-3-cdr/) <br> Paper: [Li et al., 2016](https://pubmed.ncbi.nlm.nih.gov/27161011/) |
| `009_1_NER_NCBI.ipynb` | Dataset: [NCBI Disease Corpus](https://www.ncbi.nlm.nih.gov/CBBresearch/Dogan/DISEASE/) <br> Paper: [Dogan et al., 2014](https://pubmed.ncbi.nlm.nih.gov/24393765/) |
| `009_2_NER_FiNER139.ipynb` | Dataset: [FiNER-139](https://github.com/nlpaueb/finer) <br> Paper: [Xypolopoulos et al., 2021](https://arxiv.org/abs/2108.13441) |

---

## 🧩 Reusability

Each notebook is modular and can serve as a template for:

- Sentence similarity prediction (STS, paraphrase scoring)
- Domain-specific NER (biomedical, legal, financial)
- Hugging Face-based fine-tuning pipelines


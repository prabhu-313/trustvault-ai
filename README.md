# 🔐 TrustVault AI
### Privacy-First Federated AI Assistant

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E?style=flat-square&logo=huggingface&logoColor=black)
![Opacus](https://img.shields.io/badge/Opacus-DP%20Library-0064BD?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)
![Status](https://img.shields.io/badge/Status-Complete-22C55E?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Google%20Colab-F9AB00?style=flat-square&logo=googlecolab&logoColor=white)

> A privacy-preserving NLP assistant that runs entirely on-device — no data ever sent to external servers. Combines Federated Learning, Differential Privacy, Homomorphic Encryption simulation, and RAG into a single self-contained Jupyter notebook with a live Streamlit demo.

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Stats](#-key-stats)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [NLP Models](#-nlp-models)
- [Privacy Stack](#-privacy-stack)
- [RAG Pipeline](#-rag-pipeline)
- [Streamlit App](#-streamlit-app)
- [Project Structure](#-project-structure)
- [Setup & Usage](#-setup--usage)
- [Results & Performance](#-results--performance)
- [Known Limitations](#-known-limitations)
- [Future Work](#-future-work)
- [Author](#-author)
- [References](#-references)

---

## 🔭 Overview

Large language models like ChatGPT deliver exceptional capability — but at the cost of sending every user query to a remote cloud server. TrustVault AI takes the opposite approach: **all models run locally**, no data leaves the machine during inference, and a federated training simulation demonstrates how a distributed model can be updated without sharing raw data.

Built as a self-contained research prototype, TrustVault explores the intersection of privacy-preserving machine learning and practical NLP tooling. Three core research questions drive the project:

1. Can state-of-the-art NLP tasks (generation, summarization, Q&A) be delivered at acceptable quality from lightweight models that fit in a free Colab session?
2. Can Differential Privacy via Opacus be practically applied to fine-tuning a transformer in a federated simulation, given known compatibility issues with DistilBERT's tied weights?
3. Can a RAG pipeline be built without any external API calls, using only local embeddings and FAISS?

**The answer to all three is yes** — with non-trivial engineering required for the Opacus/DistilBERT integration.

---

## 📊 Key Stats

| Metric | Value |
|---|---|
| FL Clients | 3 (simulated) |
| NLP Tasks | 5 |
| Privacy Budget (ε) | ≤ 8.0 |
| Local Inference | 100% — zero external API calls |
| Deployment | Google Colab + Ngrok |
| Notebook | Single `.ipynb` — runs end-to-end |

---

## ✨ Features

- 🧠 **Three local NLP models** — text generation, summarization, and extractive Q&A, all running on-device
- 🌐 **Federated Learning simulation** — 3 clients train independently, server aggregates via FedAvg
- 🔐 **Differential Privacy** — per-sample gradient clipping and Gaussian noise via Opacus, with DistilBERT tied-weight fix
- 🔒 **Homomorphic Encryption simulation** — encrypt → aggregate → decrypt pipeline demonstrating the HE workflow
- 🔍 **RAG pipeline** — FAISS semantic search + local sentence embeddings + DistilBERT reader, no external API
- 🖥️ **Streamlit web UI** — login authentication, privacy mode toggle, 5 task pages, live Ngrok deployment
- ⚙️ **Single configuration cell** — all tokens and credentials set in one place using Colab Secrets

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Streamlit Web UI                        │
│        Login  ·  Privacy Toggle  ·  Task Selector          │
└──────────────────────┬──────────────────────────────────────┘
                       │
       ┌───────────────┼───────────────┐
       ▼               ▼               ▼
┌─────────────┐ ┌─────────────┐ ┌──────────────┐
│ Text Gen    │ │Summarization│ │  Q&A / RAG   │
│ GPT-Neo 125M│ │  T5-Small   │ │  DistilBERT  │
└─────────────┘ └─────────────┘ └──────┬───────┘
                                        │
                                 ┌──────▼───────┐
                                 │ FAISS Index  │
                                 │ (RAG only)   │
                                 └──────────────┘

┌────────────────────────────────────────────────┐
│              Privacy Layer                     │
│  FL Simulation  (FedAvg, 3 clients)           │
│  Differential Privacy  (Opacus DP-SGD)        │
│  HE Simulation  (XOR mock → TenSEAL future)   │
└────────────────────────────────────────────────┘
```

---

## 🧠 NLP Models

All models are loaded from HuggingFace Transformers and run fully locally — no inference API calls.

| Task | Model | Parameters | Size | Notes |
|---|---|---|---|---|
| Text Generation | GPT-Neo 125M (EleutherAI) | 125M | ~500MB | Autoregressive; CPU-feasible |
| Summarization | T5-Small (Google) | 60M | ~240MB | Encoder-decoder seq2seq |
| Question Answering | DistilBERT (SQuAD fine-tune) | 66M | ~260MB | Extractive QA; 6× smaller than BERT |
| RAG Embeddings | all-MiniLM-L6-v2 | 22M | ~90MB | 384-dim dense vectors for semantic search |

---

## 🔐 Privacy Stack

### Federated Learning (FedAvg)

Each of 3 clients trains a local DistilBERT sequence classifier on their data shard. Only weight tensors — never raw data — are sent to the server, which aggregates them via Federated Averaging:

```
Client 1 ──┐
            ├──► Server (FedAvg) ──► Global Model
Client 2 ──┤         ↑
            │   Weight updates only
Client 3 ──┘   (no raw data)
```

### Differential Privacy (Opacus)

Opacus wraps the model in a `GradSampleModule` that computes per-sample gradients, clips them by `max_grad_norm`, and adds calibrated Gaussian noise before the optimizer step.

**The DistilBERT / Opacus Compatibility Fix:**
DistilBERT ties its word embedding weights to the `vocab_projector` output layer. Opacus's per-sample gradient computation accumulates gradients from both usages, causing mismatched tensor shapes:

```
RuntimeError: stack expects each tensor to be equal size,
              got [5] at entry 0 and [1] at entry 1
```

**Fix:** Freeze all embedding and vocab parameters before Opacus wraps the model, removing the tied weights from per-sample gradient computation while keeping all attention layers and the classification head trainable.

| DP Parameter | Value |
|---|---|
| `noise_multiplier` | 1.0 |
| `max_grad_norm` | 1.0 |
| `delta` | 1e-5 |
| Privacy budget (ε) | ≤ 8.0 |

### Homomorphic Encryption (Simulation)

Demonstrates the encrypt → aggregate → decrypt workflow using a mock XOR cipher on model weight tensors. In a production deployment, the `he_encrypt` / `he_decrypt` functions would be replaced with TenSEAL's CKKS scheme for real floating-point homomorphic encryption.

---

## 🔍 RAG Pipeline

```
User Query
    │
    ▼
all-MiniLM-L6-v2 (embed query → 384-dim vector)
    │
    ▼
FAISS IndexFlatL2 (L2 nearest-neighbour search)
    │
    ▼
Top-k passages retrieved from knowledge base
    │
    ▼
DistilBERT extractive QA (answer span + confidence)
    │
    ▼
Answer + retrieved sources → UI
```

- Knowledge base: 10 passages on privacy, federated learning, and TrustVault concepts
- All steps run in-process — zero network calls
- FAISS search on 10 documents: sub-100ms

---

## 🖥️ Streamlit App

Five task pages accessible from a sidebar radio selector:

| Page | Description |
|---|---|
| 📝 Text Generation | Prompt input, max-token slider, GPT-Neo generation |
| 📄 Summarization | Paste text or upload `.txt`, T5-Small summarization |
| ❓ Q&A | Context paragraph + question, DistilBERT answer + confidence |
| 🔍 RAG Q&A | Free-form question, FAISS retrieval, expandable source passages |
| 📊 Privacy Dashboard | Live FL/DP metrics, privacy technique status table |

The app is written to disk at runtime from the notebook and launched as a subprocess. `@st.cache_resource` ensures all models are loaded once per session.

---

## 📁 Project Structure

```
trustvault-ai/
│
├── README.md
├── requirements.txt
├── .gitignore
├── LICENSE
│
├── TrustVault_AI.ipynb        # Full pipeline — all 8 sections, self-contained
│
└── docs/
    └── TrustVault_AI_Report.docx   # Full project report with literature review,
                                    # architecture, implementation, and results
```

---

## ⚙️ Setup & Usage

### Requirements

- Google Colab (free tier works — T4 GPU recommended, CPU fallback supported)
- Ngrok account (free) — get your token at [dashboard.ngrok.com](https://dashboard.ngrok.com/get-started/your-authtoken)

### 1. Open in Google Colab

Upload `TrustVault_AI.ipynb` to [colab.research.google.com](https://colab.research.google.com) or open directly from GitHub.

### 2. Set your Ngrok token via Colab Secrets

In Colab: **🔑 Secrets (left sidebar) → Add new secret**

```
Name  : NGROK_TOKEN
Value : your_ngrok_token_here
```

> This keeps your token out of the notebook entirely — never hardcoded.

### 3. Run all cells in order

```
Runtime → Run all  (Ctrl+F9)
```

Sections 0–7 build the full pipeline (~10–15 minutes on CPU, ~5 minutes on T4 GPU).

### 4. Launch the app (Section 8)

Run the final cell. A public Ngrok URL will be printed:

```
🚀 TrustVault AI is live!
🔗 Public URL : https://xxxx-xx-xx-xxx.ngrok-free.app
👤 Username   : admin
🔑 Password   : trustvault
```

Open the URL in any browser.

---

## 📈 Results & Performance

### Test Cases

| ID | Test | Result |
|---|---|---|
| T01 | Login — valid credentials | ✅ Redirects to dashboard |
| T02 | Login — invalid credentials | ✅ Error shown, no access |
| T03 | Text Generation (100 tokens) | ✅ Coherent GPT-Neo output |
| T04 | Summarization (500-word input) | ✅ Condensed T5-Small summary |
| T05 | Q&A with context paragraph | ✅ DistilBERT answer + confidence |
| T06 | RAG Q&A | ✅ Retrieved passages + answer span |
| T07 | FL + DP (3 clients, noise=1.0) | ✅ All clients complete, ε ≤ 8.0 |
| T08 | Privacy Mode toggle | ✅ Banner updates correctly |
| T09 | Ngrok public URL | ✅ Public HTTPS URL printed |

### Inference Times

| Task | CPU | T4 GPU |
|---|---|---|
| Text Generation (100 tokens) | 8–12 seconds | 2–3 seconds |
| Summarization (500-word input) | 3–5 seconds | ~1 second |
| Q&A (DistilBERT) | 1–2 seconds | <1 second |
| RAG retrieval (FAISS, 10 docs) | <100ms | <100ms |
| FL + DP training (3 clients, 1 epoch) | 4–7 minutes | ~90 seconds |

---

## ⚠️ Known Limitations

- GPT-Neo 125M is a small model — output is coherent but not always factually reliable
- The FL dataset is synthetic (12 samples) — real federated learning requires real distributed clients
- The HE pipeline is a mock cipher — actual CKKS encryption would be ~100× slower in this environment
- Colab sessions time out after ~12 hours — the notebook must be re-run each session

---

## 🔮 Future Work

**Short Term**
- Replace mock HE with TenSEAL CKKS for real encrypted gradient aggregation
- Add persistent memory via local SQLite or ChromaDB vector store
- Implement JWT-based authentication to replace session state login
- Add 8-bit quantization via `bitsandbytes` to reduce GPU memory footprint

**Long Term**
- Replace single-node FL simulation with real multi-client deployment using PySyft or Flower
- Extend model suite with a local code generation model (CodeBERT or StarCoder-1B)
- Build a Docker image for one-command local deployment without Colab dependency
- Implement multilingual support via mBART or mT5
- Benchmark privacy budget vs. model accuracy trade-offs across `noise_multiplier` settings

---

## 👤 Author

**Prabhupada Samantaray**
B.Tech CSE, KIIT University (2022–2026)
[GitHub](https://github.com/prabhu-313) · [LinkedIn](https://www.linkedin.com/in/prabhupada-samantaray-13apr2002/) · [Email](mailto:psray313@gmail.com)

---

## 📚 References

- McMahan et al. (2017) — [Communication-Efficient Learning of Deep Networks from Decentralized Data](https://arxiv.org/abs/1602.05629)
- Dwork, C. (2006) — [Differential Privacy](https://link.springer.com/chapter/10.1007/11787006_1)
- Vaswani et al. (2017) — [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- Yousefpour et al. (2021) — [Opacus: User-Friendly Differential Privacy Library in PyTorch](https://arxiv.org/abs/2109.12298)
- Gentry, C. (2009) — Fully Homomorphic Encryption Using Ideal Lattices. STOC.
- Lewis et al. (2020) — [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)

---

## 📄 License

This project is licensed under the [MIT License](./LICENSE).

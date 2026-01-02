# Hope Speech Detection (English)

This repository presents a concise and reproducible pipeline for **Hope Speech Detection** in English social media text using **transformer-based sentence embeddings** and a **CNN classifier**, including an embedding-level fusion strategy.

---

## Overview

Hope speech promotes encouragement, inclusion, and positive discourse. This project builds a binary classifier to distinguish **Hope Speech** from **Non-Hope Speech** using pretrained transformer embeddings followed by a lightweight CNN.

---

## Models

| Model           | Embedding Source     | Dim  |
| --------------- | -------------------- | ---- |
| MPNet-CNN       | all-mpnet-base-v2    | 768  |
| Sentence-T5-CNN | sentence-t5-base     | 768  |
| RoBERTa-CNN     | roberta-base (CLS)   | 768  |
| Fusion-CNN      | MPNet + T5 + RoBERTa | 2304 |

---

## Methodology

**Preprocessing**

* Lowercasing
* URL and username removal
* Whitespace normalization

**Embeddings**

* MPNet and Sentence-T5 via `sentence-transformers`
* RoBERTa CLS embeddings via HuggingFace Transformers
* L2 normalization for RoBERTa

**Classifier**

* Fully connected projection
* Three Conv1D layers with ReLU and MaxPooling
* Dropout regularization
* Softmax output (binary classification)

The CNN is embedding-agnostic and supports both single and fused embeddings.

---

## Dataset

Hope Speech English dataset with Train/Dev/Test splits. The dataset is class-imbalanced.

Labels:

* 1: Hope Speech
* 0: Non-Hope Speech

---

## Training

* Optimizer: Adam
* Loss: CrossEntropyLoss
* Epochs: 10
* Learning Rate: 1e-4
* Batch Size: 32
* Model selection based on Dev Macro-F1

---

## Results

**Macro-F1 Scores**

| Model           | Dev        | Test       |
| --------------- | ---------- | ---------- |
| MPNet-CNN       | 0.7802     | 0.7818     |
| Sentence-T5-CNN | 0.7788     | 0.7619     |
| RoBERTa-CNN     | 0.7455     | 0.7436     |
| Fusion-CNN      | **0.7980** | **0.7856** |

Fusion-CNN achieves the best overall performance.

---

## Repository Structure

This repository is currently organized as a **single notebook-based project** (Kaggle-style). No strict folder structure is required.

Typical contents:

* Main Jupyter notebook containing preprocessing, embedding extraction, training, and evaluation
* Saved model checkpoints (`.pt` files)
* README.md

If you later modularize the code, you may optionally separate data, models, and notebooks, but this is **not required** for reproducing the results.

---

## Usage

Install dependencies:

```
pip install torch transformers sentence-transformers scikit-learn pandas numpy
```

Run the notebook/script to preprocess data, generate embeddings, train models, and evaluate results. Best checkpoints are saved automatically.



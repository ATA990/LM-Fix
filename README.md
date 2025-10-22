# LM‑Fix — Bit‑Flip Detection & Rapid Recovery for LLMs (ICCD 2025)

[![CI](https://img.shields.io/github/actions/workflow/status/YOUR_ORG_OR_USER/LM-Fix/ci.yml?branch=main)](https://github.com/YOUR_ORG_OR_USER/LM-Fix/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-Zenodo_TBD-blue)](https://doi.org/10.5281/zenodo.TBD)

**LM‑Fix** is a lightweight framework that detects **silent bit‑flips** in LLMs and **repairs** them **without retraining**. This repo hosts the reference implementation and scripts to reproduce core results from the ICCD 2025 paper.

> **Status:** v0.1.0 (release candidate) — 2025-10-22

---

## TL;DR (Why LM‑Fix?)

- **Detects faults fast:** Golden & layer‑wise hashing catch silent errors before sampling.
- **Pinpoints the fault:** Cache clearing → layer search → parameter‑level localization.
- **Repairs quickly:** Integer‑view weight editing + reference outputs; **>100× faster** than full model reload.
- **Low overhead:** ≈ **1%–7.7%** at **TVL=200** across tested models.

---

## Detection Accuracy (from the paper)

| Model / Scope              | Precision | TVL | Single‑bit detection | Multi‑bit detection | Runtime overhead | Notes |
|---                         |---        |---: |---:                   |---:                 |---:              |---|
| **Across tested models**   | various   | 200 | **>94%**             | **≈100%**           | **~1%–7.7%**     | Paper aggregate |
| LLaMA‑3.2‑3B               | FP8       | 200 | **>97%**             | **≈100%**           | **~1%–7.7%**     | 500K single‑bit flips evaluated |

*TVL = length of the fixed test vector (in tokens). Longer TVL generally increases detection accuracy with modest overhead.*

> If you want a per‑model, per‑TVL breakdown table, add exact rows from your evaluation logs here.

---

## Quick Start

```bash
pip install -r requirements.txt
# Example: generate a golden hash and run a quick detection check
python -m lmfix.scripts.generate_golden --model meta-llama/Llama-3.2-3B --tokens 200 --out artifacts/golden.json
python -m lmfix.scripts.detect --model meta-llama/Llama-3.2-3B --golden artifacts/golden.json
```

**Reproduce main steps** (adjust module paths to your repo layout):

```bash
# 1) Golden hash
python -m lmfix.scripts.generate_golden --model meta-llama/Llama-3.2-3B --tokens 200 --out artifacts/golden.json

# 2) Save layer outputs (constant-input & parameterized-input)
python -m lmfix.scripts.save_layer_hashes --model meta-llama/Llama-3.2-3B --mode ones  --out artifacts/layer_hashes.json
python -m lmfix.scripts.save_layer_hashes --model meta-llama/Llama-3.2-3B --mode param --out artifacts/layer_param_hashes.json

# 3) Hooks (input shape enforcement; saves outputs for reference)
python -m lmfix.scripts.step3_hooks --model meta-llama/Llama-3.2-3B --shape 1,1,<num_rows> --save artifacts/step3_outputs.pt

# 4) Rotate + repair (restores weights after)
python -m lmfix.scripts.step4_rotate_repair --model meta-llama/Llama-3.2-3B --rotate --restore --ref artifacts/step3_outputs.pt
```

---

## Glossary (Key Terms)

- **TVL (Test Vector Length):** Number of tokens in the fixed **test vector** used to generate the reference/golden outputs. Higher TVL → stronger detection (slightly more overhead).
- **Golden Hash:** A reference hash computed from a short, deterministic model run; used to flag global corruption.
- **Layer‑wise Hash:** Hash of each layer’s output (or selected tensors) to localize corruption precisely.
- **Hooked Tensor Auditing (HTA):** Capturing the last pre‑sampling tensor (or target layer tensor) and comparing it against references.
- **LOT (Layer Output Tensor):** The per‑layer output snapshot stored/compared during detection and recovery.
- **Integer View of Weights:** Reinterpreting weight tensors as fixed‑width integers (no bit changes) to avoid IEEE‑754 round‑off in recovery math, enabling **bit‑exact** restoration.
- **Cache Clearing:** Step that flushes system/GPU caches to remove transient corruptions.
- **Layer Search:** Comparing LOTs with references to identify corrupted layers.
- **Parameter‑level Localization:** Using column/row searches (with 90° rotations) to identify corrupted weight vectors.
- **Reference Output Storage:** A corpus of per‑layer reference outputs (integer‑view) used by the repair solver.

---

## Cite

```bibtex
@inproceedings{{lmfix_iccd_2025,
  title     = {{LM-Fix}: Lightweight Bit-Flip Detection and Rapid Recovery Framework for Language Models},
  booktitle = {{ICCD}},
  year      = {{2025}},
  author    = {{Add author list}},
}}
```
Also include the software citation from `CITATION.cff` (GitHub “Cite this repository”).

---

## Links

- Paper (ICCD 2025): add DOI/arXiv
- Docs/Project Page: https://YOUR_USERNAME.github.io/LM-Fix/
- Official Code (Papers with Code): add link after arXiv association
# LM‑Fix — Bit‑Flip Detection & Rapid Recovery for LLMs 

[![CI](https://img.shields.io/github/actions/workflow/status/YOUR_ORG_OR_USER/LM-Fix/ci.yml?branch=main)](https://github.com/YOUR_ORG_OR_USER/LM-Fix/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-Zenodo_TBD-blue)](https://doi.org/10.5281/zenodo.TBD)

**LM‑Fix** is a lightweight framework that detects **bit‑flips** in Large Language Models (LLMs) parameters and **repairs** them **without retraining or reloading**. This repo hosts the reference implementation and scripts to reproduce core results for LM-Fix.

> **Status:** v0.1.0 (release candidate) — 2025-10-22

---

## TL;DR (Why LM‑Fix?)

- **Detects faults fast:** Golden & layer‑wise hashing catch silent errors before sampling.
- **Pinpoints the fault:** Cache clearing → layer search → parameter‑level localization.
- **Repairs quickly:** Integer‑view weight editing + reference outputs; **>100× faster** than full model reload.
- **Low overhead:** ≈ **1%–7.7%** at **TVL=200** across tested models.

---

## Detection Accuracy (from the paper)

<!-- LM-Fix Table: Detection Coverage (%) / Performance Overhead (%) across TVL lengths -->
<table style="width:100%; border-collapse:collapse; font-size:13px; table-layout:fixed;">
  <colgroup>
    <col style="width:220px;"> <!-- Model (wide) -->
    <col style="width:70px;">  <!-- Params -->
    <col style="width:70px;">  <!-- Precision -->
    <col style="width:90px;">  <!-- Mem Overhead -->
    <!-- 8 TVL groups x 2 cols each = 16 cols; allow auto widths -->
    <col span="16">
  </colgroup>
  <thead>
    <tr>
      <th rowspan="2" style="border-bottom:1px solid #ccc; text-align:left; padding:6px; white-space:nowrap;">Model</th>
      <th rowspan="2" style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">Params</th>
      <th rowspan="2" style="border-bottom:1px solid #ccc; text-align:left; padding:6px;">Precision</th>
      <th rowspan="2" style="border-bottom:1px solid #ccc; text-align:left; padding:6px; white-space:nowrap;">Memory Overhead</th>
      <th colspan="2" style="border-bottom:1px solid #ccc; padding:6px;">TVL = 1</th>
      <th colspan="2" style="border-bottom:1px solid #ccc; padding:6px;">TVL = 10</th>
      <th colspan="2" style="border-bottom:1px solid #ccc; padding:6px;">TVL = 40</th>
      <th colspan="2" style="border-bottom:1px solid #ccc; padding:6px;">TVL = 100</th>
      <th colspan="2" style="border-bottom:1px solid #ccc; padding:6px;">TVL = 200</th>
      <th colspan="2" style="border-bottom:1px solid #ccc; padding:6px;">TVL = 600</th>
      <th colspan="2" style="border-bottom:1px solid #ccc; padding:6px;">TVL = 1000</th>
    </tr>
    <tr>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Det.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Perf.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Det.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Perf.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Det.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Perf.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Det.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Perf.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Det.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Perf.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Det.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Perf.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Det.</th>
      <th style="border-bottom:1px solid #ccc; padding:6px;">Perf.</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding:6px; white-space:nowrap;">LLaMa 3.2</td><td>1B</td><td>fp32</td><td>&lt; 1 KB</td>
      <td>47.6%</td><td>0.5%</td><td>84.4%</td><td>0.8%</td><td>90.6%</td><td>1.5%</td><td>94.2%</td><td>3.5%</td><td>96.6%</td><td>7.7%</td><td>98.2%</td><td>19.7%</td><td>98.9%</td><td>28.9%</td>
    </tr>
    <tr>
      <td style="padding:6px; white-space:nowrap;">LLaMa 3.2</td><td>3B</td><td>fp16</td><td>&lt; 1 KB</td>
      <td>25.7%</td><td>0.5%</td><td>73.7%</td><td>0.6%</td><td>85.3%</td><td>0.7%</td><td>91.0%</td><td>1.4%</td><td>95.1%</td><td>2.3%</td><td>97.8%</td><td>5.7%</td><td>98.8%</td><td>9.0%</td>
    </tr>
    <tr>
      <td style="padding:6px; white-space:nowrap;">LLaMa 3.2</td><td>3B</td><td>fp8</td><td>&lt; 1 KB</td>
      <td>28.05%</td><td>0.5%</td><td>68.4%</td><td>0.5%</td><td>88.2%</td><td>0.7%</td><td>93.1%</td><td>1.2%</td><td>98.9%</td><td>2.2%</td><td>98.9%</td><td>5.6%</td><td>99.2%</td><td>9.2%</td>
    </tr>
    <tr>
      <td style="padding:6px; white-space:nowrap;">Phi-4 mini</td><td>3.5B</td><td>int8</td><td>&lt; 1 KB</td>
      <td>34.8%</td><td>0.5%</td><td>96.7%</td><td>0.6%</td><td>99.6%</td><td>0.7%</td><td>99.7%</td><td>0.9%</td><td>99.8%</td><td>1.4%</td><td>99.9%</td><td>2.9%</td><td>99.9%</td><td>4.5%</td>
    </tr>
    <tr>
      <td style="padding:6px; white-space:nowrap;">Qwen 2</td><td>7B</td><td>bfp16</td><td>&lt; 1 KB</td>
      <td>44.7%</td><td>0.5%</td><td>88.9%</td><td>0.7%</td><td>81.7%</td><td>1.0%</td><td>97.1%</td><td>1.8%</td><td>99.4%</td><td>3.4%</td><td>99.6%</td><td>8.2%</td><td>99.6%</td><td>12.5%</td>
    </tr>
    <tr>
      <td style="padding:6px; white-space:nowrap;">Mistral 2</td><td>7B</td><td>fp32</td><td>&lt; 1 KB</td>
      <td>28.9%</td><td>0.5%</td><td>80.9%</td><td>0.9%</td><td>87.1%</td><td>1.5%</td><td>90.5%</td><td>2.8%</td><td>93.7%</td><td>4.7%</td><td>97.6%</td><td>10.8%</td><td>98.0%</td><td>16.2%</td>
    </tr>
    <tr>
      <td style="padding:6px; white-space:nowrap;">LLaMa 3.1</td><td>8B</td><td>bfp16</td><td>&lt; 1 KB</td>
      <td>18.9%</td><td>0.5%</td><td>69.1%</td><td>0.6%</td><td>83.7%</td><td>0.9%</td><td>89.6%</td><td>2.2%</td><td>93.6%</td><td>4.0%</td><td>97.0%</td><td>10.7%</td><td>98.1%</td><td>17.0%</td>
    </tr>
    <tr>
      <td style="padding:6px; white-space:nowrap;">Gemma 2</td><td>9B</td><td>fp16</td><td>&lt; 1 KB</td>
      <td>36.1%</td><td>0.5%</td><td>71.4%</td><td>0.6%</td><td>81.5%</td><td>0.8%</td><td>91.6%</td><td>2.2%</td><td>95.7%</td><td>2.4%</td><td>97.4%</td><td>6.5%</td><td>98.8%</td><td>9.2%</td>
    </tr>
    <tr>
      <td style="padding:6px; white-space:nowrap;">LLaMa 2</td><td>13B</td><td>fp16</td><td>&lt; 1 KB</td>
      <td>24.3%</td><td>0.4%</td><td>67.0%</td><td>0.6%</td><td>81.0%</td><td>0.6%</td><td>84.6%</td><td>0.7%</td><td>91.1%</td><td>1.0%</td><td>96.6%</td><td>2.1%</td><td>97.7%</td><td>3.5%</td>
    </tr>
    <tr>
      <td style="padding:6px; white-space:nowrap;">QwQ</td><td>30B</td><td>fp8</td><td>&lt; 1 KB</td>
      <td>9.56%</td><td>0.3%</td><td>72.3%</td><td>0.5%</td><td>87.1%</td><td>0.8%</td><td>93.0%</td><td>1.5%</td><td>95.7%</td><td>2.8%</td><td>97.2%</td><td>5.9%</td><td>97.5%</td><td>9.2%</td>
    </tr>
  </tbody>
</table>
<p style="font-size:12px;">
<b>Notes.</b> “Det.” = Detection Coverage; “Perf.” = Performance Overhead. Memory overhead is &lt;1&nbsp;KB for all models.
</p>


*TVL = length of the fixed test vector (in tokens). Longer TVL generally increases detection accuracy with modest overhead.*


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
  booktitle = {{2025 IEEE 42nd International Conference on Computer Design (ICCD)}},
  year      = {{2025}},
  organization={{IEEE}},
  author    = {{Ahmad Tahmasivand∗, Noureldin Zahran, Saba Al-Sayouri, Mohammed Fouda, and Khaled N. Khasawneh}},
}}
```

---

## Links

- Paper (ICCD 2025): add DOI/arXiv
- Docs/Project Page: https://ATA990.github.io/LM-Fix/
- Official Code (Papers with Code): add link after arXiv association

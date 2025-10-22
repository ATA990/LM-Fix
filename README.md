# LM-Fix — Bit‑Flip Detection & Rapid Recovery for LLMs (ICCD 2025)

[![CI](https://img.shields.io/github/actions/workflow/status/YOUR_ORG_OR_USER/LM-Fix/ci.yml?branch=main)](https://github.com/YOUR_ORG_OR_USER/LM-Fix/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-Zenodo_TBD-blue)](https://doi.org/10.5281/zenodo.TBD)

**LM‑Fix** is a lightweight framework that detects **silent bit‑flips** in LLMs and performs **rapid recovery** without retraining. This repository contains the reference implementation and reproduction scripts for the ICCD 2025 paper.

> 📣 **Status:** First public release candidate — 2025-10-22

---

## Abstract (from the ICCD 2025 paper)

—Bit-flip attacks threaten the reliability and security of Language Models (LMs) by altering internal parameters and compromising output integrity. Recent studies show that flipping only a few bits in model parameters can bypass safety mech- anisms and jailbreak the model. Existing detection approaches for DNNs and CNNs are not suitable for LMs, as the massive number of parameters significantly increases timing and memory overhead for software-based methods and chip area overhead for hardware-based methods. In this work, we present LM-Fix, a lightweight LM-driven detection and recovery framework that leverages the model’s own capabilities to identify and recover faults. Our method detects bit-flips by generating a single output token from a predefined test vector and auditing the output tensor of a target layer against stored reference data. The same mechanism enables rapid recovery without reloading the entire model. Experiments across various models show that LM-Fix detects more than 94% of single-bit flips and nearly 100% of multi-bit flips, with very low computational overhead ( ≈1%– 7.7% atTVL = 200 across models). Recovery achieves more than 100 ×speedup compared to full-model reload, which is critical in edge devices. LM-Fix can handle bit-flips affecting any part of the model’s computation, including memory, cache, and arithmetic operations. Evaluation against recent LM-specific bit-flip attacks confirms its robustness and practical value for real-world deployment.

**Keywords:** —Language Models (LMs), Bit-flip attacks, Secu- rity, Jailbreaking, Dependability. I. I NTRODUCTION Language Models (LMs) have become foundational com- ponents in modern artificial intelligence systems, powering a wide range of applications such as intelligent assistants, ma- chine translation, code generation, and biomedical information retrieval. This progress has been largely enabled by the under- lying Transformer architecture [1], which scales effectively to capture long-range dependencies through attention mech- anisms. Building upon this architecture, models have scaled to hundreds of billions of parameters, exhibiting remarkable capabilities in language understanding and generation. As LMs are deployed across cloud infrastructure, consumer devices, and enterprise systems [2], their memory requirements have expanded dramatically. This expanded memory require- ment, particularly in DRAM and high-bandwidth memory [3], increases bit-flip and soft-error rates [4], [5], [6]. These faults arise both naturally from cosmic radiation and electromagneticinterference [7], [8] and maliciously through fault injection techniques such as Rowhammer [9]. Although adversarial input attacks and prompt-based jail- breaks have been extensively investigated in the context of LMs [10]–[12], bit-flip attacks against LMs represent an emerging and dangerous threat vector [4]. Recent works [13], [14] show that flipping as few as 5to25bits in model weights can bypass safety alignment constraints, effectively jailbreak- ing the model and can persistently compromise it to generate harmful content. Specifically, these attacks modify weights directly in memory, using the Rowhammer vulnerability [9], to jailbreak the model without altering user input. Existing defenses against bit-flip attacks fall into two primary categories: hardware- and software-based de- fenses. Hardware techniques such as error-correcting codes (ECCs) [15] can detect and repair memory faults. However, recent research has shown vulnerabilities in these mechanisms, with attacks such as ECCploit successfully bypassing ECC protections [16]. Furthermore, comprehensive field studies on DDR4 DRAM systems reveal an increasing prevalence of single-bit and multi-bit errors driven by semiconductor scaling and contemporary workload demands [17]. Traditional Single Error Correction-Double Error Detection (SEC-DED) ECC schemes are inadequate to address a substantial por- tion of these faults, highlighting the need for more robust error correction approaches such as Chipkill ECC [17]. Other hardware-based approaches include dual modular redundancy (DMR) and triple modular redundancy (TMR) that replicate critical components and use voting mechanisms to mask faults [18]–[20]. However, they incur significant overhead in area, power, and latency, making them impractical for large-scale deployment. On the other hand, adversarial training and software-based fine-tuning approaches are computationally infeasible for multibillion parameter models due to high latency and storage demands [21]. In addition, recent detection mechanisms such as Concurrent Weight Encoding-based Detection (WED) [22] and Aspis [23] have shown efficacy in protecting DNNs and CNNs by identifying sensitive weights and embedding error detection structures. However, these methods are not directly transferable to the LM setting. First, the volume of parameters Bit-FlipBit-flip Det ection and R esponse Generation Recov ery LM R esponse Gener ation User PromptResponse Audit is Passed? DeadLock Check Alert Message for AdministratorYes No Yes 1 token generation for Test Vector - Pre-Randomization hook User response generation Hooked T ensor Auditing Cache ClearingCorrup ted P arame ters Loc alization Layer Sear chColumn Sear chRow Sear chParame ters Recalcula tionNoTasks possible to run in ParallelFig. 1: LM-Fix Framework Overview in LMs is orders of magnitude larger, making full-scale re- dundancy impractical. Second, the statistical and architectural differences between feedforward DNNs and transformer-based LMs limit the effectiveness of sensitivity-guided protection mechanisms. These challenges motivate the development of a lightweight, scalable and LM-specific solution to detect and mitigate bit- flip faults in real-time (LM-Fix). In this work, we introduce a low-overhead detection and recovery framework that takes advantage of the architectural regularity of transformer blocks to provide robust integrity check and fast response to faults. LM-Fix centers on the architecture of LMs itself, which is highly sensitive to parameter perturbation. Specifically, we introduce a hooked tensor auditing scheme , in which a fixed test vector is passed through the model to generate a known single-token output. The corresponding output tensor from the final Transformer layer before sampling is generated and stored as a reference auditing data during deployment. At inference time or under scheduled validation, the same t

---

## TL;DR

- Detect **silent faults** using **golden** and **layer‑wise hashes**.
- **Localize** errors via **cache clearing** and **layer search**.
- **Recover** parameters using **integer‑view weights**, **test vectors**, and **reference outputs**.
- Low overhead; designed for **commodity hardware** and **PyTorch** toolchains.

---

## Framework Overview

in LMs is orders of magnitude larger, making full-scale re- dundancy impractical. Second, the statistical and architectural differences between feedforward DNNs and transformer-based LMs limit the effectiveness of sensitivity-guided protection

**Threat Model:** , frame- work overview,

---

## Methodology

### Detection
functions by identifying and flagging any faults that occur within the weights of model layers by generating a fixed single-token output. When such a fault is introduced into one of these layers, the resulting change in the hooked tensor becomes observable because of the model’s predictable and deterministic behavior before any sampling is applied to the last layer’s output. The core reason why this mechanism is effective is because faults’ impact is efficiently propagated forward through the subsequent computational stages of LM models. This effect is particularly amplified because of the presence of additional linear layers further along the data path. As a result, even small changes introduced in the early or intermediate layers can accumulate and eventually lead to significant alterations in the hooked output tensor produced by the model. LM-Fix has empirically demonstrated that, for instance, the LLaMA 3.2 3B FP8 model, using approximately 200 tokens as test vector, allows for the successful detection of more than 97% of randomly injected single-bit flips (500K single-bit flip evaluated). These fault injections were performed in various layers of the model, and the results consistently confirmed the robustness of LM-Fix’s detection framework. As detailed in Algorithm 1, the framework initiates the first stage of its workflow by generating a response to the user’s input prompt. In the next step, the detection function then generates a single token using the fixed input vector and captures the corresponding hooked output tensor from the final linear layer of the model. This hooked tensor is compared directly with the reference tensor. If the two tensors are not identical, the system indicates the presence of a modification and starts the recovery mechanism. V. R ECOVERY METHODOLOGY In this section, we elaborate on the details of LM-Fix’s proposed recovery approach (Algorithm 2), which has been specifically designed based on the inherent linear proper- ties present within the Language Models. Once a fault is detected in the model, instead of performing a full model reload either from a remote server or from local disk stor- age into GPU memory, which is both time-consuming and resource-intensive, LM-Fix’s recovery mechanism focuses on selectively restoring only the corrupted parameters. This is made possible through a custom-designed mechanism that effectively restores only the faulty parameters to a safe state, without requiring the complete reloading of the model. In LM-Fix, the model plays an active role in detecting inconsistencies in its own behavior. This design choice al- lows us to detect faults that may not be directly observable through external monitoring. For example, if a fault occurs in the system cache, it is very likely that the corruption will influence the model’s output vector. As a result, LM-Fix is also able to detect and recover from cache-induced faults, an advantage that broadens the scope and robustness of the overall methodology. LM-Fix’s recovery process consists of four main compo- nents, each targeting a specific aspect of fault localization and correction: A. Cache Clearing After a fault is detected, the recovery procedure begins with clearing the system cache. If the origin of the fault lies within the cache, this step is sufficient to eliminate the problem. B. Layer Search In the event that the fault persists even after the cache has been cleared, we proceed with a more granular analysis through a layer-wise search. Using the same test vector as input, we regenerate a Layer Output Tensor (LOT) for each layer, and they are then compared against a reference LOT. If a mismatch is found, that specific layer is identified as containing a fault. This process enables precise layer-level fault localization. C. Parameter-Level Localization Once the faulty layers have been identified through the layer-wise search, the next step is to isolate the specificAlgorithm 2 Restore Model Weights( M) Require: reference Layer Output Tensors LOT ref, Model M with parameters W, Redundant data D, Test Vector T V, Model’s Reference Tensor RefT Ensure: Recovered M 1:// Step 1: Clear system cache 2:ClearCache () 3:hooked Tensor ←Get_hooked_tensor (T V,M) 4:ifhooked Tensor ==RefTthen 5: return M 6:end if 7:// Step 2: Perform layer search 8:for all layer linMdo 9:LOT l←GenerateLayerOutTensor (l,TestVector ) 10: ifLOT l̸=LOT ref[l]then 11: AddltoFaultyLayers 12: end if 13:end for 14:// Step 3: Parameter-level localization 15:for all layer linFaultyLayers do 16: Columns ←DetectFaultyColumns (l,D) 17: Rotate weight matrix of lby 90◦ 18: Rows←DetectFaultyRows (l,D) 19: // Step 4: Recover parameters 20: FaultyParams ←Intersect (Rows ,Columns ) 21:Wl[FaultyParams ] ← SolveLinearSystem (FaultyParams ,D) 22:end for 23:return Recovered M=0 parameters within the layers that have been affected. We begin by once again generating a token using the same test vector and comparing the output of the faulty layer to its previously stored reference output. The key point is that if there are discrepancies at specific indices in the output vectors, this indicates that certain parameters in the weight matrix, represented as a two-dimensional array, Figure 2. To further refine the localization, we rotate the weight matrix of the affected layer 90 degrees and regenerate the token using the test vector once again. The new output is then compared to a rotated reference output. If further mismatches are observed, we can identify the corresponding rows in the original matrix (which now appear as columns in the rotated matrix) that contain corrupted values. This bidirectional approach helps to pinpoint the exact parameter vectors that have been corrupted, Figure 3. D. Parameter Recovery After identifying both the rows and columns that likely contain corrupted parameters, by using stored reference data, we recover the corrupted parameters to the original values. To generate robust reference data for layers under fault injection conditions, we produce a special form of output for Fig. 2: Column Search Fig. 3: Row Search each layer using a test vector that aligns with the number of parameters we aim to recover. 1) Test Vector Design: Letndenote the number of pa- rameters to be supported for recovery in a specific layer. We construct a test input tensor with the following shape: x(ℓ) int∈R1×n×d(ℓ) in This test vector contains exactly ntoken vectors, each of dimension d(ℓ) in. 2) Integer View of Weights: To eliminate numerical errors inherent to IEEE-754 arithmetic, we adopt an integer-view rep- resentation of all quantities. Even small floating-point round- ing can accumulate and obstruct exact inversion when solving the linear system induced by bit-flip events. By operating directly on fixed-width binary integer encoding, computations are exact and the recovery step is bitwise lossless, returning the parameter’s original value. Before applying the test input to layer ℓ, each parameter in the weight tensor θℓis logically reinterpreted as an integer viewing its binary representation without altering the underly- ing bits. Formally: ˜θℓ= view( θℓ,asint) The linear transformation of the layer is then computed using these integer-view weights: y(ℓ) int=fℓ(x(ℓ) int;˜θℓ)∈R1×n×d(ℓ) out 3) Reference Output Storage: The output tensor is stored directly as: RefIntOutputℓ=y(ℓ) int The complete set of integer-view-based reference outputs is: {RefIntOutput1,RefIntOutput2, . . . , RefIntOutputL}based on the intersection of suspect vectors, we construct a system of linear equations. These equations model the mathematical relationship between the observed corrupted outputs and the original reference outputs. The system is then solved analytically, enabling an accurate reconstruction of the correct parameter values. By solving this system, we restore the original values of all identified faulty parameters using the previously saved reference data. This marks the completion of the fourth and final stage of the LM-Fix recovery method. VI. E VALUATION AND RESULTS A. Experimental Setup In this work, LM-Fix was implemented using the PyTorch library [31]. The models used for both experimental setup and

### Recovery
(*Recovery snippet not auto-extracted; cache clearing, layer search, integer-view repair.*)

---

## Evaluation & Results (paper excerpt)
(*Add evaluation highlights: detection accuracy, overhead, recovery rates.*)

### Conclusion (paper excerpt)
. II. T HREAT MODEL Our threat model considers deployment environments in which models operate in GPU/accelerator-equipped systems with weights loaded into DRAM and processed through high- speed cache hierarchies. Recently, the performance of open-source LMs has im- proved significantly, achieving SOTA results comparable to those of closed-source models in many tasks. Therefore, we consider a white-box threat model in which the attacker has full access to the architecture and parameters of the victim model. We assume all bit-flip occurrences, whether maliciously injected or naturally occurring, as potential security threats. Specifically, we recognize that any parameter alteration, re- gardless of origin, can lead to security vulnerabilities, per- formance degradation, or model misuse. We make no dis- tinction between malicious attacks (such as gradient-based manipulation [13], [14], [24] or physical fault injection) and accidental faults due to environmental factors (radiation, ther- mal fluctuations, or electromagnetic interference) [25], [26], as both can equally compromise the integrity of the model. This comprehensive protection approach is crucial for large- scale models where the sheer number of parameters inherently increases the probability of bit-flip events [7], [8], [27]. III. F RAMEWORK OVERVIEW In this section, we present a comprehensive overview of the LM-Fix framework, which provides robust detection and recovery mechanisms for bit-flip vulnerabilities in Language Models. The framework implements a systematic approach to ensure model integrity during inference. Specifically, LM-Fix functions as an integrated protection module that monitors and protects the LM without requiring architectural modifi- cations to the underlying model. The framework consists of two primary components: (1) a continuous bit-flip detection mechanism that operates with each user prompt, and (2) an efficient recovery system that activates when corruptions are identified (Figure 1). Verification can be performed simultaneously with response generation or scheduled at specific intervals, depending on the required security level. In our approach, integrity verification is executed after processing each user prompt but before deliver- ing the generated response. This critical design choice ensures that every output delivered to users has passed through our verification process, guaranteeing that responses are generated from an uncorrupted state of the model. Verification at longer intervals can still ensure the integrity of generated responses because bit-flip attacks such as RowHammer demand extensive effort and time. Each flip often requires tens of thousands of activations [28], and in ECC memory, profiling vulnerable bits may take days [29]. Recent research on GPU memory bit flips using RowHammer shows that, in their experiments, it took approximately 30 hours per memory bank to induce an effective bit flip [30]. The detection process leverages a Hooked Tensor Auditing technique, which takes the output tensor of the last layer before the sampling step in the LM model architecture and then compares runtime computation values against pre-established

---

## Reproducing the Paper

> **Note:** Use the provided scripts to run the four core steps end‑to‑end. Small models are recommended for quick tests.

1. **Generate Golden Hash**
   ```bash
   python -m lmfix.scripts.generate_golden --model meta-llama/Llama-3.2-3B --tokens 128 --out artifacts/golden.json
   ```
2. **Save Layer Output Hashes (constant-input & param‑input)**
   ```bash
   python -m lmfix.scripts.save_layer_hashes --model meta-llama/Llama-3.2-3B --mode ones --out artifacts/layer_hashes.json
   python -m lmfix.scripts.save_layer_hashes --model meta-llama/Llama-3.2-3B --mode param --out artifacts/layer_param_hashes.json
   ```
3. **Step 3 Hooks (Input Shape Enforcement)**
   ```bash
   python -m lmfix.scripts.step3_hooks --model meta-llama/Llama-3.2-3B --shape 1,1,<num_rows> --save artifacts/step3_outputs.pt
   ```
4. **Step 4 Rotation + Repair (then restore)**
   ```bash
   python -m lmfix.scripts.step4_rotate_repair --model meta-llama/Llama-3.2-3B --rotate --restore --ref artifacts/step3_outputs.pt
   ```

*(Replace module paths with your actual script/module names if different.)*

---

## Installation

```bash
git clone https://github.com/YOUR_ORG_OR_USER/LM-Fix.git
cd LM-Fix
pip install -r requirements.txt
```

---

## Cite

If you use LM‑Fix, please cite the software and the paper:

```bibtex
@inproceedings{{lmfix_iccd_2025,
  title     = {{LM-Fix}: Lightweight Bit-Flip Detection and Rapid Recovery Framework for Language Models},
  booktitle = {{ICCD}},
  year      = {{2025}},
  author    = {{Add author list}},
}}
```
Also see `CITATION.cff` for software citation and DOI.

---

## Project Links

- **Docs/Project Page:** https://YOUR_USERNAME.github.io/LM-Fix/
- **Paper (ICCD 2025):** (add DOI or arXiv link)
- **Demo (Hugging Face Space):** (add link)
- **Official Code on Papers with Code:** (link after arXiv association)

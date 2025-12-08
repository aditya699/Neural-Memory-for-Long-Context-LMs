# Continual Pre-Training of Large Language Models: A Practitioner's Guide

**Aditya Bhatt**  
December 2025

---

## Abstract

Large Language Models (LLMs) are typically trained on static datasets, leading to knowledge cutoffs and domain gaps. Continual Pre-Training (CPT) offers a solution by incrementally updating models on new data without complete retraining. This paper synthesizes insights from recent research, particularly the comprehensive survey by Shi et al. (2024), and demonstrates practical implementation through a medical domain adaptation experiment using Qwen2.5-0.5B.

---

## 1. Introduction

The development of LLMs like GPT-5, LLaMA, and Qwen has revolutionized natural language processing. However, these models face a fundamental limitation: they are trained on static, pre-collected datasets. This creates two critical problems:

1. **Temporal Degradation**: Knowledge becomes outdated as the world evolves
2. **Domain Gaps**: General-purpose models lack specialized expertise for vertical domains

Retraining from scratch is prohibitively expensive—GPT-4 reportedly cost over $100 million to train. Continual Pre-Training (CPT) emerges as an efficient alternative, enabling models to learn new knowledge while preserving existing capabilities.

## 2. Understanding Continual Pre-Training

### 2.1 Definition and Scope

Continual Pre-Training refers to the process of further training a pre-trained LLM on additional unlabeled corpora to adapt it to new domains, time periods, or languages. Unlike fine-tuning, which targets specific downstream tasks with labeled data, CPT maintains the original pre-training objective (next-token prediction) while exposing the model to new distributions.

The survey by Shi et al. introduces a crucial framework distinguishing two dimensions of continuity:

**Vertical Continuity**: Adaptation from general to specific capabilities (e.g., general LLM → medical LLM → clinical note generator). This follows a hierarchical structure where each stage builds upon the previous.

**Horizontal Continuity**: Adaptation across time and domains at the same level (e.g., news from 2023 → news from 2024). This addresses the temporal evolution of knowledge.

### 2.2 Types of Distributional Shifts

CPT addresses three primary types of shifts:

| Shift Type | Description | Example |
|------------|-------------|---------|
| **Temporal** | Knowledge changes over time | "Messi plays for Barcelona" → "Messi plays for Inter Miami" |
| **Content** | Different subject domains | Chemistry corpus → Biology corpus |
| **Language** | Different linguistic corpora | English → Chinese |

## 3. The Catastrophic Forgetting Problem

The central challenge in CPT is **catastrophic forgetting**—the tendency of neural networks to lose previously learned knowledge when trained on new data. When a model's weights shift to accommodate new patterns, they may overwrite representations critical for earlier tasks.

Research by Li & Lee (2024) demonstrates that forgetting manifests differently across model components:
- **Knowledge (factual information)**: Tends to remain relatively intact
- **Reliability and formatting**: Degrades more significantly
- **Instruction-following ability**: Particularly vulnerable to degradation

### 3.1 Mitigation Techniques

The continual learning community has developed several strategies:

**Replay-Based Methods**: Maintain a buffer of previous data and interleave it during training. Simple yet effective—mixing 5-25% general data during domain CPT significantly reduces forgetting.

**Regularization-Based Methods**: Penalize changes to important parameters. Elastic Weight Consolidation (EWC) uses the Fisher Information Matrix to identify and protect critical weights.

**Architecture-Based Methods**: Expand the network for new knowledge. LoRA (Low-Rank Adaptation) trains small adapter modules while freezing base weights, achieving near-zero forgetting on original capabilities.

## 4. Practical Implementation: Medical Domain Adaptation

To demonstrate CPT in practice, we conducted an experiment adapting Qwen2.5-0.5B for medical documentation.

### 4.1 Experimental Setup

- **Base Model**: Qwen2.5-0.5B (494M parameters)
- **Dataset**: AGBonnet/augmented-clinical-notes (30,000 clinical summaries)
- **Hardware**: RTX 4000 Ada (20GB VRAM)
- **Training**: 1 epoch, batch size 16 (effective), learning rate 2e-5

### 4.2 Training Configuration

```python
TrainingArguments(
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch = 16
    learning_rate=2e-5,             # Small LR preserves knowledge
    warmup_steps=100,               # Gradual LR increase
    bf16=True                       # Match model precision
)
```

### 4.3 Results

| Metric | Before CPT | After CPT |
|--------|------------|-----------|
| Output Style | MCQ/Exam format | Clinical documentation |
| Medical Terminology | Present but contextually weak | Contextually appropriate |
| Training Loss | - | 2.44 |
| Training Time | - | ~29 minutes |

**Qualitative Improvement Example**:

*Prompt*: "On examination, the patient showed signs of dystonia with"

- **Before**: "most likely diagnosis is A. Parkinson's B." (exam format)
- **After**: "myoclonus in both arms and lower limbs. There was no evidence for muscle atrophy..." (clinical finding)

## 5. Key Insights from Current Research

### 5.1 Observations from the Survey

Shi et al.'s comprehensive analysis of 100+ papers reveals:

1. **Technique Diversity is Limited**: Most CPT implementations use architecture expansion (MoE, LoRA), with fewer exploring regularization or sophisticated replay strategies.

2. **Production Gap**: Academic studies typically explore 4-8 domain shifts, while real-world scenarios involve continuous streams over months or years.

3. **Small Models Benefit More**: Models under 1.5B parameters show consistent improvement from CPT, while larger models exhibit more complex forgetting patterns.

### 5.2 Domain-Adaptive Pre-Training (DAP)

DAP represents a specific form of CPT for vertical domains. The survey catalogs 41 studies across:
- **Medical**: PMC-LLaMA, HuatuoGPT-II, Me-LLaMA
- **Financial**: BBT-Fin, CFGPT, XuanYuan
- **Legal**: SaulLM, Lawyer LLaMA
- **Code**: Code LLaMA, DeepSeek-Coder, StarCoder

A critical finding: mixing 5-25% general-domain data during DAP consistently reduces vertical forgetting while maintaining domain adaptation benefits.

## 6. Best Practices for Practitioners

Based on research synthesis and practical experience:

1. **Start with Small Learning Rates**: Use 1e-5 to 5e-5; larger rates accelerate forgetting.

2. **Implement Data Mixing**: Include 5-10% general data in your training batches.

3. **Use Parameter-Efficient Methods**: LoRA or adapters when computational resources are limited or forgetting must be minimized.

4. **Monitor Both Directions**: Evaluate on both new domain tasks AND general benchmarks.

5. **Consider Model Size**: Smaller models (<1.5B) are more amenable to CPT but also more susceptible to forgetting.

6. **Warmup is Critical**: Gradual learning rate increase prevents early instability and catastrophic weight updates.

## 7. Future Directions

The survey identifies several open challenges:

- **Theoretical Foundations**: Lack of rigorous understanding of when and why CPT succeeds or fails
- **Efficient Replay**: Developing smarter data selection for replay buffers
- **Controllable Memory**: External memory systems enabling selective knowledge updates
- **Continuous Streams**: Moving beyond discrete task boundaries to true online learning

## 8. Conclusion

Continual Pre-Training represents a crucial capability for keeping LLMs relevant and specialized without prohibitive retraining costs. While catastrophic forgetting remains a challenge, practical strategies—particularly data mixing and parameter-efficient methods—enable effective domain adaptation. Our medical documentation experiment demonstrates that even small-scale CPT (30K examples, 30 minutes) can meaningfully shift model behavior toward domain-appropriate outputs.

As the field matures, we expect to see more sophisticated techniques bridging the gap between academic benchmarks and production requirements, enabling truly lifelong learning systems.

---

## References

1. Shi, H., Xu, Z., Wang, H., et al. (2024). Continual Learning of Large Language Models: A Comprehensive Survey. *arXiv:2404.16789*.

2. Li, C.-A., & Lee, H.-Y. (2024). Examining Forgetting in Continual Pre-training of Aligned Large Language Models. *arXiv:2401.03129*.

3. Yıldız, Ç., et al. (2024). Investigating Continual Pretraining in Large Language Models: Insights and Implications. *arXiv:2402.17400*.

4. Hu, E. J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*.

5. Kirkpatrick, J., et al. (2017). Overcoming Catastrophic Forgetting in Neural Networks. *PNAS*.

6. Gururangan, S., et al. (2020). Don't Stop Pretraining: Adapt Language Models to Domains and Tasks. *ACL 2020*.

---

*Code and experimental details available at: github.com/aditya699*
# BART-FaCT 实验方案

## 论文题目

**中文:** BART-FaCT: 基于层次化结构编码与校准式解码的长文档摘要事实性增强方法

**English:** BART-FaCT: Faithfulness-Enhanced Long-Document Summarization via Hierarchical Structure Encoding and Calibrated Faithfulness Attention

---

## 模块一览

| 模块 | 全称 | 针对问题 | 灵感来源 |
|:---|:---|:---|:---|
| **HSE** | Hierarchical Structure Encoding | 扁平编码丢失文档层次结构（句子→段落→章节） | Liu & Lapata, EMNLP'19; Liu et al., TACL'23 |
| **CFA** | Calibrated Faithfulness Attention | 解码器交叉注意力无法区分「检索」与「猜测」 | Chuang et al., ICLR'24; Shi et al., ACL'24 |
| **CPO** | Contrastive Preference Optimization | MLE训练缺乏事实性偏好信号 | Rafailov et al., NeurIPS'23; Gao et al., EMNLP'24 |

详细设计见 README.md 和 README_zh.md。

---

## 实验设计

### E1: 多模型对比

| 模型 | 预训练 | 参数 |
|:---|:---|:---:|
| BART-Large-CNN | CNN/DailyMail | 400M |
| PEGASUS-arXiv | arXiv | 568M |
| PEGASUS-CNN/DM | CNN/DailyMail | 568M |
| DistilBART-CNN-12-6 | BART-Large-CNN蒸馏 | 306M |
| **BART-FaCT** | BART-Large-CNN + HSE + CFA + CPO | ~403M |

所有对比模型均为 HuggingFace 上预训练好的摘要模型，直接可跑推理。

### E2: 模块消融

| 配置 | HSE | CFA | CPO |
|------|:---:|:---:|:---:|
| BART-Large-CNN | ✗ | ✗ | ✗ |
| w/o HSE | ✗ | ✓ | ✓ |
| w/o CFA | ✓ | ✗ | ✓ |
| w/o CPO | ✓ | ✓ | ✗ |
| BART-FaCT (Full) | ✓ | ✓ | ✓ |

### E3: 事实性分析
NLI蕴含率、幻觉类型分布 (内在/外在/矛盾)、CFA各层门控分布

### E4: 上下文长度消融
256 / 512 / 768 / 1024 tokens

### E5: 参数敏感性
beam_size [1,2,4,6,8] / length_penalty [0.6,1.0,1.5,2.0,2.5] / CPO λ [0.01,0.05,0.1,0.2,0.5] / CFA瓶颈维度 [32,64,128,256] / LR [1e-5,3e-5,5e-5,1e-4]

### E6: 截断策略
head_only / tail_only / head_tail_mixed

---

## 评估指标

**质量:** ROUGE-1/2/L/Lsum, BERTScore F1, METEOR

**事实性:** NLI Entailment Ratio, Hallucination Rate (intrinsic/extrinsic/contradiction), n-gram Overlap

**辅助:** Compression Ratio, JS Divergence, 4-gram Repetition Ratio

---

## 运行

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python src/run_experiments.py --mode full --dataset arxiv --max_samples 1000
python src/run_experiments.py --mode ablation --ablation_type all
```

---

## 参考文献

1. Liu & Lapata. "Hierarchical Transformers for Long Document Summarization." EMNLP 2019.
2. Chuang et al. "DoLa: Decoding by Contrasting Layers Improves Factuality." ICLR 2024.
3. Rafailov et al. "Direct Preference Optimization." NeurIPS 2023.
4. Shi et al. "Context-Aware Decoding for Faithful Summarization." ACL 2024.
5. Gao et al. "Model-based Preference Optimization in Summarization without Human Feedback." EMNLP 2024.
6. Liu et al. "Lost in the Middle: How Language Models Use Long Contexts." TACL 2023.
7. Lewis et al. "BART: Denoising Sequence-to-Sequence Pre-training." ACL 2020.
8. Zhang et al. "PEGASUS: Pre-training with Extracted Gap-sentences." ICML 2020.

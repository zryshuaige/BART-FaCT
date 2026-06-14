# SUMM-Lens 实验方案

## 论文题目

**中文：** SUMM-Lens：长文档摘要的零训练推理期增强 —— 基于链式密度与 NLI 重排的轻量级方法

**English:** SUMM-Lens: Zero-Training Inference-Time Enhancements for Long-Document Summarization via Chain-of-Density Prompting and NLI Reranking

---

## 模块一览

| 模块 | 全称 | 针对问题 | 训练？ | 灵感来源 |
|:---|:---|:---|:---:|:---|
| **CoD** | Chain-of-Density Prompting | 单次生成的摘要稀疏，遗漏关键实体/数字 | ✗ | Adams et al., EMNLP 2023 |
| **NLR** | NLI-Rerank | 单条采样不可靠，需要从多候选中选最忠实的 | ✗ | Laban et al., TACL 2022 |

详细设计见 [README.md](README.md) 与 [README_zh.md](README_zh.md)。

---

## 实验设计

### E1：基线阶梯对比（2019 → 2024）

| 模型 | 预训练 | 参数 |
|:---|:---|:---:|
| BART-Large-CNN | CNN/DailyMail | 400M |
| DistilBART-CNN | CNN/DailyMail | 306M |
| PEGASUS-arXiv | arXiv | 568M |
| LED-large-arXiv | arXiv（长文档专用） | 460M |
| **Qwen2.5-1.5B-Instruct** | 通用指令微调 | 1.5B |

所有模型从 HuggingFace 直接 `from_pretrained` 加载，**仅推理，零训练**。

### E2：模块消融（同一主干 = Qwen2.5-1.5B-Instruct）

| 配置 | CoD | NLR | 回答的问题 |
|:---|:---:|:---:|:---|
| Qwen2.5-Vanilla | ✗ | ✗ | 2024 LLM 在零样本下能达到什么水平？ |
| + CoD | ✓ | ✗ | 单独的迭代密化是否提升 ROUGE / 忠实度？ |
| + NLR | ✗ | ✓ | 单独的 NLI 候选选择是否提升？ |
| **+ CoD + NLR** | **✓** | **✓** | 两者是否互补？ |

### E3：忠实度分析

NLI 蕴含率（RoBERTa-large-MNLI）、幻觉率、按候选打分的分布对比。

> 旧方案中的"上下文长度敏感性 / 学习率敏感性 / 截断策略"扫描在零训练设定下不再适用，已删除。

---

## 评估指标

**质量：** ROUGE-1 / ROUGE-2 / ROUGE-L / ROUGE-Lsum，BERTScore F1，METEOR

**忠实度：** NLI 蕴含率，逐候选蕴含分（仅 NLR 配置）

**辅助：** JS 散度（bigram），4-gram 重复率，压缩比，新颖度

---

## 运行

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 冒烟测试（5 样本，免费 Colab/CPU 友好）
python src/run_experiments.py --mode quick_test --dataset arxiv

# 全套（baseline + ablation + figures）
python src/run_experiments.py --mode all --dataset arxiv --num_test 100
python src/run_experiments.py --mode all --dataset pubmed --num_test 100
```

输出位置：`results/<timestamp>/`，含 `eval_results.json`（每模型）、`predictions.json`（前 50 条样例）、`figures/`（PNG/PDF/CSV）和 `all_results.json`（聚合）。

---

## 参考文献

1. Adams G, et al. *From Sparse to Dense: GPT-4 Summarization with the Chain of Density Prompt.* EMNLP 2023.
2. Laban P, et al. *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL 2022.
3. Qwen Team. *Qwen2.5 Technical Report.* 2024.
4. Lewis M, et al. *BART: Denoising Sequence-to-Sequence Pre-training.* ACL 2020.
5. Zhang J, et al. *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization.* ICML 2020.
6. Beltagy I, et al. *Longformer: The Long-Document Transformer.* arXiv:2004.05150, 2020.

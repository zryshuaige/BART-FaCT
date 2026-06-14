<div align="center">

# SUMM-Lens

### 长文档摘要的零训练推理期增强方法

[![Open in Colab](https://img.shields.io/badge/Open%20in-Colab-F9AB00?logo=googlecolab&logoColor=white)](https://colab.research.google.com/github/YOUR_REPO/blob/main/notebooks/run.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.40+-FFD21E)](https://huggingface.co/docs/transformers)
[![HF Models](https://img.shields.io/badge/🤗%20Models-Qwen2.5%20%7C%20BART%20%7C%20PEGASUS%20%7C%20LED-blue)](https://huggingface.co/models?pipeline_tag=summarization)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 摘要

arXiv 与 PubMed 等长文档摘要任务面临三个相互交织的难题。**（P1）训练成本。** 主流编码器-解码器基线（BART、PEGASUS、LED）参数量在 300–600M，必须针对每个数据集单独微调，这在免费算力（Colab 免费版、CPU）上并不现实。**（P2）单次生成的稀疏性。** 指令微调 LLM 在零样本下产出的摘要虽流畅，但通常 *稀疏* —— 倾向于错过关键实体、数字、跨章节的发现，因为单次前向只能在文档上"提交一次"采样轨迹。**（P3）单样本解码的不可靠。** 一份流畅的摘要可能在不被察觉的情况下偏离源文；常规的贪心/Beam 解码内部没有任何机制偏向"最忠实于源文"的候选。

本文提出 **SUMM-Lens**，**在不训练任何参数的前提下** 同时解决上述三个问题。我们以 Qwen2.5-1.5B-Instruct（Apache 2.0，2024）作为固定的现代主干，叠加两个轻量推理期模块：

- **CoD — 链式密度提示（约 80 行）** 解决 **P2**：通过迭代地"在保持长度的前提下补全摘要中遗漏的关键实体"，把稀疏的初稿密化为信息密度更高的摘要，且过程中不更新任何梯度。
- **NLR — NLI 重排（约 120 行）** 解决 **P3**：采样 K 条多样化候选摘要，对每条候选按句拆分并使用预训练 MNLI 模型逐句对源文做蕴含判断，最终选出"每句都最被源文蕴含"的那条，从单点估计转向忠实度感知的多候选选择。

零训练这一性质本身解决了 **P1**：所有组件即插即用、与数据集无关、可在免费 Colab 与 CPU 上运行。我们在 arXiv 与 PubMed 上评估了一个 2019→2024 的基线阶梯（BART-Large-CNN / DistilBART / PEGASUS-arXiv / LED-arXiv / Qwen2.5-Vanilla），并通过 4 配置消融（Vanilla / +CoD / +NLR / +Both）分离两个模块各自的贡献。最终我们得到了一个完全可复现、零训练成本的事实性增强摘要管线 —— 既可作为算力受限场景下的强零样本基线，也可作为任何 causal-LM 摘要器之上的"即插即用"忠实度增强模块。

---

## 两个模块

### CoD · 链式密度提示

小模型一次性给出的初版摘要往往是稀疏的 —— 几句高层概括，错过实体、数字、跨章节的发现。CoD 把摘要看作一个需要被 *增稠* 的产物，而不是 *一次性生成* 的产物。从一份种子摘要出发，模型被要求重写 3 次；每一次重写必须 (a) 找到当前摘要中遗漏的 1–3 个关键实体或数字，(b) 把它们整合进摘要，同时整体长度保持基本不变。这种"长度预算"迫使模型压缩信息密度低的措辞，腾出空间给更有信息量的内容。

模块本质是一个 3 轮循环 + 一个固定提示模板，不更新任何模型权重。在 causal-LM 主干上以 chat 形式端到端运行；在 seq2seq 主干（BART/PEGASUS）上自动降级为单轮生成。

> **文献依据。** Adams et al. *From Sparse to Dense: GPT-4 Summarization with the Chain of Density Prompt.* EMNLP 2023.

### NLR · NLI 重排

单条采样得到的摘要只是一个点估计。NLR 把摘要从 generate 变成 generate-then-select：在 temperature 0.7 / top-p 0.95 下采样 4 条候选，对每条候选按句拆分，用预训练 MNLI 模型（`roberta-large-mnli`）以源文为前提、摘要中每句为假设进行打分。我们对每个候选取所有句子的"蕴含概率均值"作为忠实度分，最后返回得分最高的那条摘要。

NLR 完全零样本，复用了项目中评估幻觉时已加载的同一个 NLI checkpoint —— 没有额外权重，没有微调，约 120 行代码。

> **文献依据。** Laban et al. *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL 2022. 以及 2024 年关于忠实度感知重排的相关工作。

当 CoD 和 NLR 联合使用时，NLR 采样 K 个种子，对每个种子分别跑 CoD，再在密化后的候选集合上做 NLI 重排。

---

## 基线阶梯（2019 → 2024）

所有模型 **仅用于推理**。每个模型都通过 `AutoModel*.from_pretrained` 直接加载。

| 年份 | 模型 | HF 路径 | 参数 | 备注 |
|:---:|:---|:---|:---:|:---|
| 2019 | BART-Large-CNN | `facebook/bart-large-cnn` | 400M | seq2seq 基线 |
| 2020 | DistilBART-CNN | `sshleifer/distilbart-cnn-12-6` | 306M | 蒸馏版，推理快 |
| 2020 | PEGASUS-arXiv | `google/pegasus-arxiv` | 568M | arXiv 专用 |
| 2020 | LED-arXiv | `allenai/led-large-16384-arxiv` | 460M | 长文档专用 |
| **2024** | **Qwen2.5-1.5B-Instruct** | `Qwen/Qwen2.5-1.5B-Instruct` | 1.5B | 现代零样本基线 |
| **本文** | Qwen2.5 + CoD + NLR | — | 1.5B | 推理期增强 |

---

## 消融设计

四个消融配置共享同一个 2024 年主干（`Qwen/Qwen2.5-1.5B-Instruct`），仅推理期管线不同。

| 配置 | CoD | NLR | 所回答的问题 |
|:---|:---:|:---:|:---|
| Qwen2.5-Vanilla | ✗ | ✗ | 2024 LLM 在零样本下能达到什么水平？ |
| + CoD | ✓ | ✗ | 单独使用迭代密化是否有帮助？ |
| + NLR | ✗ | ✓ | 单独使用 NLI 候选选择是否有帮助？ |
| **+ CoD + NLR** | **✓** | **✓** | 两个模块是否互补？ |

---

## 快速开始

```bash
git clone <repo-url> && cd end
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 冒烟测试（5 个样本、3 个模型、不算 BERTScore/METEOR —— CPU 几分钟即可）
python src/run_experiments.py --mode quick_test --dataset arxiv

# 完整 baseline 阶梯
python src/run_experiments.py --mode baseline --dataset arxiv --num_test 100

# 仅模块消融
python src/run_experiments.py --mode ablation --dataset arxiv --num_test 100

# 全部 + 出图
python src/run_experiments.py --mode all --dataset arxiv --num_test 100
```

`notebooks/run.ipynb` 兼容 Colab，包含数据集预览 → 基线 → 消融 → 出图的完整流程。

---

## 架构

```
                    ┌──────────────────────────────┐
   长文档    ──────► │  Qwen2.5-1.5B-Instruct       │ ── 种子摘要 ──┐
                    │  (零样本、chat 模板)         │               │
                    └──────────────────────────────┘               │
                                                                   ▼
                                  ┌──────── Chain-of-Density ─────┐
                                  │  iter 1: 补 1-3 个遗漏实体    │
                                  │           保持长度不变        │
                                  │  iter 2: 同上                 │
                                  │  iter 3: 同上                 │
                                  └─────────────┬─────────────────┘
                                                │ K 条密化后的候选
                                                ▼
                                  ┌──────── NLI-Rerank ────────────┐
                                  │  对每个候选:                   │
                                  │    拆句 → sentence              │
                                  │    NLI(article, sentence)       │
                                  │    score = mean P(entail)       │
                                  │  返回 argmax                    │
                                  └─────────────┬───────────────────┘
                                                │
                                                ▼
                                          最终摘要
```

---

## 评估指标

**质量：** ROUGE-1/2/L/Lsum、BERTScore F1、METEOR

**忠实度：** NLI 蕴含率（RoBERTa-large-MNLI），逐候选的蕴含分

**辅助：** JS 散度（n-gram）、4-gram 重复率、压缩比、新颖度

所有指标由 `src/benchmark.py` 计算，忠实度由 `src/hallucination.py` 计算（与 NLR 共用同一份 NLI 模型）。

---

## 项目结构

```
end/
├── src/
│   ├── methods/                # ★ 推理期模块（零训练）
│   │   ├── llm_summarizer.py   #   seq2seq / causal-LM 统一封装
│   │   ├── cod.py              #   Chain-of-Density（~80 行）
│   │   ├── nli_rerank.py       #   NLI 重排（~120 行）
│   │   └── prompts.py          #   各数据集的提示模板
│   ├── config.py               # 模型注册表与运行配置
│   ├── data_utils.py           # arXiv / PubMed 加载器、HF 镜像
│   ├── evaluate.py             # 单模型推理 + 评估
│   ├── benchmark.py            # ROUGE / BERTScore / METEOR / JS / ...
│   ├── hallucination.py        # NLI 幻觉检测器（与 NLR 共用）
│   ├── analyze.py              # 出图 + LaTeX 表（中文字体感知）
│   ├── visualization.py        # notebook 辅助函数
│   └── run_experiments.py      # 顶层 CLI
├── notebooks/
│   └── run.ipynb               # Colab / VSCode 完整 walkthrough
├── data/
│   ├── arxiv/                  # 已缓存的 HF 数据集
│   └── pubmed/
├── results/                    # 评估输出（每模型 JSON + 图表）
├── requirements.txt
├── README.md / README_zh.md / 论文.md
└── EXPERIMENT_PLAN.md
```

---

## 参考文献

1. Adams G, et al. *From Sparse to Dense: GPT-4 Summarization with the Chain of Density Prompt.* EMNLP 2023.
2. Laban P, et al. *SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization.* TACL 2022.
3. Qwen Team. *Qwen2.5 Technical Report.* 2024.
4. Lewis M, et al. *BART: Denoising Sequence-to-Sequence Pre-training.* ACL 2020.
5. Zhang J, et al. *PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization.* ICML 2020.
6. Beltagy I, et al. *Longformer: The Long-Document Transformer.* arXiv:2004.05150, 2020.

---

## 许可证

SUMM-Lens 代码采用 MIT 许可证。预训练模型遵循其各自的 HuggingFace 许可证（Qwen2.5 为 Apache 2.0；BART/PEGASUS/LED 为各自的研究许可）。

<div align="center">

# BART-FaCT

### 面向长文档摘要的事实性增强方法：层次化结构编码与校准式忠实度注意力

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_REPO/blob/main/notebooks/run.ipynb)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.35+-FFD21E)](https://huggingface.co/docs/transformers)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

</div>

---

## 摘要

为科学论文生成忠实摘要是困难的，原因有二，且二者相互加剧。其一，BART 等标准编码器-解码器模型将论文处理为扁平的 token 序列——摘要中的一句话和方法章节中的一句话对编码器来说并无区别，模型不知道每条信息处于论文论证结构的什么位置。其二，解码器的交叉注意力在每一步对所有编码器位置等权分配权重，模型无法判断自己是在从源文中检索某个具体结果，还是在猜测一个听上去合理的数字。上下文越长，注意力越弥散，模型就越倾向于依赖其语言模型先验进行补全，生成流畅但缺乏源文依据的内容。最大似然估计下的交叉熵损失无法区分这两种行为。

我们提出 **BART-FaCT**，它在一个已预训练于摘要任务的 BART 模型上增加三个轻量模块，每个模块针对一类特定的失效模式。**HSE**（层次化结构编码）学习文档的层次化表示——句子、段落、章节——并将其注入 token 嵌入，使编码器知晓每个 token 位于论文论证结构的何处。**CFA**（校准式忠实度注意力）用一个瓶颈网络包裹每层解码器交叉注意力，估计逐 token 的忠实度不确定性，并据此调节源文注意力的贡献：不确定时更仔细地回看源文。**CPO**（对比式偏好优化）以模型自身在无编码器条件下的生成作为非偏好响应，用 DPO 风格的偏好损失替代启发式负采样，直接告诉模型「有源文依据的摘要优于凭空生成的摘要」。

我们在 arXiv 和 PubMed 两个长文档摘要基准上进行了评估，与四个预训练摘要模型进行了对比，并通过五组模块消融分离了各组件的独立贡献。

---

## 三个模块

### HSE · 层次化结构编码

科学论文天然具有层次化的组织结构——论点建立在方法之上，结果支撑结论——但 BART 将整篇论文当作一长串字符来读。此前的工作尝试用正则表达式检测章节标题（如匹配 "Introduction"）来解决这一问题，但这在不同学科间泛化性很差，也无法捕捉句子层面的篇章推进。

HSE 采用了不同的思路。它先用 NLTK 的语言学句子分割器检测句子边界，再对每个句子的 token 表示做 mean-pool，用一个紧凑的 2 层 Transformer（4 头、256 维 FFN、Pre-LN、GELU）对这些句子向量进行编码，建模句子之间的逻辑关系和篇章位置。结构增强的表示通过可学习门控广播回每个 token：

```
增强嵌入 = token嵌入 + σ(W·[token嵌入 ⊕ 结构上下文]) ⊙ 结构上下文
```

此后编码器看到的就不仅是「这是第 547 号 token」，而是「这是 Methods 章节的第三句话」。模块约增加 270 万参数，仅占 BART-Large 骨架的 0.7%。

> **文献依据。** Hierarchical Transformers (Liu & Lapata, EMNLP 2019); Lost in the Middle (Liu et al., TACL 2023)。

---

### CFA · 校准式忠实度注意力

在标准解码器中，交叉注意力对编码器状态求加权平均并与自注意力输出相加。无论是正在从源文检索事实还是在生成过渡句，执行的操作完全相同。模型无法表达「我对此处不确定——我需要更仔细地看源文」。

CFA 赋予了每个解码器层这种能力。一个小型瓶颈网络接收交叉注意力输出和自注意力状态，通过 128 维隐藏层压缩后估计每个 token 的不确定性标量。该不确定性作为加性偏置作用于忠实度门控：

```
不确定性 = σ(MLP([cross_attn ⊕ self_attn]))
门控 = σ(W·瓶颈 + 不确定性)
输出 = 门控 ⊙ cross_attn + (1−门控) ⊙ self_attn
```

当模型不确定时（注意力弥散、cross/self 差异大），门控值偏向源文——「回去看论文」。当模型自信时，门控允许更自由的表达。与原始解码器输出的 0.5–0.5 残差混合保证了训练稳定性。单层开销约 32 万参数，12 层合计约 380 万。

> **文献依据。** DoLa (Chuang et al., ICLR 2024); Context-Aware Decoding (Shi et al., ACL 2024)。

---

### CPO · 对比式偏好优化

交叉熵损失教模型挑选概率最高的 token，但它不会告诉模型摘要应当有源文依据。一个编造的数字和一个忠实转述的数字，只要都和参考措辞不同，就获得相同的损失。

此前的工作通过 InfoNCE 对比损失来缓解这一问题，但负样本是手工扰动构造的——替换实体、篡改数字、打乱句子。这些合成扰动未必反映模型真实的失效模式。CPO 用模型自身来构造负样本：偏好响应是人类参考摘要，非偏好响应是模型在编码器隐状态全为零的条件下——纯解码器，从 BOS 出发——生成的文本。这是模型完全脱离源文的语言先验。这份文本中的任何事实性内容，按构造即是幻觉。

DPO 风格的偏好损失将模型拉向有源文依据的生成：

```
L_cpo = −log σ(β·[log π(y_pref|x) − log π(y_disf|x)])
L_total = L_ce + λ·L_cpo   (λ=0.15, β=0.5)
```

投影头约增加 120 万参数。

> **文献依据。** DPO (Rafailov et al., NeurIPS 2023); Model-based Preference Optimization (Gao et al., EMNLP 2024)。

---

## 消融设计

| 配置 | HSE | CFA | CPO | 所回答的问题 |
|:---|:---:|:---:|:---:|:---|
| BART-Large-CNN | ✗ | ✗ | ✗ | 预训练摘要模型在无增强时的基线水平 |
| w/o HSE | ✗ | ✓ | ✓ | 结构编码能否在忠実度模块之上提供额外增益？ |
| w/o CFA | ✓ | ✗ | ✓ | 校准式注意力能否进一步降低幻觉？ |
| w/o CPO | ✓ | ✓ | ✗ | 偏好优化能否进一步提升事实性？ |
| **完整模型** | **✓** | **✓** | **✓** | 三者是否互补？ |

---

## 对比模型

所有对比模型均为预训练好的摘要模型，可直接用于推理。

| 模型 | 来源 | 参数 |
|:---|:---|:---:|
| BART-Large-CNN | `facebook/bart-large-cnn` | 400M |
| PEGASUS-arXiv | `google/pegasus-arxiv` | 568M |
| PEGASUS-CNN/DM | `google/pegasus-cnn_dailymail` | 568M |
| DistilBART-CNN-12-6 | `sshleifer/distilbart-cnn-12-6` | 306M |
| **BART-FaCT** | 本文 | **~403M** |

---

## 快速开始

```bash
git clone <repo-url> && cd end
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 冒烟测试
python src/run_experiments.py --mode quick_test --dataset arxiv

# 多模型对比
python src/run_experiments.py --mode exp1 --dataset arxiv \
    --models "bart-large-cnn,pegasus-arxiv,pegasus-cnn_dailymail,distilbart-cnn-12-6" \
    --max_samples 1000 --num_test 100

# 模块消融
python src/run_experiments.py --mode ablation --ablation_type all
```

---

## 评估指标

**质量：** ROUGE-1/2/L/Lsum, BERTScore F1, METEOR

**事实性：** NLI 蕴含率 (RoBERTa-large-MNLI), 幻觉率 (内在/外在/矛盾), n-gram 重叠率

**辅助：** 压缩比, JS 散度, 4-gram 重复率

---

## 项目结构

```
end/
├── src/
│   ├── models/
│   │   ├── bart_fact.py              # 主模型与配置
│   │   ├── hierarchical_structure.py # HSE 模块
│   │   ├── calibrated_attention.py   # CFA 模块
│   │   └── preference_loss.py        # CPO 模块
│   ├── config.py / data_utils.py
│   ├── train.py / evaluate.py / benchmark.py
│   ├── hallucination.py / ablation.py / sensitivity.py
│   ├── analyze.py / visualization.py
│   └── run_experiments.py
├── notebooks/
│   ├── run.ipynb          # 实验运行 (兼容 Colab)
│   └── preview.ipynb      # 模块可视化演示
├── data/ / results/
└── README.md / README_zh.md / EXPERIMENT_PLAN.md
```

---

## 参考文献

1. Liu & Lapata. "Hierarchical Transformers for Long Document Summarization." *EMNLP*, 2019.
2. Chuang et al. "DoLa: Decoding by Contrasting Layers Improves Factuality." *ICLR*, 2024.
3. Rafailov et al. "Direct Preference Optimization." *NeurIPS*, 2023.
4. Shi et al. "Context-Aware Decoding for Faithful Summarization." *ACL*, 2024.
5. Gao et al. "Model-based Preference Optimization in Summarization without Human Feedback." *EMNLP*, 2024.
6. Liu et al. "Lost in the Middle: How Language Models Use Long Contexts." *TACL*, 2023.
7. Lewis et al. "BART: Denoising Sequence-to-Sequence Pre-training." *ACL*, 2020.
8. Zhang et al. "PEGASUS: Pre-training with Extracted Gap-sentences." *ICML*, 2020.

---

## 许可证

MIT。预训练模型遵循 HuggingFace 各自许可证。

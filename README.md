<div align="center">
  <img src="https://github.com/datawhalechina/so-large-lm/logo.png" width="180px" alt="Project Logo" />
  
  # 🤖 大模型基础 (So-Large-LM)
  
  **从理论到实战，全面构建大模型知识体系**

  [![Datawhale](https://img.shields.io/badge/Datawhale-Community-green)](https://github.com/datawhalechina)
  [![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
  [![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/datawhalechina/so-large-lm/pulls)
  [![Stargazers](https://img.shields.io/github/stars/datawhalechina/so-large-lm)](https://github.com/datawhalechina/so-large-lm/stargazers)

  <p align="center">
    <a href="#-项目简介">项目简介</a> •
    <a href="#-精选资源">精选资源</a> •
    <a href="#-课程大纲">课程大纲</a> •
    <a href="#-学习路径">学习路径</a> •
    <a href="#-贡献与致谢">贡献与致谢</a>
  </p>
</div>

---

## 🚀 精选资源 (最新更新)

> 💡 **核心推荐**：配合视频与文档学习，效果更佳。

| 类型 | 内容 | 链接 |
| :--- | :--- | :--- |
| 📺 **视频** | **进击的AI：大模型技术全景 (第一节)** | [点击观看](https://www.bilibili.com/video/BV14x4y1x7bP/?spm_id_from=333.999.0.0&vd_source=4d086b5e84a56b9d46078e927713ffb0) |
| 📺 **视频** | **Llama开源家族：从Llama-1到Llama-3** | [点击观看](https://www.bilibili.com/video/BV1Xi421C7Ca/?share_source=copy_web&vd_source=df1bd9526052993d540dbd5f7938501f) |
| 📚 **文档** | **Llama开源家族技术详解** | [点击阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch14.md) |

---

## 📖 项目简介

**本项目致力于打造一个开源、系统、深入的大规模预训练语言模型（LLM）教程。**

项目以 [斯坦福 CS324](https://stanford-cs324.github.io/winter2022/) 和 [李宏毅生成式AI课程](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php) 为理论基石，结合开源社区的最新实践与前沿动态，涵盖从**数据准备、模型构建、训练策略**到**模型评估、安全伦理**的全链路知识。

### 🎯 适用人群
- 🎓 **学术/从业者**：希望深入了解 LLM 最新动态与技术细节的研究人员。
- 🏢 **行业探索者**：对 LLM 在医疗、金融、教育等领域应用感兴趣的专业人士。
- 🛠️ **开源贡献者**：不仅想学习，更想参与到 LLM 开源建设中的开发者。
- ⚖️ **相关从业者**：关注 AI 法律、伦理、版权及社会影响的跨领域专家。

---

## 🗺️ 学习路径

为了帮助初学者更高效地入门，Datawhale 构建了完整的 LLM 学习矩阵：

1.  **理论基石（本项目）**：`so-large-lm` —— 深入理解原理、架构与算法。
2.  **应用开发**：[`llm-universe`](https://github.com/datawhalechina/llm-universe) —— 快速入门 LLM 开发，搭建 Demo。
3.  **模型实战**：[`self-llm`](https://github.com/datawhalechina/self-llm) —— 基于 AutoDL 的开源模型部署与微调指南。

---

## 📚 课程大纲

### 第一部分：基础与架构
| 章节 | 内容亮点 | 链接 |
| :--- | :--- | :--- |
| **01. 引言** | 项目背景、GPT-3 崛起、LLM 发展简史 | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch01.md) |
| **02. 大模型的能力** | 迁移学习、In-context Learning、性能评估分析 | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch02.md) |
| **03. 模型架构** | Transformer 深度解析、位置编码、注意力机制 | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch03.md) |
| **04. 新的架构方向** | 混合专家模型 (MoE)、基于检索的模型 (RAG基础) | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch04.md) |

### 第二部分：数据与训练
| 章节 | 内容亮点 | 链接 |
| :--- | :--- | :--- |
| **05. 数据工程** | The Pile 数据集、数据清洗、分词策略 (Tokenization) | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch05.md) |
| **06. 模型训练** | 目标函数设计、优化算法选择 | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch06.md) |
| **07. 适配与微调** | Adaptation 必要性、PEFT (高效微调)、Probing | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch07.md) |
| **08. 分布式训练** | 数据并行、模型并行、流水线并行、混合策略 | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch08.md) |

### 第三部分：安全、伦理与前沿
| 章节 | 内容亮点 | 链接 |
| :--- | :--- | :--- |
| **09/10. 有害性分析** | 社会偏见、有毒信息检测、虚假信息 (Hallucination) | [上篇](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch09.md) / [下篇](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch10.md) |
| **11. 法律与伦理** | 版权法挑战、合理使用、司法案例汇总 | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch11.md) |
| **12. 环境影响** | 碳排放估算、绿色 AI | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch12.md) |
| **13. 智能体 (Agent)** | Agent 组件详解、挑战与机遇 | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch13.md) |
| **14. Llama 家族** | Llama 1-3 进化史、架构对比、生态复盘 | [阅读](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch14.md) |

---

## ✨ 核心贡献者

感谢所有为本项目付出心血的贡献者！

<table border="0">
  <tr>
    <td align="center"><a href="https://scholar.google.com/citations?user=tcb9VT8AAAAJ&hl=zh-CN"><img src="https://github.com/datawhalechina.png" width="80px;" alt=""/><br /><sub><b>陈安东</b></sub></a><br />(负责人/内容构建)</td>
    <td align="center"><a href="https://github.com/zhangfanTJU"><img src="https://avatars.githubusercontent.com/u/108520626?v=4" width="80px;" alt=""/><br /><sub><b>张帆</b></sub></a><br />(内容构建)</td>
     <td align="center"><a href="https://github.com/mlw67"><img src="https://avatars.githubusercontent.com/u/22756847?v=4" width="80px;" alt=""/><br /><sub><b>王茂霖</b></sub></a><br />(Issues维护)</td>
  </tr>
</table>

**项目负责人**: 陈安东 (ands691119@gmail.com)

---

## 📈 关注度趋势

[![Stargazers over time](https://starchart.cc/datawhalechina/so-large-lm.svg?variant=adaptive)](https://starchart.cc/datawhalechina/so-large-lm)

---
<div align="center">
  <b>🌟 如果这个项目对你有帮助，请给我们一个 Star！</b>
</div>

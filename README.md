# 大模型基础

已更新
>[Datawhale开源大模型入门课-第一节-进击的AI：大模型技术全景](https://www.bilibili.com/video/BV14x4y1x7bP/?spm_id_from=333.999.0.0&vd_source=4d086b5e84a56b9d46078e927713ffb0)
>
> [文字教程：Llama开源家族：从Llama-1到Llama-3](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch14.md)
> 
> [视频教程：Llama开源家族：从Llama-1到Llama-3](https://www.bilibili.com/video/BV1Xi421C7Ca/?share_source=copy_web&vd_source=df1bd9526052993d540dbd5f7938501f)

## 项目受众

1. 人工智能、自然语言处理和机器学习领域的研究者和从业者：该项目旨在为研究者和从业者提供大规模预训练语言模型的知识和技术，帮助他们更深入地了解当前领域的最新动态和研究进展。
2. 学术界和产业界对大型语言模型感兴趣的人士：项目内容涵盖了大型语言模型的各个方面，从数据准备、模型构建到训练和评估，以及安全、隐私和环境影响等方面。这有助于拓宽受众在这一领域的知识面，并加深对大型语言模型的理解。
3. 想要参与大规模语言模型开源项目的人士：本项目提供代码贡献和理论知识，降低受众在大规模预训练学习的门槛。
4. 其余大型语言模型相关行业人员：项目内容还涉及大型语言模型的法律和道德考虑，如版权法、合理使用、公平性等方面的分享，这有助于相关行业从业者更好地了解大型语言模型的相关问题。


## 项目简介

&emsp;&emsp;本项目旨在作为一个大规模预训练语言模型的教程，从数据准备、模型构建、训练策略到模型评估与改进，以及模型在安全、隐私、环境和法律道德方面的方面来提供开源知识。

&emsp;&emsp;项目将以[斯坦福大学大规模语言模型课程](https://stanford-cs324.github.io/winter2022/)和[李宏毅生成式AI课程](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php)为基础，结合来自开源贡献者的补充和完善，以及对前沿大模型知识的及时更新，为读者提供较为全面而深入的理论知识和实践方法。通过对模型构建、训练、评估与改进等方面的系统性讲解，以及代码的实战，我们希望建立一个具有广泛参考价值的项目。

&emsp;&emsp;我们的项目团队成员将分工负责各个章节的内容梳理和撰写，并预计在三个月内完成初始版本内容。随后，我们将持续根据社区贡献和反馈进行内容的更新和优化，以确保项目的持续发展和知识的时效性。我们期待通过这个项目，为大型语言模型研究领域贡献一份宝贵的资源，推动相关技术的快速发展和广泛应用。

## 项目意义

在当今时代，自然语言处理（NLP）领域以及其他人工智能（AI）的分支已经迎来了一场革命性的变革，这场变革的核心驱动力是大模型 (LLMs)的出现和发展。这些模型不仅构成了许多任务中最先进系统的基础，而且已经在医疗、金融、教育等众多行业展现出了前所未有的强大能力和应用潜力。

随着这些大模型在社会层面的影响力日益扩大，它们已经成为了公众讨论的焦点，激发了社会各界对人工智能发展趋势和潜在影响的深入思考与广泛兴趣。然而，尽管这一领域引人注目，相关的讨论和文章的质量却是参差不齐，缺乏系统性和深度，这不利于公众对于这一技术复杂性的真正理解。

正是基于这一现状，本教程的编写旨在填补这一空白，提供一套不仅易于理解但也理论丰富的大模型教程: 通过本教程，我们期望让广大群体不仅能够深刻理解大模型的原理和工作机制，而且能够掌握其在实际应用中的关键技术和方法，从而能够在这一领域内继续探索和创新。

特别是对于自然语言处理领域的初学者来说，面对以大模型为核心的各种新兴技术和知识，能够快速上手并有效学习是进入这一领域的关键。当前现有的自然语言处理教程在大模型内容的覆盖上仍显不足，这无疑增加了初学者的学习难度。因此，本教程从最基础的概念讲起，逐步深入，力求全面覆盖大模型的核心知识和技术要点，使读者能够从理论到实践都有深刻的理解和掌握。

> 关于实战的部分，欢迎学习同样是Datawhale出品的[self-llm开源课程](https://github.com/datawhalechina/self-llm)，该课程提供了一个全面实战指南，旨在通过AutoDL平台简化开源大模型的部署、使用和应用流程。从而使学生和研究者能够更高效地掌握环境配置、本地部署和模型微调等技能。在学习完大模型基础以及大模型部署后，关于Datawhale的大模型开发课程[llm-universe](https://github.com/datawhalechina/llm-universe)旨在帮助初学者最快、最便捷地入门 LLM 开发，理解 LLM 开发的一般流程，可以搭建出一个简单的 Demo。

**我们坚信，通过这样一套全面而深入的学习材料，能够极大地促进人们对自然语言处理和人工智能领域的兴趣和理解，进一步推动这一领域的健康发展和技术创新。**

## 项目亮点

1. 项目的及时性：当前大模型发展迅速，社会和学习者缺少较为全面和系统的大模型教程
2. 项目可持续性：当前大模型发展还在初期阶段，对行业的渗透还未全面展开，因此随着大模型的发展，该项目可持续的为学习者提供帮助

## 项目规划

**目录**
1. [引言](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch01.md)
    - 项目目标：目前对大规模预训练语言模型的相关知识的重点讲解
    - 项目背景：GPT-3等大型语言模型的出现，以及相关领域研究的发展
2. [大模型的能力](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch02.md)
    - 模型适应转换：大模型预训练往下游任务迁移
    - 模型性能评估：基于多个任务对GPT-3模型进行评估和分析
3. [模型架构](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch03.md)
    - 模型结构：研究和实现RNN, Transformer等网络结构
    - Transformer各层细节：从位置信息编码到注意力机制
4. [新的模型架构](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch04.md)
    - 混合专家模型（MoE）
    - 基于检索的模型
5. [大模型的数据](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch05.md)
    - 数据收集：从公开数据集中获取训练和评估所需数据，如The Pile数据集
    - 数据预处理：数据清洗、分词等
6. [模型训练](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch06.md)
    - 目标函数：大模型的训练方法
    - 优化算法：模型训练所使用的优化算法
7. [大模型之Adaptation](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch07.md)
    - 讨论为什么需要Adaptation
    - 当前主流的Adaptation方法（Probing/微调/高效微调） 
8. [分布式训练](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch08.md)
    - 为什么需要分布式训练
    - 常见的并行策略：数据并行、模型并行、流水线并行、混合并行
9. [大模型的有害性-上](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch09.md)
    - 模型性能差异：预训练或数据处理影响大模型性能
    - 社会偏见：模型表现出的显性的社会偏见
10. [大模型的有害性-下](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch10.md)
    - 模型有害信息：模型有毒信息的情况
    - 模型虚假信息：大模型的虚假信息情况
11. [大模型法律](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch11.md)
    - 新技术引发的司法挑战：司法随着新技术的出现而不断完善
    - 过去司法案例汇总：过去案例的汇总
12. [环境影响](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch12.md)
    - 了解大语言模型对环境的影响
    - 估算模型训练产生的排放量
13. [智能体（Agent）](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch13.md)
    - 了解Agent各组件细节
    - Agent的挑战与机遇
14. [Llama开源家族：从Llama-1到Llama-3](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch14.md)
    - Llama进化史（第1节）/ 模型架构（第2节）/训练数据（第3节）/训练方法（第4节）/效果对比（第5节）/社区生态（第6节）


## 核心贡献者

- [陈安东](https://scholar.google.com/citations?user=tcb9VT8AAAAJ&hl=zh-CN)：哈尔滨工业大学自然语言处理方向博士在读(发起人，项目负责人，项目内容构建)
- [张帆](https://github.com/zhangfanTJU)：天津大学自然语言处理方法硕士（项目内容构建）
  
### 参与者
- [王茂霖](https://github.com/mlw67)：华中科技大学博士在读 （解决issues问题）

## 项目负责人

陈安东 
联系方式: ands691119@gmail.com


## 感谢支持
[![Stargazers over time](https://starchart.cc/datawhalechina/so-large-lm.svg?variant=adaptive)](https://starchart.cc/datawhalechina/so-large-lm)

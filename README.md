# 生成大模型基础

## 项目简介

&emsp;&emsp;本项目旨在作为一个大规模预训练语言模型的教程，从数据准备、模型构建、训练策略到模型评估与改进，以及模型在安全、隐私、环境和法律道德方面的方面来提供开源知识。

&emsp;&emsp;项目将以[斯坦福大学大规模语言模型课程](https://stanford-cs324.github.io/winter2022/)和[李宏毅生成式AI课程](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php)为基础，结合来自开源贡献者的补充和完善，以及对前沿大模型知识的及时更新，为读者提供较为全面而深入的理论知识和实践方法。通过对模型构建、训练、评估与改进等方面的系统性讲解，以及代码的实战，我们希望建立一个具有广泛参考价值的项目。

&emsp;&emsp;我们的项目团队成员将分工负责各个章节的内容梳理和撰写，并预计在三个月内完成初始版本内容。随后，我们将持续根据社区贡献和反馈进行内容的更新和优化，以确保项目的持续发展和知识的时效性。我们期待通过这个项目，为大型语言模型研究领域贡献一份宝贵的资源，推动相关技术的快速发展和广泛应用。

## 立项理由

&emsp;&emsp;自然语言处理（NLP）领域以及其他人工智能领域已经被大规模预训练模型深刻改变。这些模型构成了许多任务中最先进系统的基础，并在各行各业迅速展现出强大的实力。

&emsp;&emsp;大模型在社会层面已经成为了一个热门话题，大众对此产生了浓厚兴趣。然而，目前关于这一领域的文章质量参差不齐。本教程旨在提供一套易于理解且理论丰富的大模型教程，让广大人群能够了解和学习。

&emsp;&emsp;从业界角度来看，未来自然语言处理领域的初学者可能会接触到以大模型为核心的各种知识，而现有的自然语言处理教程尚缺乏大模型相关的学习资料。因此，我们从全面的角度为大家提供大模型的学习内容。

&emsp;&emsp;此外，本教程借鉴了[斯坦福大学的CS324课程](https://stanford-cs324.github.io/winter2022/)和[李宏毅生成式AI课程](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php)旨在将优质且前沿的学术内容引入国内，为学习者提供更多资源。

## 项目受众

1. 人工智能、自然语言处理和机器学习领域的研究者和从业者：该项目旨在为研究者和从业者提供大规模预训练语言模型的知识和技术，帮助他们更深入地了解当前领域的最新动态和研究进展。
2. 学术界和产业界对大型语言模型感兴趣的人士：项目内容涵盖了大型语言模型的各个方面，从数据准备、模型构建到训练和评估，以及安全、隐私和环境影响等方面。这有助于拓宽受众在这一领域的知识面，并加深对大型语言模型的理解。
3. 想要参与大规模语言模型开源项目的人士：本项目提供代码贡献和理论知识，降低受众在大规模预训练学习的门槛。
4. 其余大型语言模型相关行业人员：项目内容还涉及大型语言模型的法律和道德考虑，如版权法、合理使用、公平性等方面的分享，这有助于相关行业从业者更好地了解大型语言模型的相关问题。

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
13. [全面释放ChatGPT的无限潜能 ‐ 实用技巧、创新应用与前沿探索](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch13.md)
14. [探索深层式人工智能的无限潜力:从"工具人"到创造伙伴](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch14.md)

**负责人**

- [陈安东](https://github.com/andongBlue)：
- [张帆](https://github.com/zhangfanTJU)：
- [王茂霖](https://github.com/mlw67)


**各章节预估完成日期**

整体教程开源内容发布第二版分为三个步骤：
- Step 1: 基于原有第一版的内容内容进行整体的润色和优化【预计两个月结束】；
- Step 2: 在上一步的基础上，加入代码模块增加内容的实用性；
- Step 3: 对前沿的大模型内容进行补充。

## 项目负责人

陈安东 
微信: andong---

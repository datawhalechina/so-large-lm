# Foundation of Large Models

Updated
>[Datawhale Open Source Large Model Introduction Course - Lesson 1 - Advancing AI: A Panorama of Large Model Technology](https://www.bilibili.com/video/BV14x4y1x7bP/?spm_id_from=333.999.0.0&vd_source=4d086b5e84a56b9d46078e927713ffb0)

> [Text Tutorial: Llama Open Source Family: From Llama-1 to Llama-3](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch14.md)

> [Video Tutorial: Llama Open Source Family: From Llama-1 to Llama-3](https://www.bilibili.com/video/BV1Xi421C7Ca/?share_source=copy_web&vd_source=df1bd9526052993d540dbd5f7938501f)

## Project Audience

1. **Researchers and professionals in AI, Natural Language Processing, and Machine Learning**: This project aims to provide knowledge and techniques related to large-scale pre-trained language models, helping them gain deeper insights into the latest developments and research progress in the field.
2. **Individuals interested in large language models from both academia and industry**: The content covers various aspects of large language models, including data preparation, model construction, training, evaluation, and aspects like security, privacy, and environmental impacts. This broadens the audience's understanding of the field.
3. **Those wishing to participate in open-source large language model projects**: The project offers both code contributions and theoretical knowledge, lowering the entry barrier for those interested in large-scale pre-training learning.
4. **Professionals in industries related to large language models**: The project also discusses legal and ethical considerations of large language models, such as copyright law, fair use, and fairness, helping industry professionals better understand related issues.

## Project Introduction

This project aims to serve as a tutorial on large-scale pre-trained language models, offering open-source knowledge on data preparation, model construction, training strategies, model evaluation and improvement, and the ethical, privacy, environmental, and legal aspects of these models.

The project is based on the [Stanford University Large Language Models Course](https://stanford-cs324.github.io/winter2022/) and [Li Hongyi's Generative AI Course](https://speech.ee.ntu.edu.tw/~hylee/genai/2024-spring.php), supplemented by contributions from open-source contributors and timely updates on cutting-edge model knowledge, providing readers with comprehensive theoretical knowledge and practical methods. Through systematic explanations of model construction, training, evaluation, improvement, and practical coding, we aim to build a project with wide reference value.

Our project team members will be responsible for organizing and writing the content for each chapter, with the initial version set to be completed in three months. Afterward, we will continue to update and optimize the content based on community contributions and feedback to ensure the project's ongoing development and the timeliness of knowledge. We look forward to contributing a valuable resource to the field of large language model research and promoting the rapid development and widespread application of related technologies.

## Project Significance

In today’s world, the field of Natural Language Processing (NLP) and other branches of Artificial Intelligence (AI) have undergone a revolutionary transformation, driven by the emergence and development of large models (LLMs). These models not only form the foundation of state-of-the-art systems for many tasks but have already demonstrated unprecedented power and application potential in industries like healthcare, finance, and education.

As these large models increasingly impact society, they have become a focal point of public discussion, sparking deeper reflection and widespread interest in the development trends and potential impacts of artificial intelligence. However, despite the attention in this field, discussions and articles lack consistency and depth, making it difficult for the public to truly understand the complexity of this technology.

Given this situation, the purpose of this tutorial is to fill the gap by providing a set of large model tutorials that are both easy to understand and theoretically rich. Through this tutorial, we aim to help a wide audience not only deeply understand the principles and workings of large models but also master the key techniques and methods for practical applications, enabling them to continue exploring and innovating in this field.

Especially for beginners in the field of NLP, being able to quickly grasp and effectively learn emerging technologies and knowledge centered around large models is key to entering the field. Existing NLP tutorials still lack coverage of large models, which undoubtedly increases the learning difficulty for beginners. Therefore, this tutorial starts from the most basic concepts and gradually deepens, aiming to comprehensively cover the core knowledge and technical points of large models, ensuring readers develop both a deep theoretical understanding and practical skills.

> For practical application, we recommend the [self-llm open-source course](https://github.com/datawhalechina/self-llm), also produced by Datawhale, which provides a comprehensive practical guide to simplifying the deployment, use, and application processes of open-source large models via the AutoDL platform. This will help students and researchers more efficiently master skills such as environment configuration, local deployment, and model fine-tuning. After studying the basics and deployment of large models, the [llm-universe course](https://github.com/datawhalechina/llm-universe) by Datawhale will help beginners quickly and easily enter LLM development, understanding the general process and enabling the creation of simple demos.

**We firmly believe that this comprehensive and in-depth set of learning materials will greatly promote interest and understanding in Natural Language Processing and AI, further advancing the healthy development and technological innovation of this field.**

## Project Highlights

1. **Timeliness of the Project**: Large models are developing rapidly, and there is a lack of comprehensive and systematic tutorials on large models for society and learners.
2. **Sustainability of the Project**: The development of large models is still in its early stages, and their penetration into the industry is not yet fully widespread. Therefore, as large models continue to develop, this project will provide sustained support to learners.

## Project Plan

**Table of Contents**
1. [Introduction](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch01.md)
    - Project Goals: Focused explanation of knowledge on large-scale pre-trained language models.
    - Project Background: Emergence of large language models such as GPT-3 and the development of related research fields.
2. [Capabilities of Large Models](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch02.md)
    - Model Adaptation: Transition of large model pre-training to downstream tasks.
    - Model Performance Evaluation: Evaluation and analysis of GPT-3 across multiple tasks.
3. [Model Architecture](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch03.md)
    - Model Structures: Research and implementation of RNN, Transformer, and other network architectures.
    - Details of Transformer Layers: From positional encoding to attention mechanisms.
4. [New Model Architectures](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch04.md)
    - Mixture of Experts (MoE)
    - Retrieval-based Models
5. [Data for Large Models](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch05.md)
    - Data Collection: Gathering training and evaluation data from public datasets like The Pile.
    - Data Preprocessing: Data cleaning, tokenization, etc.
6. [Model Training](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch06.md)
    - Objective Functions: Training methods for large models.
    - Optimization Algorithms: Optimization algorithms used for model training.
7. [Adaptation for Large Models](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch07.md)
    - Why Adaptation is Needed.
    - Mainstream Adaptation Methods (Probing/Fine-tuning/Efficient Fine-tuning)
8. [Distributed Training](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch08.md)
    - Why Distributed Training is Necessary.
    - Common Parallel Strategies: Data Parallelism, Model Parallelism, Pipeline Parallelism, Hybrid Parallelism.
9. [Harmful Effects of Large Models - Part 1](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch09.md)
    - Performance Differences: How pre-training or data processing affects large model performance.
    - Societal Bias: Explicit societal biases exhibited by models.
10. [Harmful Effects of Large Models - Part 2](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch10.md)
    - Harmful Information: Toxic information produced by models.
    - Misinformation: False information generated by large models.
11. [Legal Aspects of Large Models](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch11.md)
    - Legal Challenges Arising from New Technology: Legal improvements in response to new technologies.
    - Summary of Past Legal Cases: A summary of past legal cases.
12. [Environmental Impact](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch12.md)
    - Understanding the Environmental Impact of Large Language Models.
    - Estimating Emissions from Model Training.
13. [Agents](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch13.md)
    - Understanding the Components of Agents.
    - Challenges and Opportunities of Agents.
14. [Llama Open Source Family: From Llama-1 to Llama-3](https://github.com/datawhalechina/so-large-lm/blob/main/docs/content/ch

14.md)
    - Introduction to Llama and open-source models.
    - How to run and fine-tune Llama models.

## Future Development

1. **Open-Source Contributions**: The project will provide an open-source community for collaborative contributions, improving the quality of the content with community feedback and updates.
2. **Further Research**: Based on the development of large models, we will keep adding relevant updates on new models, techniques, and methods.
3. **Practical Demos**: Through future developments, we will work on creating more practical demos and implementations for learners.

## Key Points

1. **Start from basics**: Beginners can easily pick up the concepts.
2. **Target advanced learners**: While providing beginner-friendly content, advanced topics and cutting-edge research are also incorporated for professionals.
3. **Comprehensive resource**: Covers both theoretical and practical aspects of large models, providing learners with a complete learning path.
4. **Sustainability**: Updates and content will keep evolving with the field.

---

# **Core Contributors**  

- [**Chen Andong**](https://scholar.google.com/citations?user=tcb9VT8AAAAJ&hl=zh-CN): Ph.D. candidate in Natural Language Processing at Harbin Institute of Technology (Initiator, Project Lead, Content Development)  
- [**Zhang Fan**](https://github.com/zhangfanTJU): Master's in Natural Language Processing from Tianjin University (Content Development)  

### **Participants**  
- [**Wang Maolin**](https://github.com/mlw67): Ph.D. candidate at Huazhong University of Science and Technology (Issue Resolution)  

## **Project Lead**  

**Chen Andong**  
Contact: ands691119@gmail.com  

## **Thank You for Your Support**  

[![Stargazers over time](https://starchart.cc/datawhalechina/so-large-lm.svg?variant=adaptive)](https://starchart.cc/datawhalechina/so-large-lm)  

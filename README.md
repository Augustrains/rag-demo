# 📘 模块化 RAG Demo

一个基于 LangChain 构建的模块化 Retrieval-Augmented Generation（RAG）示例项目，实现了：

* ✅ Baseline RAG（基础检索增强生成）
* 🔍 Multi-Query RAG（多查询召回增强）
* 🧭 基于 LLM 的多知识库 Routing
* ⚙️ 可扩展的模块化架构设计

本项目重点探索：

* 检索策略对系统性能的影响
* Recall / Precision 的权衡
* 向量空间行为分析
* 实际 RAG 系统的工程实现

---

# 🚀 功能特性

## 1️⃣ Baseline RAG

标准 RAG 流程：

```
用户问题 → 向量检索 → 拼接上下文 → LLM 生成答案
```

核心组件：

* FAISS 向量数据库
* RecursiveCharacterTextSplitter 文本切分
* HuggingFace Embedding（all-MiniLM-L6-v2）
* DeepSeek Chat 模型

---

## 2️⃣ Multi-Query RAG（召回增强）

通过生成多个语义相近的问题来提高召回率：

* 使用 LLM 生成多个改写问题
* 对每个问题进行独立检索
* 合并结果并去重
* 再交给 LLM 生成最终答案

### 为什么可以提升 Recall？

不同表述方式的 query 在 embedding 空间中可能落在不同区域。

Multi-query 等价于：

> 在 embedding 空间进行多方向搜索

从而提高覆盖率，提升召回率。

### Trade-off

| 优点        | 代价          |
| --------- | ----------- |
| 提高 recall | 增加 token 消耗 |
| 覆盖更多语义    | 噪声可能增加      |
| 提高鲁棒性     | 推理延迟增加      |

---

## 3️⃣ LLM Routing（多知识库路由）

支持多个领域知识库：

* 🤖 Agent 相关（ReAct、Task Decomposition、Reflection）
* 📐 动态规划（DP 教程与算法题）
* 📊 微积分（极限、导数、积分）

工作流程：

```
用户问题 → LLM 路由判断 → 选择对应知识库 → 检索 → 生成答案
```

### Routing 的优势

* 减少搜索空间
* 降低噪声
* 提高检索效率
* 支持领域扩展
* 支持不同检索策略

本质是：

> Coarse-grained 分类 → Fine-grained 检索

---

# 🏗 项目结构

```
.
├── advanced_rag.py        # 主程序（包含 baseline / multi-query / routing）
├── baseline.ipynb         # Baseline Notebook 演示
├── .env.example           # 环境变量模板
├── .gitignore
└── README.md
```

---

# ⚙️ 安装方法

## 1️⃣ 克隆仓库

```
git clone https://github.com/<你的用户名>/rag-demo.git
cd rag-demo
```

## 2️⃣ 安装依赖

如果没有 requirements.txt，可以手动安装：

```
pip install langchain langchain-community langchain-core \
langchain-classic langchain-text-splitters \
langchain-huggingface langchain-deepseek \
sentence-transformers faiss-cpu python-dotenv bs4
```

---

# 🔐 环境变量配置

创建 `.env` 文件：

```
DEEPSEEK_API_KEY=你的_deepseek_key
LANGCHAIN_API_KEY=你的_langsmith_key   # 可选
```

⚠️ 请勿将 `.env` 提交到 GitHub。

---

# ▶️ 运行方式

通过命令行运行：

```
python advanced_rag.py --task baserag
```

## 支持的模式

| 参数       | 说明      |
| -------- | ------- |
| baserag  | 基础 RAG  |
| multirag | 多查询召回增强 |
| routing  | 多知识库路由  |

示例：

```
python advanced_rag.py --task multirag --query "What is dynamic programming?"
```

---

# 🧠 设计思考

## 🔹 Chunk Size 的 Trade-off

| chunk 较大    | chunk 较小    |
| ----------- | ----------- |
| ↑ Recall    | ↓ Recall    |
| ↓ Precision | ↑ Precision |
| ↑ Token 消耗  | ↓ Token 消耗  |

最优 chunk_size 取决于：

* 文档结构
* 语义边界
* 模型上下文窗口
* 噪声容忍度

---

## 🔹 Multi-Query vs Baseline

| Baseline        | Multi-Query      |
| --------------- | ---------------- |
| 单向 embedding 检索 | 多方向 embedding 搜索 |
| 成本低             | 召回高              |
| 简单快速            | 更鲁棒              |

---

## 🔹 Routing vs 单知识库

Routing 解决的是：

* 搜索空间膨胀问题
* 噪声比例增加问题
* 多领域扩展问题

属于典型的：

> 分层检索架构设计

---

# 🧩 可扩展方向

本项目架构支持扩展：

* 🔁 Reranker（交叉编码器重排序）
* 🛠 工具调用（Tool Use）
* 🌐 Web Search
* 🔄 Corrective RAG
* 🤖 Agent 工作流
* 📊 评估指标（Recall@K / Accuracy）

---

# 🎯 项目目标

本项目旨在探索：

* RAG 检索策略差异
* Recall / Precision 权衡
* 向量空间的语义分布问题
* 模块化 RAG 系统设计
* LLM 工程实践能力

---

# 📌 后续优化方向

* 引入 Hybrid Retrieval（BM25 + 向量）
* 增加异步并发支持
* 增加缓存机制
* 加入评测框架
* 支持多模型对比

---



你想优化成哪种方向？

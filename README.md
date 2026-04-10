# ⚖️ Smart Legal Core - Enterprise RAG Contract Review System

<p align="left">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" />
  <img src="https://img.shields.io/badge/LLM-Qwen2.5--14B-orange.svg" />
  <img src="https://img.shields.io/badge/VectorDB-ChromaDB-red.svg" />
  <img src="https://img.shields.io/badge/Embedding-BGE--M3-green.svg" />
</p>

## 📖 專案簡介 | Introduction
**Smart Legal Core** 是一套基於 **RAG (Retrieval-Augmented Generation)** 架構的智能合約審查系統。本專案旨在解決企業法務在處理大量草稿時的效率痛點，透過地端 LLM 結合向量資料庫，實現精準的風險偵測、合約生成與報價分析。

**Smart Legal Core** is an intelligent contract review system powered by **RAG (Retrieval-Augmented Generation)**. Designed to streamline corporate legal workflows, it leverages local LLMs and Vector Databases to provide precise risk detection, automated contract generation, and historical price analysis.

---

## 🚀 核心功能 | Key Features

### 1. 智能動態審查 | Dynamic Semantic Review (`/review`)
- [cite_start]**語義對標**：自動將草稿切分為條文，並與企業標準模板進行語義比對 [cite: 76]。
- [cite_start]**風險標註**：精準識別「重大衝突」、「一般偏離」與「條款缺漏」，並提供修正建議 [cite: 17, 19]。
- [cite_start]**Semantic Mapping**: Automatically segments drafts into articles and compares them against corporate standard templates[cite: 76].
- [cite_start]**Risk Labeling**: Identifies "Critical Conflicts," "General Deviations," and "Missing Clauses" with actionable suggestions[cite: 17, 19].

### 2. 商業風險預警 | Business Price Risk Analysis (`/risk`)
- **歷史比對**：串接資料庫分析廠商歷史報價，當前報價若超出均價一定比例將發出紅色警示。
- **Historical Benchmarking**: Analyzes vendor pricing history and triggers alerts if the current quote significantly exceeds the historical average.

### 3. 自動化合約生成 | Automated Contract Generation (`/generate`)
- **意圖解析**：透過 LLM 解析使用者需求，自動調用對應的 DOCX 模板並填入變數。
- **Intent Parsing**: Uses LLM to parse user requests and automatically populates variables into the most relevant DOCX template.

---

## 🏗️ 技術深度 | Technical Depth

### 🧩 專用語義切塊演算法 | Legal-Specific Chunking
為確保法條語義完整，系統開發了「條文導向切塊」機制，優先識別「第 X 條」作為物理分割點，大幅降低 RAG 檢索時的語義斷裂問題。
To ensure legal context integrity, the system utilizes a "Clause-Oriented Chunking" mechanism that prioritizes "Article X" markers as split points, significantly reducing context loss during retrieval.

### 🛡️ AI 安全防護層 | Prompt Injection Shield
內建資安攔截邏輯，偵測並過濾合約草稿中可能包含的惡意指令，防止外部文本干擾系統審查準則。
Built-in security logic to detect and filter malicious instructions within contract drafts, preventing external text from overriding the system's review guidelines.

### 🏠 地端隱私部署 | Privacy-First Local Deployment
整合 **Ollama (Qwen2.5)** 與 **BGE-M3** 模型，支援全地端運行，確保敏感合約數據不流向公有雲。
Integrated with **Ollama (Qwen2.5)** and **BGE-M3**, supporting fully local execution to ensure sensitive contract data never leaves the corporate intranet.

---

## 🛠️ 技術棧 | Tech Stack
- **Language**: Python 3.9+
- **LLM Engine**: Ollama (Qwen2.5:14B)
- **Embedding Model**: BGE-M3
- **Vector Database**: ChromaDB
- **Relational Database**: SQLite
- **UI Framework**: Streamlit

---

## 📫 聯絡我 | Contact
- [cite_start]**Name**: Sheng-Lun Lin (林聖倫) [cite: 63]
- [cite_start]**Email**: q26221922@gmail.com [cite: 47]
- [cite_start]**University**: National Chengchi University (NCCU) [cite: 50]

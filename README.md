# 🧠 NeuroCV – AI-Based Resume Evaluator & Job Matcher

![NeuroCV Banner](https://img.shields.io/badge/AI-Powered-blue) ![License](https://img.shields.io/github/license/aky-ds/NeuroCV) ![Python](https://img.shields.io/badge/Made%20with-Python-blue)

## 💡 Problem with Traditional Resume Screening

Recruiters and hiring managers often:
- Spend hours manually going through resumes.
- Miss top candidates due to lack of keyword alignment.
- Suffer from biased or inconsistent decision-making.
- Struggle to match resumes to specific job descriptions effectively.

---

## 🚀 What is NeuroCV?

**NeuroCV** is an AI-powered application that evaluates resumes by:
- Scoring them against job descriptions using **Natural Language Processing** and **Sentence Embeddings**.
- Highlighting compatibility based on **Education**, **Experience**, and **Certifications**.
- Generating a **Total Matching Score** with detailed explanation.
- Built with **Groq API**, **LLaMA-3**, **LangChain**, and **Flask**.

---

## 🔍 Features

- ✅ Resume Parsing & Cleaning  
- ✅ JD Matching using **LLMs (LLaMA-3)** via Groq  
- ✅ Scoring System for **Education**, **Experience**, **Certifications**  
- ✅ Clean UI with Flask backend  
- ✅ Supports multiple resumes and JDs

---

## 🛠️ Tech Stack

| Component           | Technology                  |
|---------------------|-----------------------------|
| Language            | Python                      |
| Backend             | Flask                       |
| AI/LLM              | LLaMA-3 via Groq + LangChain|
| NLP Embedding       | Sentence Transformers       |
| Vector Store        | FAISS                       |
| File Handling       | PyMuPDF, PyPDF2             |
| Prompt Engineering  | Custom LLM Templates        |

---

## 📦 Installation

```bash
git clone https://github.com/aky-ds/NeuroCV.git
cd NeuroCV
pip install -r requirements.txt

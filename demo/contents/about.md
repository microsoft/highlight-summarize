# Highlight, then Summarize
## RAG Without the Jailbreaks

> **A new design pattern for secure, high-quality LLM-based question answering.**

---

### 🚀 What is H&S?

**Highlight & Summarize (H&S)** is a novel enhancement to Retrieval-Augmented Generation (RAG) systems that **prevents jailbreaks by design**. Instead of letting users directly influence the LLM's output, H&S introduces a two-step pipeline:

1. **Highlight**: Extract relevant passages from trusted documents.
2. **Summarize**: Use an LLM to generate an answer *without ever seeing the user's question*.

This simple shift dramatically improves **security**, **robustness**, and often even **answer quality**.

📢 **Head over to the [chat](/Chat) to play with a H&S Chatbot, which can answers your questions about H&S!**

---

### 🧠 Why It Matters

Traditional RAG systems are vulnerable to:
- 🧨 **Jailbreaking**: Users tricking the LLM into generating harmful or off-topic content.
- 🛠️ **Model Hijacking**: Repurposing the system for unintended tasks.
- 💬 **Misrepresentation**: Generating outputs that could be mistaken as official statements.

**H&S eliminates these risks** by ensuring the LLM never sees untrusted input.

---

### 🛠️ How It Works

{hs_image}


### 📊 Key Results

- ✅ **Better Accuracy**: Outperforms vanilla RAG in correctness and quality.
- 🔐 **Zero Jailbreaks**: Immune to known prompt injection attacks.
- 🧪 **Flexible Implementations**: Works with both extractive and generative highlighters.

| Pipeline                | Correctness (RepliQA) | Jailbreak Resistance | Avg. Response Quality |
|------------------------|------------------------|----------------------|------------------------|
| H&S Structured         | ⭐ Highest              | ✅ 100% Secure        | ⭐ Top-rated            |
| Vanilla RAG            | Moderate               | ❌ Vulnerable         | Moderate               |
| H&S DeBERTaV3 (RepliQA)| Good                   | ✅ 100% Secure        | Good                   |

---

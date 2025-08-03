# 🧠 ConsultIQ – PwC-Style AI Knowledge Agent

ConsultIQ is an AI-powered document question-answering app, built for consulting scenarios like PwC. It uses advanced multi-agent orchestration with LLMs to answer complex business questions by reading internal documents, summarizing them, and generating business strategies.

---

## 💡 How It Works

You upload internal strategy PDFs → AI agents scan + understand them → You ask a question → Agents collaborate (search, summarize, strategize) → You get a clear business-style answer with document sources.

---

## 🛠️ Features

- 🔎 Vector-based document search
- 🧠 Simple QA using Retrieval-Augmented Generation
- 🤖 Multi-agent pipeline using LangChain Agents
- 📄 Streamlit interface
- 📚 Sources linked from results

---

## 🧰 Tech Stack

- Python
- Streamlit
- LangChain
- OpenAI (text-embedding-3-large + GPT-4 or GPT-3.5)
- Chroma DB (for local vector storage)

---

## 🚀 Getting Started

### 1. Clone the Repo

```bash
git clone https://github.com/Rushikesh1234/ConsultIQ-AI.git
cd ConsultIQ-AI
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> If `requirements.txt` is missing, install manually:

```bash
pip install streamlit langchain openai chromadb tiktoken python-dotenv
```

### 3. Add Your OpenAI API Key

Create a `.env` file:

```env
OPENAI_API_KEY=your-key-here
```

---

## ▶️ Running the App

```bash
streamlit run app.py
```

Then open: [http://localhost:8501](http://localhost:8501)

---

## 🧪 How to Use

1. **Upload your PDF docs** (currently configured to pre-indexed ones)
2. **Ask your question** using:
   - 🧠 Simple QA (Retrieval)
   - 🤖 Agent Mode (multi-step strategy)
3. **View generated answers + sources**
4. **Explore top matching documents**

---

## 📁 Project Structure

```
├── app.py                  # Streamlit frontend app
├── agent_tools.py          # Tools used by the multi-agent pipeline
├── model_prompt.py         # Custom prompt for RAG
├── data_ingestor.py        # Data Ingestor to for converting PDFs docuements to embedding vedtors
├── Chroma_Indexes/         # Saved vector DB
├── static/                 # PDF viewer support
└── .env                    # API Key (excluded from version control)
```

---

## 📌 Example Use Cases

- "What’s PwC’s strategy for automotive sector?"
- "Summarize M&A guidelines from the internal policy doc"
- "Generate a client-ready strategy based on these regulations"

---

## ▶️ Demo Video
[ConsultIQ Demo](https://github.com/user-attachments/assets/f72b6b26-59ef-485f-9af1-7efe5a5c8339)

---

## 🖥️ Screenshots
![ConsultIQ Screenshots](/demo/ss2.png)

---

## ▶️ Multi-Agent Demo Video
![Watch the video](/demo/multi-agentmodel.gif)

---

## 🤝 Contribution & Collaboration

This is a learning-focused AI PoC project. PRs welcome!

---

## ⚠️ Disclaimer

For demo and educational purposes only. Not affiliated with PwC. Do not use with confidential data.

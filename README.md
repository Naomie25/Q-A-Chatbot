# Final-Project-Bootcamp
# 📊 Q&A Chatbot for Public Datasets

An intelligent assistant capable of automatically answering questions about open public datasets.  
This project combines data cleaning, vector-based retrieval (FAISS), NLP with pre-trained models (GPT/BERT), and a user-friendly Gradio interface.

---

## 🎯 Project Goal

The goal of this project is to build a natural language interface that allows users to ask questions about real-world public datasets (e.g., health, environment, demographics) and get accurate, understandable answers — powered by a combination of machine learning and retrieval-augmented generation (RAG).

---

## 🧱 Project Structure

```
project/
├── data/                  # Public datasets (CSV format)
│   ├── health.csv
│   └── emissions.csv
├── faiss_index/           # FAISS vector index file
│   └── vector_store.faiss
├── app/
│   ├── chatbot.py         # Core chatbot logic (RAG)
│   ├── rag_utils.py       # Utility functions (embedding, search, loading)
│   └── interface.py       # Gradio interface
├── notebooks/
│   └── data_cleaning.ipynb
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── main.py                # Script to launch the app
```

---

## 🗂️ Public Datasets Used

- **WHO Global Health Indicators**
- **World CO₂ Emissions Dataset**
- **World Bank Development Indicators**

> All datasets are open access, publicly available from sources like [Our World In Data](https://ourworldindata.org), [data.worldbank.org](https://data.worldbank.org), and [datahub.io](https://datahub.io).

---

## ⚙️ How It Works

1. **Data Cleaning & Preparation**
   - Standardize and clean CSV datasets
   - Convert data rows into readable text chunks

2. **Semantic Indexing with Embeddings**
   - Use `sentence-transformers` to generate vector embeddings
   - Store in a FAISS index for fast retrieval

3. **Question Answering**
   - When a user asks a question:
     - Embed the question
     - Retrieve relevant data chunks
     - Feed those into GPT to generate an answer

4. **User Interface**
   - Simple Gradio app where users can type questions and get answers
   - Retrieved context is shown for transparency

---

## 🧪 Example Questions

- "What was the life expectancy in France in 2020?"
- "Which countries had the highest CO₂ emissions in 2015?"
- "How has life expectancy changed in Africa since 1990?"

---

## 🚀 How to Run the Project

### 1. Clone the repository
```bash
git clone https://github.com/your-username/public-dataset-chatbot.git
cd public-dataset-chatbot
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch the chatbot app
```bash
python main.py
```

---

## 🧰 Tech Stack

| Purpose                     | Tools/Libraries                   |
|-----------------------------|------------------------------------|
| Data manipulation           | `pandas`, `numpy`                  |
| Data visualization          | `seaborn`, `matplotlib`            |
| NLP embeddings              | `sentence-transformers`            |
| Vector search               | `faiss`                            |
| LLM for answering           | `openai`, `transformers`           |
| User interface              | `gradio`                           |

---

## ⚖️ Ethical Considerations

- **Data Bias**: Some countries may have incomplete or unreliable data
- **LLM Hallucinations**: GPT may generate incorrect answers even with context
- **Transparency**: Retrieved data is displayed to help users verify answers
- **Explainability**: The model explains its answers based on the input context

---

## 💡 Future Improvements

- Add support for more datasets with automatic loading
- Use ChromaDB or Pinecone as scalable vector DB
- Enable summarization of trends using GPT
- Multilingual support for questions and answers

---

## 📜 License

This is an academic project under the MIT License.  
All data is licensed under the terms of their original sources (WHO, World Bank, etc.).

---

## 👤 Author

**Naomie Marciano**  
Built as part of a project with OMNILab & BaiYuLan Open AI Community

---

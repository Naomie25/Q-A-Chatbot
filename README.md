# 📊 Public Data QA Chatbot + Analytics

An interactive **Q&A chatbot** and **data exploration tool** built with Python, Pandas, Gradio, and World Bank Open Data.  
It allows you to **ask questions** about GDP, life expectancy, and literacy rates for any country, and also **explore trends** with charts, descriptive statistics, and forecasts.

---
🎯 Project Goal
The goal of this project is to create an interactive data exploration and question-answering tool that allows users to easily retrieve and analyze GDP, life expectancy, and literacy rate data for countries worldwide.

It combines:

Automated data retrieval from the World Bank API

Natural language question parsing for quick answers

Statistical analysis and forecasting for deeper insights

Interactive visualizations to explore historical trends

The system aims to make global socio-economic data more accessible, helping students, researchers, and analysts quickly obtain insights without manually processing datasets.

---
## 🚀 Features

### **1. Data Loading & Caching**
- Automatically fetches **World Bank datasets**:
  - **Literacy Rate** (`SE.ADT.LITR.ZS`)
  - **Life Expectancy** (`SP.DYN.LE00.IN`)
  - **GDP** (`NY.GDP.MKTP.CD`)
- Cleans and merges data into a unified format.
- Saves results in a **SQLite database** for faster reuse.

### **2. Question Answering**
- Supports natural language queries like:
  ```
  GDP of France in 2020
  Life expectancy in Japan in 2015
  Literacy rate in India in 2018
  ```
- Looks up answers from the local cache.
- Falls back to **live World Bank API** if data is missing locally.
- As a last resort, can use a **Puppeteer-based web fallback** (if running).

### **3. Analytics & Statistics**
- Descriptive statistics (count, mean, std, min, quartiles, max).
- Pearson correlation between metrics.
- **t-tests** between countries.
- Simple **1-year linear regression forecasts**.

### **4. Visualization**
- Interactive **time series charts** for GDP, life expectancy, and literacy rate.
- Custom start/end years.
- Optional **forecast overlay**.

### **5. Gradio Web Interface**
- **Ask** mode: free-text Q&A.
- **Explore** mode: dropdown selection for metric/country, charts, stats, and predictions.

---

## 🛠 Installation

### 1. Clone this repository
```bash
git clone https://github.com/yourusername/public-data-qa.git
cd public-data-qa
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

**`requirements.txt`** should contain:
```
pandas
numpy
matplotlib
scipy
scikit-learn
gradio
requests
```

---

## ▶️ Usage

### **Run the app**
```bash
python qa_engine.py
```

The interface will start locally at:
```
http://0.0.0.0:7860
```

---

## 💬 Example Questions

**In Ask mode:**
- `GDP of France in 2020`
- `Life expectancy in Japan in 2015`
- `Literacy rate in India in 2018`
- `Correlation between GDP and life expectancy in Germany`
- `Predict GDP of Canada in 2022`

**In Explore mode:**
- Choose a **metric** (GDP / life expectancy / literacy rate).
- Choose a **country**.
- Set optional **start** and **end years**.
- View **charts**, **stats**, and **forecasts**.

---

## 📂 Project Structure
```
qa_engine.py       # Main application code
public_data.db     # SQLite cache (auto-generated)
requirements.txt   # Dependencies
```

---

## 🔌 Optional Puppeteer Fallback
If you have a Puppeteer-based web scraper running locally (Node.js API at `http://localhost:3000/ask`),  
the chatbot will use it for **general web queries** when the World Bank API cannot answer.

---

## 📡 Data Source
All metrics are from the **World Bank Open Data API**:
- https://data.worldbank.org/

---

## ⚠️ Notes & Limitations
- Only **GDP**, **life expectancy**, and **literacy rate** are supported for now.
- Live lookups require **internet access**.
- Puppeteer fallback is optional and **disabled by default** unless you run the API locally.

---

## 📜 License
This project is released under the **MIT License**.

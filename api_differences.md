# Fundamental Differences

**World Bank API** → Data updated in real-time through an HTTP request
(JSON format).

**Kaggle API** → Only provides datasets already published by users or
Kaggle itself. The data is **not** automatically updated like with the
World Bank.

On Kaggle, you must first download the dataset (CSV, XLSX, etc.) and
then import it locally with Pandas. The API does not allow direct
"querying" like a live service.

##  Puppeteer Integration

Originally, the project included a **Puppeteer fallback** to fetch answers from the web whenever the API or local database did not provide a result.  

However, due to technical limitations, establishing a stable connection with Puppeteer was not possible in this setup.  
This means that the **Puppeteer module is present in the code, but not functional**.  

 Instead, all answers are provided directly through:
- The **World Bank API**  
- The **local SQLite database** (with preloaded datasets)  
- The **statistical and machine learning modules** (for analysis and forecasting)  

Therefore, Puppeteer is considered an **optional / non-functional feature** in the current version of the project.


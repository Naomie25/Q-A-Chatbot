# qa_engine.py

# - Loads World Bank datasets (literacy, life expectancy, GDP)
# - Cleans & merges, saves to SQLite GOOD
# - Provides:
#     * direct pandas lookup for questions GOOD 
#     * descriptive stats, correlation, t-test
#     * simple ML forecast (linear regression)
#     * Puppeteer fallback via http://localhost:3000/ask?q=... DOESN'T WORK!!!!!!!
# - Gradio UI with Ask and Explore modes

import os
import re
import sqlite3
import json
import math
from urllib.parse import quote_plus
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# stats & ML
from scipy import stats
from sklearn.linear_model import LinearRegression

# UI
import gradio as gr

# ---------------------------
# 0. Configuration
# ---------------------------
DB_PATH = "public_data.db"
WORLD_BANK_BASE = "http://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000"
IND_LITERACY = "SE.ADT.LITR.ZS"
IND_LIFE = "SP.DYN.LE00.IN"
IND_GDP = "NY.GDP.MKTP.CD"
PUPPETEER_API = "http://localhost:3000/ask"  # doesn't work

# --- Mapping pays → ISO3 pour requêtes live World Bank
country_iso3_map = {
    "afghanistan": "AFG",
    "albania": "ALB",
    "algeria": "DZA",
    "angola": "AGO",
    "argentina": "ARG",
    "australia": "AUS",
    "austria": "AUT",
    "bangladesh": "BGD",
    "brazil": "BRA",
    "canada": "CAN",
    "chile": "CHL",
    "china": "CHN",
    "egypt": "EGY",
    "france": "FRA",
    "germany": "DEU",
    "india": "IND",
    "indonesia": "IDN",
    "italy": "ITA",
    "japan": "JPN",
    "kenya": "KEN",
    "mexico": "MEX",
    "nigeria": "NGA",
    "pakistan": "PAK",
    "russia": "RUS",
    "south africa": "ZAF",
    "south korea": "KOR",
    "spain": "ESP",
    "turkey": "TUR",
    "ukraine": "UKR",
    "united kingdom": "GBR",
    "united states": "USA",
}

# ---------------------------
# 1. Data loading & cleaning
# ---------------------------
def fetch_worldbank(indicator):
    url = WORLD_BANK_BASE.format(indicator=indicator)
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    data = resp.json() #Transforme la réponse HTTP en objet Python
                       # L’API World Bank renvoie un tableau JSON du type
    if not data or len(data) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(data[1])
    return df

def load_all_data(cache=True):
    if cache and os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH) #Connexion à la base de données SQLite
            df = pd.read_sql_query("SELECT * FROM merged_data", conn) # Lire la table 'merged_data' puis stockez dans un DataFrame
            conn.close()
            if not df.empty:
                return df
        except Exception:
            pass
# Si pas de cache valide, on récupère les données depuis l’API World Bank
    lit = fetch_worldbank(IND_LITERACY)
    life = fetch_worldbank(IND_LIFE)
    gdp = fetch_worldbank(IND_GDP)

# On met les données dans un format uniforme
    def normalize(df, indicator_name):
        if df.empty:
            return df
        # On garde seulement les colonnes utiles
        df = df[['country', 'countryiso3code', 'date', 'value']].copy()
        df['country'] = df['country'].apply(lambda c: c.get('value') if isinstance(c, dict) else c)
        df['indicator'] = indicator_name
        df = df.rename(columns={'value': 'value_raw'})
        df['value'] = pd.to_numeric(df['value_raw'], errors='coerce')
        # On retourne uniquement les colonnes formatées
        return df[['country', 'countryiso3code', 'date', 'indicator', 'value']]
    
# Normalisation des 3 jeux de données
    lit_n = normalize(lit, 'literacy_rate')
    life_n = normalize(life, 'life_expectancy')
    gdp_n = normalize(gdp, 'gdp')

# Fusion des 3 DataFrames en un seul
    merged = pd.concat([lit_n, life_n, gdp_n], ignore_index=True, sort=False)
    merged = merged[merged['country'].notnull() & merged['date'].notnull()]
    merged['country'] = merged['country'].str.strip()
    # Sauvegarde dans un fichier SQLite pour réutilisation future (cache)
    conn = sqlite3.connect(DB_PATH)
    merged.to_sql('merged_data', conn, if_exists='replace', index=False)
    conn.close()
    return merged

# ---------------------------
# 2. Pandas query utilities
# ---------------------------

#Convertit le nom de l’indicateur en sa forme standard, 
# puis filtre les données pour retourner la valeur correspondante à cet indicateur,
#  pays et année spécifiques
def query_metric_country_year(df, metric, country, year):
    metric_map = {
        'literacy': 'literacy_rate',
        'literacy rate': 'literacy_rate',
        'life expectancy': 'life_expectancy',
        'life': 'life_expectancy',
        'gdp': 'gdp'
    }
    key = metric_map.get(metric.lower(), metric.lower())
    sub = df[df['indicator'] == key]
    sub = sub[sub['country'].str.lower() == country.strip().lower()]
    sub = sub[sub['date'] == str(year)]
    if sub.empty:
        return None
    return sub.iloc[0]['value']

#Cette fonction normalise le nom de la métrique, 
# filtre les données pour un pays donné,
#  trie par année, applique des bornes de dates facultatives, 
# puis renvoie une série temporelle des valeurs indexée par année.
def get_timeseries(df, metric, country, start_year=None, end_year=None):
    metric_map = {
        'literacy': 'literacy_rate',
        'literacy rate': 'literacy_rate',
        'life expectancy': 'life_expectancy',
        'life': 'life_expectancy',
        'gdp': 'gdp'
    }
    key = metric_map.get(metric.lower(), metric.lower())
    sub = df[df['indicator'] == key]
    sub = sub[sub['country'].str.lower() == country.strip().lower()]
    sub = sub[['date', 'value']].copy()
    sub['date'] = sub['date'].astype(int)
    sub = sub.sort_values('date')
    if start_year:
        sub = sub[sub['date'] >= int(start_year)]
    if end_year:
        sub = sub[sub['date'] <= int(end_year)]
    return sub.set_index('date')['value']

# ---------------------------
# 3. Statistics & tests
# ---------------------------
#On calcule et retourne les stats
def descriptive_stats(series):
    s = series.dropna()
    if s.empty:
        return {}
    return {
        'count': int(s.count()),
        'mean': float(s.mean()),
        'std': float(s.std()),
        'min': float(s.min()),
        '25%': float(s.quantile(0.25)),
        '50%': float(s.median()),
        '75%': float(s.quantile(0.75)),
        'max': float(s.max())
    }

#C Calcul de la corrélation de Pearson entre deux indicateurs (metric_x et metric_y) 
# pour un pays donné, en combinant leurs séries temporelles et en ignorant les valeurs manquantes.
def correlation_between(df, country, metric_x, metric_y, years=None):
    ts_x = get_timeseries(df, metric_x, country)
    ts_y = get_timeseries(df, metric_y, country)
    joined = pd.concat([ts_x, ts_y], axis=1).dropna()
    if joined.empty:
        return None
    return float(joined.corr().iloc[0,1])

# On realise un test t de Student pour comparer une métrique donnée 
# entre deux pays à une année spécifique
def t_test_between_countries(df, metric, country_a, country_b, year):
    series_a = get_timeseries(df, metric, country_a, start_year=year, end_year=year)
    series_b = get_timeseries(df, metric, country_b, start_year=year, end_year=year)
    if series_a.empty or series_b.empty:
        return None
    try:
        t, p = stats.ttest_ind(series_a.dropna(), series_b.dropna(), equal_var=False, nan_policy='omit')
        return {'t_stat': float(t), 'p_value': float(p)}
    except Exception:
        return None

# ---------------------------
# 4. Simple ML forecast
# ---------------------------

#On prédit la valeur future d’une métrique 
# en se basant sur une tendance linéaire des données passées.
def forecast_linear(series, years_ahead=1):
    s = series.dropna()
    if s.empty or len(s) < 2:
        return None, None
    X = np.array(s.index).reshape(-1,1)
    y = np.array(s.values).reshape(-1,1)
    model = LinearRegression()
    model.fit(X, y)
    last_year = int(s.index.max())
    future_year = last_year + years_ahead
    pred = model.predict(np.array([[future_year]]))
    return float(pred[0,0]), model

# ---------------------------
# 5. Puppeteer fallback (calls Node API)
# ---------------------------
def puppeteer_fallback(question):
    try:
        r = requests.get(PUPPETEER_API, params={'q': question}, timeout=15)
        r.raise_for_status()
        try:
            data = r.json()
            return data.get('answer') if isinstance(data, dict) else str(data)
        except Exception:
            return r.text
    except Exception as e:
        return f"[Puppeteer error] {e}"

# ---------------------------
# 6. Live World Bank API query (si pas trouvé en local)
# ---------------------------

#On interroge en direct l’API de la Banque Mondiale 
# pour récupérer la valeur d’un indicateur spécifique pour un pays et une année donnés, 
# et renvoie cette valeur ou None si indisponible.
def fetch_worldbank_live(indicator_code, country_code, year):
    url = f"http://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?date={year}&format=json"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if len(data) < 2 or not data[1]:
            return None
        value = data[1][0].get('value')
        return value
    except Exception as e:
        print(f"Erreur API World Bank: {e}")
        return None

# ---------------------------
# 7. Natural language QA routing with live fallback
# ---------------------------

#On extrait l’indicateur, le pays et l’année d’une question simple formulée dans un style type 
# « literacy rate in France in 2020 ». 
def parse_basic_question(question):
    q = question.strip().lower()
    pattern = r'(literacy rate|life expectancy|gdp)\s+(?:of|in)\s+([a-zA-Z\s\.\-]+?)\s+(?:in\s+)?(\d{4})$'
    m = re.search(pattern, q, re.IGNORECASE)
    if not m:
        return None
    metric = m.group(1)
    country = m.group(2).strip()
    year = m.group(3)
    return metric, country, int(year)


#La fonction `handle_question` analyse une question en langage naturel 
# pour extraire une métrique, un pays et une année, puis :
# * cherche la donnée correspondante dans la base locale,
# * si absente, interroge l’API World Bank en direct,
# * sinon, utilise un fallback web via Puppeteer,
# * et gère aussi des questions spécifiques 
# sur la corrélation ou la prévision pour répondre de façon adaptée.

def handle_question(question, df):
    p = parse_basic_question(question)
    if p:
        metric, country, year = p
        val = query_metric_country_year(df, metric, country, year)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            return f"{metric.title()} in {country.title()} in {year}: {val}"

        # Pas trouvé en local : on tente live API World Bank
        indicator_map = {
            'literacy_rate': IND_LITERACY,
            'literacy': IND_LITERACY,
            'literacy rate': IND_LITERACY,
            'life_expectancy': IND_LIFE,
            'life expectancy': IND_LIFE,
            'life': IND_LIFE,
            'gdp': IND_GDP
        }
        key = metric.lower()
        indicator_code = indicator_map.get(key)
        if not indicator_code:
            return "Metric not supported for live lookup."

        iso3 = country_iso3_map.get(country.lower())
        if not iso3:
            return f"Country '{country}' not recognized for live lookup."

        live_val = fetch_worldbank_live(indicator_code, iso3, year)
        if live_val is not None:
            return f"{metric.title()} in {country.title()} in {year} (live API): {live_val}"

        # Sinon fallback Puppeteer
        fallback = puppeteer_fallback(question)
        return f"[Puppeteer] {fallback}"

    q = question.lower()
    if 'correlation' in q and 'gdp' in q and 'life' in q:
        countries = extract_countries_from_text(question, df)
        country = countries[0] if countries else None
        if not country:
            return "Please specify a country for correlation (e.g., 'correlation between GDP and life expectancy in France')."
        corr = correlation_between(df, country, 'gdp', 'life_expectancy')
        if corr is None:
            return f"No overlapping data for correlation in {country}."
        return f"Pearson correlation (GDP vs life expectancy) in {country}: {corr:.3f}"
    if 'forecast' in q or 'predict' in q:
        p = parse_basic_question(question)
        if p:
            metric, country, year = p
            ts = get_timeseries(df, metric, country)
            pred, _ = forecast_linear(ts, years_ahead=1)
            if pred is None:
                return "Not enough data to produce a forecast."
            next_year = int(ts.index.max()) + 1
            return f"Predicted {metric} for {country.title()} in {next_year}: {pred:.2f}"
        else:
            return "Please ask like: 'predict GDP of France in 2021' (needs country and metric)."
    return f"[Puppeteer] {puppeteer_fallback(question)}"

# Détecter quels pays sont mentionnés dans une phrase ou un texte donné
def extract_countries_from_text(text, df):
    countries = df['country'].unique()
    found = []
    txt = text.lower()
    for c in countries:
        if isinstance(c, str) and c.lower() in txt:
            found.append(c)
    return found

# ---------------------------
# 8. Gradio UI
# ---------------------------
DF_ALL = load_all_data()

def gradio_ask(question):
    if not question or not question.strip():
        return "Please enter a question."
    return handle_question(question, DF_ALL)

def plot_timeseries(metric, country, start_year, end_year):
    if not country:
        return "Please select a country."
    series = get_timeseries(DF_ALL, metric, country, start_year, end_year)
    if series.empty:
        return "No data for this selection."
    fig, ax = plt.subplots(figsize=(6,3))
    ax.plot(series.index, series.values, marker='o')
    ax.set_title(f"{metric.title()} - {country.title()}")
    ax.set_xlabel("Year")
    ax.set_ylabel(metric.title())
    ax.grid(True)
    plt.tight_layout()
    plt.close(fig)  
    return fig


def gradio_explore(metric, country, start_year, end_year, show_stats, do_forecast):
    if not country:
        return "Please select a country.", ""
    series = get_timeseries(DF_ALL, metric, country, start_year, end_year)
    if series.empty:
        return "No data available for this selection.", ""
    summary = descriptive_stats(series).copy()
    summary_text = json.dumps(summary, indent=2)
    fig = plot_timeseries(metric, country, start_year, end_year)
    forecast_text = ""
    if do_forecast:
        pred, model = forecast_linear(series, years_ahead=1)
        if pred is None:
            forecast_text = "Not enough data to forecast."
        else:
            next_year = int(series.index.max()) + 1
            forecast_text = f"Predicted {metric} for {country.title()} in {next_year}: {pred:.2f}"
    stats_text = summary_text if show_stats else ""
    result_text = stats_text + ("\n\n" + forecast_text if forecast_text else "")
    return fig, result_text

country_list = sorted([c for c in DF_ALL['country'].unique() if isinstance(c, str)])
metrics = ['gdp', 'life_expectancy', 'literacy_rate']

with gr.Blocks(title="Public Data QA Chatbot + Analytics") as demo:
    gr.Markdown("# Public Data QA Chatbot + Analytics")
    gr.Markdown("Ask questions or explore country metrics (GDP, Life expectancy, Literacy). Puppeteer fallback used for general web queries.")
    with gr.Tabs():
        with gr.TabItem("Ask (chat)"):
            txt = gr.Textbox(label="Ask a question", placeholder="e.g., What is the GDP of France in 2021")
            out = gr.Textbox(label="Answer", lines=5)
            ask_btn = gr.Button("Ask")
            ask_btn.click(fn=gradio_ask, inputs=txt, outputs=out)
        with gr.TabItem("Explore (charts & stats)"):
            with gr.Row():
                metric_dd = gr.Dropdown(label="Metric", choices=metrics, value='gdp')
                country_dd = gr.Dropdown(label="Country", choices=country_list, value=country_list[0])
            with gr.Row():
                start_year_in = gr.Number(label="Start Year (optional)", value=2000)
                end_year_in = gr.Number(label="End Year (optional)", value=2020)
            show_stats_cb = gr.Checkbox(label="Show descriptive stats", value=True)
            forecast_cb = gr.Checkbox(label="Do 1-year forecast", value=False)

            plot_out = gr.Plot()
            summary_out = gr.Textbox(label="Summary / Forecast", lines=6)

            plot_btn = gr.Button("Plot & Analyze")
            plot_btn.click(
        fn=gradio_explore,
        inputs=[metric_dd, country_dd, start_year_in, end_year_in, show_stats_cb, forecast_cb],
        outputs=[plot_out, summary_out]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

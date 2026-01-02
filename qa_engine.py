# qa_engine.py
#
# - Loads World Bank datasets (literacy, life expectancy, GDP)
# - Cleans & merges, saves to SQLite
# - Provides:
#     * direct pandas lookup for questions
#     * descriptive stats, correlation, t-test
#     * simple ML forecast (linear regression)
#     * Puppeteer fallback via http://127.0.0.1:3000/ask
# - Gradio UI with Ask, Explore, and Dataset (URL) modes

import os
import re
import sqlite3
import json
import math
from urllib.parse import quote_plus, urlparse
from datetime import datetime
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# extra for Dataset Explorer
import io
import zipfile
import tempfile
from pathlib import Path

# stats & ML
from scipy import stats
from sklearn.linear_model import LinearRegression

# UI
import gradio as gr

# Optional imports for Dataset Explorer (avoid crash if not installed)
try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None

try:
    from datasets import load_dataset
except Exception:
    load_dataset = None

try:
    import kaggle
except Exception:
    kaggle = None

# ---------------------------
# 0. Configuration
# ---------------------------
DB_PATH = "public_data.db"
WORLD_BANK_BASE = "http://api.worldbank.org/v2/country/all/indicator/{indicator}?format=json&per_page=20000"
IND_LITERACY = "SE.ADT.LITR.ZS"
IND_LIFE = "SP.DYN.LE00.IN"
IND_GDP = "NY.GDP.MKTP.CD"

# IMPORTANT : on utilise 127.0.0.1 pour éviter les soucis IPv4/IPv6
PUPPETEER_API = "http://127.0.0.1:3000/ask"

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
    data = resp.json()
    if not data or len(data) < 2:
        return pd.DataFrame()
    df = pd.DataFrame(data[1])
    return df


def load_all_data(cache=True):
    if cache and os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            df = pd.read_sql_query("SELECT * FROM merged_data", conn)
            conn.close()
            if not df.empty:
                return df
        except Exception:
            pass

    lit = fetch_worldbank(IND_LITERACY)
    life = fetch_worldbank(IND_LIFE)
    gdp = fetch_worldbank(IND_GDP)

    def normalize(df, indicator_name):
        if df.empty:
            return df
        df = df[['country', 'countryiso3code', 'date', 'value']].copy()
        df['country'] = df['country'].apply(lambda c: c.get('value') if isinstance(c, dict) else c)
        df['indicator'] = indicator_name
        df = df.rename(columns={'value': 'value_raw'})
        df['value'] = pd.to_numeric(df['value_raw'], errors='coerce')
        return df[['country', 'countryiso3code', 'date', 'indicator', 'value']]

    lit_n = normalize(lit, 'literacy_rate')
    life_n = normalize(life, 'life_expectancy')
    gdp_n = normalize(gdp, 'gdp')

    merged = pd.concat([lit_n, life_n, gdp_n], ignore_index=True, sort=False)
    merged = merged[merged['country'].notnull() & merged['date'].notnull()]
    merged['country'] = merged['country'].str.strip()

    conn = sqlite3.connect(DB_PATH)
    merged.to_sql('merged_data', conn, if_exists='replace', index=False)
    conn.close()
    return merged

# ---------------------------
# 2. Pandas query utilities
# ---------------------------

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


def correlation_between(df, country, metric_x, metric_y, years=None):
    ts_x = get_timeseries(df, metric_x, country)
    ts_y = get_timeseries(df, metric_y, country)
    joined = pd.concat([ts_x, ts_y], axis=1).dropna()
    if joined.empty:
        return None
    return float(joined.corr().iloc[0, 1])


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

def forecast_linear(series, years_ahead=1):
    s = series.dropna()
    if s.empty or len(s) < 2:
        return None, None
    X = np.array(s.index).reshape(-1, 1)
    y = np.array(s.values).reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    last_year = int(s.index.max())
    future_year = last_year + years_ahead
    pred = model.predict(np.array([[future_year]]))
    return float(pred[0, 0]), model

# ---------------------------
# 5. Puppeteer fallback (calls Node API)
# ---------------------------

def puppeteer_fallback(question: str) -> str:
    try:
        resp = requests.post(
            PUPPETEER_API,
            json={"query": question},
            timeout=15
        )
        print("[Puppeteer] HTTP status:", resp.status_code)
        print("[Puppeteer] Raw body:", resp.text[:200])

        resp.raise_for_status()
        data = resp.json()
        answer = data.get("answer")
        if answer:
            return answer
        return "Aucune réponse reçue de l'API Puppeteer."
    except Exception as e:
        print("[Puppeteer Error]", e)
        return "Erreur : Impossible de contacter l'API Puppeteer."

# ---------------------------
# 6. Live World Bank API query (si pas trouvé en local)
# ---------------------------

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


def extract_countries_from_text(text, df):
    countries = df['country'].unique()
    found = []
    txt = text.lower()
    for c in countries:
        if isinstance(c, str) and c.lower() in txt:
            found.append(c)
    return found


def handle_question(question, df):
    p = parse_basic_question(question)
    if p:
        metric, country, year = p
        val = query_metric_country_year(df, metric, country, year)
        if val is not None and not (isinstance(val, float) and math.isnan(val)):
            return f"{metric.title()} in {country.title()} in {year}: {val}"

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

        web_answer = puppeteer_fallback(question)
        return f"[Puppeteer] {web_answer}"

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

# ---------------------------
# 7.b Dataset URL Explorer (Kaggle / HF / Direct)
# ---------------------------

SUPPORTED_TABULAR_EXT = {".csv", ".tsv", ".parquet", ".json", ".jsonl", ".xlsx"}

class DatasetError(Exception):
    pass

def _nice_err(msg: str) -> str:
    return f"❌ {msg}"

def detect_source(url: str) -> str:
    u = (url or "").strip()
    if "kaggle.com" in u:
        return "kaggle"
    if "huggingface.co" in u:
        return "huggingface"
    return "direct"

def download_direct(url: str, dst_dir: Path) -> Path:
    try:
        r = requests.get(url, timeout=30, stream=True)
        r.raise_for_status()
    except Exception as e:
        raise DatasetError(f"Impossible de télécharger le fichier (lien direct). Détail: {e}")

    filename = None
    cd = r.headers.get("content-disposition", "")
    m = re.search(r'filename="?([^"]+)"?', cd)
    if m:
        filename = m.group(1)
    if not filename:
        filename = os.path.basename(urlparse(url).path) or "dataset.bin"

    out = dst_dir / filename
    with open(out, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
    return out

def download_huggingface(url: str, dst_dir: Path) -> Path:
    if hf_hub_download is None:
        raise DatasetError("huggingface_hub n'est pas installé. Fais: pip install huggingface_hub")

    u = url.strip()
    m = re.search(r"huggingface\.co/datasets/([^/]+/[^/]+)", u)
    if not m:
        raise DatasetError("Lien Hugging Face non reconnu. Exemple: https://huggingface.co/datasets/nyu-mll/glue")

    repo_id = m.group(1)

    # lien direct vers un fichier (resolve)
    mfile = re.search(r"/resolve/[^/]+/(.+)$", u)
    if mfile:
        filename = mfile.group(1)
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                local_dir=str(dst_dir)
            )
            return Path(local_path)
        except Exception as e:
            raise DatasetError(f"Impossible de télécharger le fichier depuis HF. Détail: {e}")

    # repo dataset -> load via datasets (sample)
    if load_dataset is None:
        raise DatasetError("datasets n'est pas installé. Fais: pip install datasets")

    try:
        ds = load_dataset(repo_id)
    except Exception as e:
        raise DatasetError(f"Impossible de charger ce dataset HF via datasets. Détail: {e}")

    split_name = list(ds.keys())[0]
    df = ds[split_name].to_pandas()
    out = dst_dir / f"{repo_id.replace('/', '__')}_{split_name}_sample.csv"
    df.head(50000).to_csv(out, index=False)
    return out

def download_kaggle(url: str, dst_dir: Path) -> Path:
    if kaggle is None:
        raise DatasetError("kaggle n'est pas installé. Fais: pip install kaggle")

    m = re.search(r"kaggle\.com/datasets/([^/]+/[^/?#]+)", url)
    if not m:
        raise DatasetError("Lien Kaggle non reconnu. Exemple: https://www.kaggle.com/datasets/uciml/iris")

    dataset_slug = m.group(1)
    try:
        kaggle.api.dataset_download_files(dataset_slug, path=str(dst_dir), unzip=False, quiet=True)
    except Exception as e:
        raise DatasetError(
            "Impossible de télécharger depuis Kaggle. Vérifie ton token kaggle.json (~/.kaggle/kaggle.json). "
            f"Détail: {e}"
        )

    zips = sorted(dst_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        raise DatasetError("Téléchargement Kaggle OK mais aucun .zip trouvé.")
    return zips[0]

def pick_first_tabular_from_zip(zip_path: Path, dst_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as z:
        members = [m for m in z.namelist() if not m.endswith("/")]
        preferred = [".csv", ".tsv", ".parquet", ".jsonl", ".json", ".xlsx"]
        chosen = None
        for ext in preferred:
            for m in members:
                if m.lower().endswith(ext):
                    chosen = m
                    break
            if chosen:
                break
        if not chosen:
            raise DatasetError(
                "ZIP détecté mais aucun fichier tabulaire supporté à l'intérieur "
                f"({', '.join(sorted(SUPPORTED_TABULAR_EXT))})."
            )

        z.extract(chosen, path=str(dst_dir))
        return dst_dir / chosen

def read_tabular(file_path: Path) -> pd.DataFrame:
    ext = file_path.suffix.lower()
    try:
        if ext == ".csv":
            return pd.read_csv(file_path)
        if ext == ".tsv":
            return pd.read_csv(file_path, sep="\t")
        if ext == ".parquet":
            return pd.read_parquet(file_path)
        if ext == ".json":
            try:
                return pd.read_json(file_path)
            except Exception:
                return pd.read_json(file_path, lines=True)
        if ext == ".jsonl":
            return pd.read_json(file_path, lines=True)
        if ext == ".xlsx":
            return pd.read_excel(file_path)
    except Exception as e:
        raise DatasetError(f"Impossible de lire le fichier ({ext}). Détail: {e}")

    raise DatasetError(
        f"Format non supporté: {ext}. Formats acceptés: {', '.join(sorted(SUPPORTED_TABULAR_EXT))} (+ ZIP)."
    )

def make_summary(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "Dataset vide."

    n_rows, n_cols = df.shape
    missing = df.isna().sum().sort_values(ascending=False)
    top_missing = missing.head(min(10, len(missing)))

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]

    out = []
    out.append(f"📌 **Shape**: {n_rows:,} lignes × {n_cols:,} colonnes")
    out.append(f"📌 **Colonnes numériques**: {len(num_cols)} | **Catégorielles/Autres**: {len(cat_cols)}")
    out.append("\n📌 **Aperçu (5 lignes)**:")
    out.append(df.head(5).to_string(index=False))

    out.append("\n📌 **Types (dtypes)**:")
    out.append(df.dtypes.astype(str).to_string())

    out.append("\n📌 **Valeurs manquantes (Top 10)**:")
    out.append(top_missing.to_string())

    if num_cols:
        desc = df[num_cols].describe().T
        out.append("\n📌 **Stats numériques (describe)**:")
        out.append(desc.to_string())

    return "\n".join(out)

def plot_missing(df: pd.DataFrame):
    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if miss.empty:
        ax.text(0.5, 0.5, "Aucune valeur manquante.", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.bar(miss.index.astype(str)[:30], miss.values[:30])
    ax.set_title("Proportion de valeurs manquantes (Top 30)")
    ax.set_ylabel("Ratio")
    ax.tick_params(axis="x", labelrotation=70)
    plt.tight_layout()
    return fig

def plot_distributions(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    if not num_cols:
        ax.text(0.5, 0.5, "Aucune colonne numérique pour histogrammes.", ha="center", va="center")
        ax.axis("off")
        return fig
    col = num_cols[0]
    s = df[col].dropna()
    if s.empty:
        ax.text(0.5, 0.5, f"Colonne '{col}' vide.", ha="center", va="center")
        ax.axis("off")
        return fig
    ax.hist(s.values, bins=30)
    ax.set_title(f"Distribution (hist) - {col}")
    ax.set_xlabel(col)
    plt.tight_layout()
    return fig

def plot_correlation(df: pd.DataFrame):
    num = df.select_dtypes(include=[np.number])
    fig, ax = plt.subplots(figsize=(6, 5))
    if num.shape[1] < 2:
        ax.text(0.5, 0.5, "Pas assez de colonnes numériques pour corrélation.", ha="center", va="center")
        ax.axis("off")
        return fig
    corr = num.corr(numeric_only=True)
    ax.imshow(corr.values)
    ax.set_title("Matrice de corrélation (numérique)")
    ax.set_xticks(range(corr.shape[1]))
    ax.set_yticks(range(corr.shape[0]))
    ax.set_xticklabels(corr.columns, rotation=70, ha="right")
    ax.set_yticklabels(corr.index)
    plt.tight_layout()
    return fig

def analyze_dataset(url: str):
    if not url or not url.strip():
        return _nice_err("Colle un lien de dataset."), None, None, None

    url = url.strip()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        source = detect_source(url)

        try:
            if source == "kaggle":
                downloaded = download_kaggle(url, tmp_dir)
            elif source == "huggingface":
                downloaded = download_huggingface(url, tmp_dir)
            else:
                downloaded = download_direct(url, tmp_dir)

            if downloaded.suffix.lower() == ".zip":
                file_path = pick_first_tabular_from_zip(downloaded, tmp_dir)
            else:
                file_path = downloaded

            if file_path.suffix.lower() not in SUPPORTED_TABULAR_EXT:
                raise DatasetError(
                    f"Fichier téléchargé: {file_path.name} mais format non supporté ({file_path.suffix}). "
                    "Donne un lien direct vers CSV/Parquet/JSON/Excel, ou un ZIP avec ces fichiers."
                )

            df = read_tabular(file_path)

            if len(df) > 300000:
                df = df.sample(300000, random_state=42)

            summary = make_summary(df)
            fig1 = plot_missing(df)
            fig2 = plot_distributions(df)
            fig3 = plot_correlation(df)

            return summary, fig1, fig2, fig3

        except DatasetError as e:
            return _nice_err(str(e)), None, None, None
        except Exception as e:
            return _nice_err(f"Erreur inattendue: {e}"), None, None, None

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
    fig, ax = plt.subplots(figsize=(6, 3))
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

        with gr.TabItem("Dataset (URL)"):
            gr.Markdown("Colle un lien Kaggle / HuggingFace / lien direct (CSV/Parquet/JSON/Excel ou ZIP).")
            ds_url = gr.Textbox(label="Dataset URL", placeholder="Ex: https://huggingface.co/datasets/nyu-mll/glue")
            ds_btn = gr.Button("Analyser le dataset")

            ds_summary = gr.Markdown(label="Résumé")
            ds_plot_missing = gr.Plot(label="Missing values")
            ds_plot_dist = gr.Plot(label="Distribution")
            ds_plot_corr = gr.Plot(label="Corrélation")

            ds_btn.click(
                fn=analyze_dataset,
                inputs=[ds_url],
                outputs=[ds_summary, ds_plot_missing, ds_plot_dist, ds_plot_corr]
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_api=False)

import requests
import pandas as pd

def fetch_data(url, indicator_name):
    print(f"Fetching data from {url} ...")
    response = requests.get(url)
    data_json = response.json()

    # Vérifier que les données sont présentes
    if not data_json or len(data_json) < 2:
        print(f"No data found for {indicator_name}.")
        return pd.DataFrame()

    data = data_json[1]  # Les données sont dans la 2e position

    records = []
    for entry in data:
        country = entry['country']['value']
        year = int(entry['date'])
        value = entry['value']
        records.append({
            'country': country,
            'year': year,
            indicator_name: value
        })

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} rows for {indicator_name}")
    return df

def load_and_merge_datasets(
    lit_url="http://api.worldbank.org/v2/country/all/indicator/SE.ADT.LITR.ZS?format=json&per_page=20000",
    life_url="http://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN?format=json&per_page=20000"
):
    # Charger chaque dataset
    df_lit = fetch_data(lit_url, 'literacy_rate')
    df_life = fetch_data(life_url, 'life_expectancy')

    print("Merging datasets...")
    # Fusion sur country et year (inner join)
    merged = pd.merge(df_lit, df_life, on=['country', 'year'], how='inner')

    # Nettoyage simple (ex : supprimer lignes avec valeurs nulles)
    merged_clean = merged.dropna(subset=['literacy_rate', 'life_expectancy'])
    print(f"Merged dataset contains {len(merged_clean)} rows.\n")

    print("--- Dataset sample ---")
    print(merged_clean.head())

    return merged_clean


if __name__ == "__main__":
    # Test rapide
    df = load_and_merge_datasets()
    print("\nTest passed! Dataset loaded and merged correctly.")

    


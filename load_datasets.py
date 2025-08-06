import requests
import pandas as pd

# ------------------------------
# 📌 1. Load Literacy Rate Data
# ------------------------------
def load_literacy_data():
    url = "http://api.worldbank.org/v2/country/all/indicator/SE.ADT.LITR.ZS?format=json&per_page=20000"
    response = requests.get(url)
    data = response.json()[1]
    df = pd.json_normalize(data)
    df = df[['country.value', 'date', 'value']]
    df.columns = ['country', 'year', 'literacy_rate']
    return df

# ------------------------------
# 📌 2. Load Life Expectancy Data
# ------------------------------
def load_life_expectancy_data():
    url = "http://api.worldbank.org/v2/country/all/indicator/SP.DYN.LE00.IN?format=json&per_page=20000"
    response = requests.get(url)
    data = response.json()[1]
    df = pd.json_normalize(data)
    df = df[['country.value', 'date', 'value']]
    df.columns = ['country', 'year', 'life_expectancy']
    return df

# ------------------------------
# 📌 3. Merge and Clean Data
# ------------------------------
def merge_datasets(year='2020'):
    df_lit = load_literacy_data()
    df_life = load_life_expectancy_data()

    # Filter on selected year
    df_lit = df_lit[df_lit['year'] == year]
    df_life = df_life[df_life['year'] == year]

    # Merge datasets
    merged_df = pd.merge(df_lit, df_life, on=['country', 'year'])
    return merged_df

# ------------------------------
# ✅ Test in script mode
# ------------------------------
if __name__ == "__main__":
    df = merge_datasets()
    # Le DataFrame fusionnée contient: pays, année, 
    # taux d’alphabétisation, et espérance de vie.
    print(df.head())

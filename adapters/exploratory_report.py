import pandas as pd
from ydata_profiling import ProfileReport

def gerar_relatorio_exploratorio():
    # Carrega o dataset
    df = pd.read_csv("application/data/SpotifyFeatures.csv")

    # Gera o relatório
    profile = ProfileReport(df, title="Relatório Exploratório - Spotify", explorative=True)

    # Salva como HTML
    profile.to_file("spotify_relatorio.html")

    print("✅ Relatório gerado com sucesso: spotify_relatorio.html")

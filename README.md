# Cross-Asset-Analysis-Tool
#Global Macro &amp; Cross-Asset Analytics Tool A Python-based quantitative tool for financial market analysis. Features automated data ingestion via YFinance, rolling correlation #matrices, and risk-reward performance tracking (Sharpe Ratio, Volatility). Built with Pandas &amp; Seaborn to detect macro regime shifts and portfolio anomalies.




import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# --- FONCTION 1 : ACQUISITION ET NETTOYAGE ---
def get_financial_data(tickers, period="2y"):
    """
    Télécharge les prix de clôture et calcule les rendements.
    """
    print(f"Étape 1 : Téléchargement de {len(tickers)} actifs...")
    data = yf.download(tickers, period=period)['Close']
    
    # Nettoyage des données manquantes
    data = data.dropna()
    
    # Calcul des rendements quotidiens
    returns = data.pct_change().dropna()
    
    return data, returns

# --- FONCTION 2 : ANALYSE DE CORRÉLATION ---
def plot_correlation_analysis(returns):
    """
    Calcule et affiche la matrice de corrélation.
    """
    print("Étape 2 : Analyse de la corrélation...")
    correlation_matrix = returns.corr()
    
    plt.figure(figsize=(12, 9))
    sns.heatmap(correlation_matrix, 
                annot=True, 
                cmap='coolwarm', 
                fmt=".3f", 
                linewidths=0.5, 
                vmin=-1, 
                vmax=1)
    
    plt.title('Matrice de Corrélation Cross-Asset', fontsize=18)
    plt.show()
    return correlation_matrix

# --- FONCTION 3 : PERFORMANCE ET RISQUE ---
def calculate_performance_metrics(returns, risk_free_rate=0.03):
    """
    Calcule le Rendement, la Volatilité et le Ratio de Sharpe.
    """
    print("Étape 3 : Calcul des indicateurs de performance...")
    
    # Calculs annualisés (252 jours de bourse par an)
    annual_return = returns.mean() * 252
    annual_volatility = returns.std() * np.sqrt(252)
    sharpe_ratio = (annual_return - risk_free_rate) / annual_volatility
    
    # Création du DataFrame résumé
    metrics = pd.DataFrame({
        'Rendement Annuel (%)': annual_return * 100,
        'Volatilité Annuelle (%)': annual_volatility * 100,
        'Ratio de Sharpe': sharpe_ratio
    })
    
    # Visualisation des métriques
    plt.figure(figsize=(10, 6))
    sns.heatmap(metrics[['Ratio de Sharpe']], annot=True, cmap='RdYlGn', fmt=".2f")
    plt.title('Classement par Ratio de Sharpe (Efficacité Risque/Rendement)', fontsize=15)
    plt.show()
    
    return metrics

# =============================================================================
# EXÉCUTION DU PROJET (Le "Main")
# =============================================================================
if __name__ == "__main__":
    # 1. Configuration
    my_assets = ['^GSPC', '^TNX', 'GC=F', 'EURUSD=X', 'CL=F', 'TSLA', 'BTC-USD']
    
    # 2. Appel des fonctions
    prices, daily_returns = get_financial_data(my_assets, period="2y")
    
    corr_matrix = plot_correlation_analysis(daily_returns)
    
    perf_table = calculate_performance_metrics(daily_returns)
    
    # 3. Affichage final du tableau dans la console
    print("\n--- TABLEAU RÉSUMÉ FINAL ---")
    print(perf_table.round(2))

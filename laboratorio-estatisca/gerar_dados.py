import numpy as np
import pandas as pd

np.random.seed(42)
n = 1200  # Cumpre o requisito de ter mais de 1.000 registros

# 4 Variáveis Numéricas com correlações reais
potencia = np.random.normal(loc=180, scale=60, size=n).clip(75, 500)
consumo_cidade = (45 - 0.05 * potencia + np.random.normal(0, 3, size=n)).clip(8, 30)
consumo_estrada = consumo_cidade * 1.25 + np.random.normal(0, 1.5, size=n)
preco = (
    15000 + 350 * potencia - 400 * consumo_cidade + np.random.exponential(10000, size=n)
).clip(20000, 250000)

# 2 Variáveis Categóricas
cambio = np.random.choice(["Automático", "Manual"], size=n, p=[0.7, 0.3])
combustivel = np.random.choice(["Gasolina", "Flex", "Diesel", "Híbrido"], size=n, p=[0.4, 0.4, 0.1, 0.1])

df = pd.DataFrame({
    "Potencia_CV": np.round(potencia, 1),
    "Consumo_Cidade_KmL": np.round(consumo_cidade, 1),
    "Consumo_Estrada_KmL": np.round(consumo_estrada, 1),
    "Preco_R$": np.round(preco, 2),
    "Tipo_Cambio": cambio,
    "Combustivel": combustivel
})

df.to_csv("dados.csv", index=False)
print("Arquivo dados.csv gerado com sucesso!")
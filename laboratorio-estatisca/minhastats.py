import math
from collections import Counter

def media(valores):
    return sum(valores) / len(valores)

def mediana(valores):
    ordenados = sorted(valores)
    n = len(ordenados)
    meio = n // 2
    if n % 2 == 0:
        return (ordenados[meio - 1] + ordenados[meio]) / 2.0
    return float(ordenados[meio])

def moda(valores):
    contagem = Counter(valores)
    max_freq = max(contagem.values())
    modas = [k for k, v in contagem.items() if v == max_freq]
    return modas[0] if len(modas) == 1 else modas

def amplitude(valores):
    return max(valores) - min(valores)

def variancia(valores, tipo="amostral"):
    m = media(valores)
    n = len(valores)
    denominador = (n - 1) if tipo == "amostral" else n
    return sum((x - m) ** 2 for x in valores) / denominador

def desvio_padrao(valores, tipo="amostral"):
    return math.sqrt(variancia(valores, tipo=tipo))

def percentil(valores, p):
    """Percentil linear padrão NumPy (p de 0 a 100)."""
    ordenados = sorted(valores)
    n = len(ordenados)
    pos = (p / 100.0) * (n - 1)
    inferior = int(math.floor(pos))
    superior = int(math.ceil(pos))
    if inferior == superior:
        return float(ordenados[inferior])
    peso = pos - inferior
    return ordenados[inferior] * (1.0 - peso) + ordenados[superior] * peso

def quartis(valores):
    return percentil(valores, 25), percentil(valores, 50), percentil(valores, 75)

def coeficiente_variacao(valores):
    m = media(valores)
    return (desvio_padrao(valores, tipo="amostral") / m) * 100 if m != 0 else 0.0

def covariancia(x, y, tipo="amostral"):
    n = len(x)
    mx, my = media(x), media(y)
    den = (n - 1) if tipo == "amostral" else n
    return sum((x[i] - mx) * (y[i] - my) for i in range(n)) / den

def correlacao_pearson(x, y):
    cov = covariancia(x, y, "amostral")
    dp_x = desvio_padrao(x, "amostral")
    dp_y = desvio_padrao(y, "amostral")
    return cov / (dp_x * dp_y) if (dp_x * dp_y) != 0 else 0.0

def regressao_linear_simples(x, y):
    mx, my = media(x), media(y)
    beta1 = covariancia(x, y, "amostral") / variancia(x, "amostral")
    beta0 = my - beta1 * mx
    
    # R²
    y_pred = [beta0 + beta1 * xi for xi in x]
    ss_tot = sum((yi - my) ** 2 for yi in y)
    ss_res = sum((yi - y_pred[i]) ** 2 for i, yi in enumerate(y))
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    
    return beta0, beta1, r2
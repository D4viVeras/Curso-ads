import pytest
import numpy as np
from scipy import stats
import minhastats as ms

@pytest.fixture
def dados_teste():
    np.random.seed(42)
    return np.random.normal(loc=50, scale=15, size=200).tolist()

@pytest.fixture
def dados_par():
    np.random.seed(42)
    x = np.random.uniform(10, 100, size=150)
    y = 2.5 * x + np.random.normal(0, 5, size=150)
    return x.tolist(), y.tolist()

def test_medidas_tendencia_central(dados_teste):
    assert pytest.approx(ms.media(dados_teste), rel=1e-5) == np.mean(dados_teste)
    assert pytest.approx(ms.mediana(dados_teste), rel=1e-5) == np.median(dados_teste)

def test_dispersao(dados_teste):
    assert pytest.approx(ms.amplitude(dados_teste), rel=1e-5) == np.ptp(dados_teste)
    assert pytest.approx(ms.variancia(dados_teste, "amostral"), rel=1e-5) == np.var(dados_teste, ddof=1)
    assert pytest.approx(ms.variancia(dados_teste, "populacional"), rel=1e-5) == np.var(dados_teste, ddof=0)
    assert pytest.approx(ms.desvio_padrao(dados_teste, "amostral"), rel=1e-5) == np.std(dados_teste, ddof=1)

def test_quartis(dados_teste):
    q1, q2, q3 = ms.quartis(dados_teste)
    nq1, nq2, nq3 = np.percentile(dados_teste, [25, 50, 75])
    assert pytest.approx(q1, rel=1e-4) == nq1
    assert pytest.approx(q2, rel=1e-4) == nq2
    assert pytest.approx(q3, rel=1e-4) == nq3

def test_correlacao_e_regressao(dados_par):
    x, y = dados_par
    # Pearson
    corr_propria = ms.correlacao_pearson(x, y)
    corr_np = np.corrcoef(x, y)[0, 1]
    assert pytest.approx(corr_propria, rel=1e-4) == corr_np
    
    # Regressão
    b0, b1, r2 = ms.regressao_linear_simples(x, y)
    res = stats.linregress(x, y)
    assert pytest.approx(b1, rel=1e-4) == res.slope
    assert pytest.approx(b0, rel=1e-4) == res.intercept
    assert pytest.approx(r2, rel=1e-4) == (res.rvalue ** 2)
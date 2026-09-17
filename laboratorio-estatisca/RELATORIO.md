# Relatório: Laboratório Estatístico Interativo

## 1. Identificação do Projeto
* **Equipe:** Davi Veras Gonçalves do Nascimento - Matrícula: 72650199
* **Dataset:** Dados simulados baseados no *Car Features and MSRP* (Kaggle).
* **Justificativa do Dataset:** A escolha por dados automotivos justifica-se pela presença clara de variáveis contínuas (potência, consumo, preço) que possuem correlação lógica e esperada no mundo real, além de variáveis categóricas para agrupamento.

## 2. Decisões de Implementação do Núcleo Estatístico
O núcleo estatístico (`minhastats.py`) foi desenvolvido inteiramente em Python puro, sem o uso de bibliotecas de terceiros para os cálculos matemáticos.

**Fórmulas Matemáticas Implementadas:**
* **Média:**
  $$ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i $$
* **Variância Amostral:**
  $$ s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2 $$
* **Desvio Padrão Amostral:**
  $$ s = \sqrt{s^2} $$
* **Covariância Amostral:**
  $$ Cov(X,Y) = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y}) $$
* **Coeficiente de Correlação de Pearson:**
  $$ r = \frac{Cov(X,Y)}{s_x s_y} $$
* **Regressão Linear Simples (Mínimos Quadrados):**
  $$ \beta_1 = \frac{Cov(X,Y)}{Var(X)} $$
  $$ \beta_0 = \bar{y} - \beta_1 \bar{x} $$

## 3. Resultados da Validação Automatizada
Foi construída uma suíte de testes automatizados (`test_stats.py`) utilizando a biblioteca `pytest`. Todas as funções implementadas no núcleo próprio foram testadas contra os equivalentes consolidados do `numpy` e do `scipy.stats`. As validações passaram com sucesso, aceitando uma tolerância numérica (`rel=1e-5`) decorrente das diferenças na precisão de ponto flutuante do Python puro contra a implementação em C do NumPy.

## 4. As 3 Descobertas Estatísticas
1. **Simetria de Distribuição:** Ao analisar a variável `Potencia_CV`, constatou-se que a distribuição é aproximadamente simétrica. Isso é provado pela forte proximidade entre a Média (183.04) e a Mediana (182.90).
2. **Comprovação do Monte Carlo:** A simulação ilustrou a Lei dos Grandes Números perfeitamente. No gráfico, notamos que nos primeiros lançamentos a frequência relativa oscila, mas ao chegar em 1.000, estabiliza de forma quase idêntica à probabilidade teórica de 0.5.
3. **Correlação Negativa Lógica:** Identificou-se correlação negativa ($r = -0.3216$) entre Potência e Consumo na Cidade. Isso prova estatisticamente a lógica de que motores mais potentes gastam mais combustível. O $R^2$ revelou que ~10% do consumo é explicado exclusivamente pelo aumento da potência.
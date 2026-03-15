# Trabalho de Inteligência Artificial Computacional

**Universidade de Fortaleza - UNIFOR**  
**Disciplina:** Inteligência Artificial Computacional  

---

## Integrantes

- Lucas Cardoso
- 

---

## Descrição

Este repositório contém a implementação dos problemas propostos no trabalho da disciplina de Inteligência Artificial Computacional, envolvendo modelos de **Regressão Linear Simples** e **Regressão Linear Múltipla**, implementados do zero utilizando apenas NumPy.

---

## Problemas

### Problema 1 — Regressão Linear Simples (`prob1.py`)
Análise da relação entre o número médio de jardas por tentativa de passagem (x) e a pontuação (y) dos principais quarterbacks da Liga Americana de Futebol em 2004.

- Cálculo das estimativas de mínimos quadrados (b0 e b1)
- Previsão de pontuação média para um valor específico de jardas
- Análise do impacto de variações nas jardas sobre a pontuação
- Tabela de resíduos

### Problema 2 — Regressão Linear Múltipla (`prob2.py`)
Análise do efeito da inspeção de raios X em circuitos integrados, modelando a dose de radiação em função da corrente (mAmp) e do tempo de exposição (minutos).

- Ajuste de modelo de regressão linear múltipla
- Previsão de dose de radiação para valores específicos
- Avaliação do modelo com R², R² ajustado, RMSE e MAE
- Comparação entre modelos alternativos

---

## Datasets

| Arquivo | Descrição |
|---|---|
| `qb_2004.csv` | Dados dos quarterbacks da NFL em 2004 |
| `dose_radiacao_expandido.csv` | Dados de dose de radiação por corrente e tempo |

---

## Tecnologias

- Python 3
- NumPy

> Não foram bibliotecas com implementações prontas de modelos, como `sklearn` ou `pandas`.

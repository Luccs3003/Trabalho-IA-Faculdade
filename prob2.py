import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class RegressionLinearM:
    def __init__(self, X, y, fit_intercept=True):
        X = np.array(X)
        if len(X.shape) == 1:
            self.X = X.reshape(-1, 1)
        else:
            self.X = X
            
        self.y = np.array(y)
        self.beta = None
        self.fit_intercept = fit_intercept
        self.N = self.X.shape[0]

    def fit(self):
        if self.fit_intercept:
            X_train = np.column_stack((np.ones(self.N), self.X))
        else:
            X_train = self.X
            
        self.beta = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ self.y

    def predict(self, X_new):
        X_new = np.array(X_new)
        num_features = self.X.shape[1]
        X_new = X_new.reshape(-1, num_features)
        
        N_new = X_new.shape[0]
        
        if self.fit_intercept:
            X_pred = np.column_stack((np.ones(N_new), X_new))
        else:
            X_pred = X_new
            
        return X_pred @ self.beta

def r2_score(y_true, y_pred):
    numerador = np.sum((y_true - y_pred) ** 2)
    denominador = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (numerador / denominador)

def r2_ajustado(r2, n, p):
    return 1 - (1 - r2) * ((n - 1) / (n - p - 1))

def mse_score(y_true, y_pred):
    n = len(y_true)
    return (1/n) * np.sum((y_true - y_pred) ** 2)

def rmse_score(y_true, y_pred):
    return np.sqrt(mse_score(y_true, y_pred))

def mae_score(y_true, y_pred):
    n = len(y_true)
    return (1/n) * np.sum(np.abs(y_true - y_pred))


dados = np.loadtxt("dose_radiacao_expandido.csv", delimiter=",", skiprows=1, usecols=(1, 2, 3))

y = dados[:, 0] # Dose de Radiacao
x1 = dados[:, 1] # mAmp
x2 = dados[:, 2] # Tempo de Exposicao

X_multiplo = np.column_stack((x1, x2))
n_obs = len(y)



# ITEM A, C e D: MODELO MÚLTIPLO COMPLETO
modelo = RegressionLinearM(X_multiplo, y, fit_intercept=True)
modelo.fit() 
y_pred = modelo.predict(X_multiplo)

r2_completo = r2_score(y, y_pred)
r2_adj_completo = r2_ajustado(r2_completo, n_obs, p=2)

# ITEM B: PREVISÃO ESPECÍFICA (15 mAmp, 5 min)
X_novo = np.array([15, 5]) 
previsao_especifica = modelo.predict(X_novo)

# ITEM E: MODELO ALTERNATIVO (APENAS CORRENTE)
X_simples = x1 
modelo_simples = RegressionLinearM(X_simples, y, fit_intercept=True)
modelo_simples.fit()
y_pred_simples = modelo_simples.predict(X_simples)

r2_simples = r2_score(y, y_pred_simples)
r2_adj_simples = r2_ajustado(r2_simples, n_obs, p=1) 

# ITEM F: MODELO MÚLTIPLO COM INTERCEPTO FORÇADO A ZERO
modelo_zero = RegressionLinearM(X_multiplo, y, fit_intercept=False)
modelo_zero.fit()
y_pred_zero = modelo_zero.predict(X_multiplo)

r2_zero = r2_score(y, y_pred_zero)
rmse_zero = rmse_score(y, y_pred_zero)

# ITEM H: COMPARAÇÃO DE MÉTRICAS DE ERRO
mse_completo = mse_score(y, y_pred)
rmse_completo = rmse_score(y, y_pred)
mae_completo = mae_score(y, y_pred)

mse_simples = mse_score(y, y_pred_simples)
rmse_simples = rmse_score(y, y_pred_simples)
mae_simples = mae_score(y, y_pred_simples)




x1_grid, x2_grid = np.meshgrid(np.linspace(min(x1), max(x1), 50),
                               np.linspace(min(x2), max(x2), 50))

y_grid = modelo.beta[0] + modelo.beta[1]*x1_grid + modelo.beta[2]*x2_grid
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(x1, x2, y, color="red", label="dados reais")

# plano da regressão
ax.plot_surface(x1_grid, x2_grid, y_grid, alpha=0.5, color="blue")
plt.show()

## Plotagem com plotly
fig = go.Figure()
fig.add_scatter3d(x=x1, y=x2, z=y, mode="markers",
                  marker=dict(color="red", size=3), name="Dados Originais")
fig.add_scatter3d(x=x1, y=x2, z=y_pred, mode="markers",
                marker=dict(color="green", size=3), name="Dados Previstos")
fig.add_surface(x=x1_grid, y=x2_grid, z=y_grid, showscale=False, opacity=0.5)
fig.update_layout(title="grafico da regressão multipla",
                  scene=dict(xaxis_title="x1",
                             yaxis_title="x2",
                             zaxis_title="y"))

fig.show()

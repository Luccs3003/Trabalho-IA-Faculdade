import numpy as np
#visualizacao de dados
import matplotlib.pyplot as plt
import plotly.graph_objects as go

class RegressionLinearM:
    def __init__(self, X,y): #construtor
        self.X = X
        self.y = y
        self.beta = None #parametros
        self.N = X.shape[0]
    def fit(self): #treinamento
        self.X = np.column_stack((np.ones((self.N)), self.X))
        self.beta = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
    def predict(self, X_new):
        N = X_new.shape[0]
        X_new = np.column_stack((np.ones((N)), X_new))
        return X_new @ self.beta
        

dados = np.loadtxt("dose_radiacao_expandido.csv", delimiter=",", skiprows=1, usecols=(1, 2, 3))
print(dados)

y = dados[:, 0] # Dose de Radiacao
x1 = dados[:, 1] # mAmp
x2 = dados[:, 2] # Tempo de Exposicao

X = np.column_stack((x1, x2))


modelo = RegressionLinearM(X, y)
modelo.fit() #treinamento

print("Intercepto", modelo.beta[0])
print("Angular", modelo.beta[1])
print(f"y= {modelo.beta[0]} + {modelo.beta[1]}*x1 + {modelo.beta[2]}*x2")


y_pred = modelo.predict(X) #predizer os valores de y
#print(y_pred)

def r2_score(y_true, y_prediction):
    numerador = np.sum((y_true - y_prediction) ** 2)
    denominador = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (numerador/denominador)
    return r2

print(r2_score(y, y_pred))

## Plotagem

#print(np.linspace(min(x1), max(x1), 50))
#print(np.linspace(min(x2), max(x2), 50))
x1_grid, x2_grid = np.meshgrid(np.linspace(min(x1), max(x1), 50),
                               np.linspace(min(x2), max(x2), 50))

y_grid = modelo.beta[0] + modelo.beta[1]*x1_grid + modelo.beta[2]*x2_grid
fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(x1, x2, y, color="red", label="dados reais")

#plano da regressão
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
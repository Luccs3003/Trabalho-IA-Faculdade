import numpy as np

class LinearRegression:

    def __init__(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        self.b0 = None
        self.b1 = None

    def fit(self):
        xbar = np.mean(self.x)
        ybar = np.mean(self.y)

        self.b1 = np.sum((self.x - xbar) * (self.y - ybar)) / np.sum((self.x - xbar) ** 2)
        self.b0 = ybar - self.b1 * xbar

        return self

    def predict(self, x_new):
        return self.b0 + self.b1 * np.array(x_new)

    def summary(self):
        print(f"Modelo: y = {self.b0:.4f} + {self.b1:.4f}x")
        print("Intercepto (b0):", round(self.b0, 4))
        print("Coeficiente angular (b1):", round(self.b1, 4))


#Carregar dataset
dados = np.loadtxt("qb_2004.csv", delimiter=",", skiprows=1, usecols=(2, 3))

X = dados[:, 0]  # Jardas por Tentativa
Y = dados[:, 1]  # Pontos Feitos

#(a) Estimativas de mínimos quadrados: encontrar b0 (intercepto) e b1 (inclinação)
print("=" * 45)
print("(a) Estimativas de Mínimos Quadrados")
print("=" * 45)
modelo = LinearRegression(X, Y)
modelo.fit()
modelo.summary()

#(b) Previsão para 7,5 jardas
print("\n(b) Previsão para x = 7.5 jardas:")
print(round(modelo.predict(7.5), 4))

#(c) Mudança associada a -1 jarda/tentativa
print("\n(c) Mudança na pontuação para -1 jarda/tentativa:")
print(round(-modelo.b1, 4))

#(d) Valor ajustado e resíduo para x = 7,21
print("\n(d) Para x = 7.21 jardas:")
y_hat_d = modelo.predict(7.21)
print("Valor ajustado:", round(y_hat_d, 4))

for i in range(len(X)):
    if np.isclose(X[i], 7.21):
        print(f"  Obs {i+1}: y observado = {Y[i]}, residuo = {round(Y[i] - y_hat_d, 4)}")

#(e) Tabela de resíduos
y_hat = modelo.predict(X)
residuos = Y - y_hat

print("\n(e) Tabela de Resíduos")
print(f"{'Obs':>4}  {'y_obs':>8}  {'y_ajust':>10}  {'residuo':>10}")
for i in range(len(Y)):
    print(f"{i+1:>4}  {Y[i]:>8.1f}  {y_hat[i]:>10.4f}  {residuos[i]:>10.4f}")
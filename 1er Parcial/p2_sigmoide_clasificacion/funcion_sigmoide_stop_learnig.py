import numpy as np
import matplotlib.pyplot as plt

# Función sigmoide 
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# Datos de entrenamiento 
X = np.array([[2], [3], [4], [5], [6], [7], [8], [9], [10],
              [11], [12], [13], [14], [15], [16], [17], [18],
              [19], [20]])

Ydes = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1,
                 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

m = len(Ydes)

# Agregar columna de unos para el término bias en la matriz X
X_bias = np.ones((m, 1))
X = np.hstack((X_bias, X))

# Inicialización de los parámetros 
theta = np.array([0.5, 1.0])
lr = 0.01
epocas = 200
i = 0
cont = 0
epsilon = 0.000001
puntosJ = np.zeros(epocas)

# Bucle de entrenamiento 
while i < epocas:
    # Calcular Z = X * theta (incluye bias)
    Z = np.dot(X, theta)
    # Predicción H usando la función sigmoide
    H = sigmoide(Z)

    # Evaluar la función de costo y guardar en el arreglo
    J = (-1/m) * np.sum(Ydes * np.log(H) + (1 - Ydes) * np.log(1 - H))
    puntosJ[i] = J

    # Calcular el gradiente 
    gradiente = (1/m) * np.dot(X.T, (H - Ydes))

    # Actualizar todos los theta (incluyendo theta_0)
    theta = theta - lr * gradiente

    cont = i

    # Criterio de parada por convergencia
    if i > 0 and abs(J - puntosJ[i-1]) < epsilon:
        i = epocas  # salir del bucle

    i += 1

# Recortar el arreglo para que solo tenga los datos útiles
puntosJ = puntosJ[:cont+1]

print("Historial de costo J:", puntosJ)
print("Theta final:", theta)

# Graficar los valores de J a lo largo de las épocas 
plt.plot(range(len(puntosJ)), puntosJ)
plt.title('Función de Costo J vs Épocas')
plt.xlabel('Épocas')
plt.ylabel('Costo J')
plt.grid(True)
plt.show()

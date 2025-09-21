import numpy as np
import matplotlib.pyplot as plt

# Función sigmoide 
def sigmoide(z):
    return 1 / (1 + np.exp(-z))

# Datos de entrenamiento 
X = np.array([[2], [3], [4], [5]])
Ydes = np.array([0, 0, 1, 1])

m = len(Ydes)

# Agregar columna de unos para el término bias en la matriz X
X_bias = np.ones((m, 1))
X = np.hstack((X_bias, X))

# Inicialización de los parámetros 
# -- PRUEBAS --
# theta = np.array([0.5, 1.0])
# lr = 0.01
# epocas = 200

theta = np.array([0.0, 0.0])
lr = 0.1
epocas = 10000


# Bucle de entrenamiento 
for i in range(epocas):
    # Calcular Z = X * theta (incluye bias)
    Z = np.dot(X, theta)
    
    # Predicción H usando la función sigmoide
    H = sigmoide(Z)
    
    # Evaluar la función de costo 
    J = (-1/m) * np.sum(Ydes * np.log(H) + (1 - Ydes) * np.log(1 - H))
    
    # Calcular el gradiente 
    gradiente = (1/m) * np.dot(X.T, (H - Ydes))
    
    # Actualizar todos los theta (incluyendo theta_0)
    theta = theta - lr * gradiente

print(f"theta final: {theta}")

#  ------------ VISTA GRAFICA ------------
# Crear un rango de valores en X (de 0 a 6 por ejemplo)
x_vals = np.linspace(0, 6, 100)
x_bias = np.ones((len(x_vals), 1))
X_plot = np.hstack((x_bias, x_vals.reshape(-1, 1)))

# Calcular las probabilidades con el modelo entrenado
y_probs = sigmoide(np.dot(X_plot, theta))

# Graficar los datos originales
plt.scatter(X[:, 1], Ydes, color='red', label='Datos reales')

# Graficar la curva sigmoide
plt.plot(x_vals, y_probs, color='blue', label='Modelo sigmoide')

# Personalización
plt.xlabel("X")
plt.ylabel("Probabilidad")
plt.title("Regresión logística")
plt.legend()
plt.grid(True)
plt.show()


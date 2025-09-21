import numpy as np
import matplotlib.pyplot as plt

def data_set():
    np.random.seed(0)
    # set 1 de datos - CUADRÁTICA
    x1 = np.linspace(-10, 10, 40)
    y1 = 0.7*x1**2 + 1.7*x1 + 2*np.random.normal(0, 5, x1.shape)

    # set 2 de datos
    x2 = np.linspace(-10, 10, 40)
    y2 = 0.8*x2**2 + 1.7*x2 + 1.75*np.random.normal(0, 5, x2.shape)

    # set 3 de datos
    x3 = np.linspace(-10, 10, 40)
    y3 = 2.1*x3**3 - 25*x3 + 1.3*x3 + 0.8*x3**2 + 1.7*x3 + .8*np.random.normal(0, 5, x3.shape)

    return (x1,y1),(x2,y2),(x3,y3)

(x1,y1),(x2,y2),(x3,y3) = data_set()

def normalización(X):
    return (X-np.min(X))/(np.max(X)-np.min(X))

def desnormalizar(X_norm, X_original):
    return X_norm * (np.max(X_original) - np.min(X_original)) + np.min(X_original)

# Definición de parámetros o hiperparámetros
lr = 0.01
epocas = 45000
Yd = y2
X = x2

# Pedirle al usuario el grado del polinomio a ajustar
print("Introduce el grado del polinomio | Entre 1 y 15")
n = int(input())
B = np.random.rand(n) * 0.1
m = len(Yd)

# Normalización de los vectores
X_norm = normalización(X)
Yd_norm = normalización(Yd)

# Crear una matriz de dimensiones m x n
XX = np.zeros((m, n))
for j in range(n):
    XX[:, j] = X_norm**(j + 1)

historial_ecm = np.zeros(epocas)

# Descenso de gradiente
for t in range(epocas):
    Yobt = B @ XX.T
    ECM = (1/(2*m)) * np.sum((Yobt-Yd_norm)**2)
    historial_ecm[t] = ECM
    
    # Calcular el vector de Betas
    B = B - (lr/m)*np.dot((Yobt-Yd_norm), XX) 
    B[0] = B[0] - (lr/m)*np.sum((Yobt-Yd_norm))  # Valor de beta 0
    
Y = desnormalizar(Yobt, Yd)

# Graficar el ECM
plt.plot(historial_ecm)
plt.title("Evolución del error cuadrático medio")
plt.xlabel("Épocas")
plt.ylabel("ECM")
plt.show()

# Graficación del ajuste del polinomio
plt.scatter(X, Yd, label="Datos originales")
plt.plot(X, Y, label=f"Ajuste de grado {n}", color="green")
plt.legend()
plt.title("Ajuste de polinomio")
plt.show()

# Solicitar al usuario un punto
print("Introduce un valor en X")
x_class = int(input())  # coordenada "x"
print("Introduce un valor en Y")
y_class = int(input())  # coordenada "y"

# Normalizamos el valor X
x_class_norm = (x_class-np.min(X))/(np.max(X)-np.min(X))

# Crear vector XX_class
XX_class_norm = np.zeros(n)
for j in range(n):
    XX_class_norm[j] = x_class_norm**(j + 1)

# Calcular el valor de Y usando la fórmula
y_class_pred = desnormalizar(B @ XX_class_norm, Yd)

# Clasificación
clasificacion = y_class_pred - y_class
if clasificacion > 0:
    print("El punto es clase 0")
else:
    print("El punto es clase 1")

# Imprimir gráfica con nuevo punto
plt.scatter(X, Yd, label="Datos originales")
plt.scatter(x_class, y_class, color="red", label="Nuevo punto clasificado")
plt.plot(X, Y, label=f"Ajuste de grado {n}", color="green")
plt.legend()
plt.title("Clasificación de un nuevo punto")
plt.show()

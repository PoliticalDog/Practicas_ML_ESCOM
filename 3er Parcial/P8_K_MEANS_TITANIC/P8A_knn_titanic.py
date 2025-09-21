import pandas as pd
import numpy as np
import os

def LeerArchivo():
    base_path = os.path.dirname(__file__)  # Carpeta donde está Clasificador.py
    file_path = os.path.join(base_path, "titanic.csv")
    Datos = pd.read_csv(file_path)
    #Datos = pd.read_csv("titanic.csv")
    Cantidad_Datos = len(Datos.values)
    
    Vector_X = np.zeros((Cantidad_Datos, 3))
    Vector_Y = np.zeros(Cantidad_Datos)

    for i in range(Cantidad_Datos):
        # Codificar Y en {-1, 1}
        Vector_Y[i] = -1 if Datos.values[i][1] == 0 else 1
        Vector_X[i] = np.array([Datos.values[i][2], Datos.values[i][5], Datos.values[i][9]])  # [Clase, Edad, Costo]

    Prom_Edad = np.nanmean(Vector_X[:, 1]) 
    Vector_X[np.isnan(Vector_X[:, 1]), 1] = Prom_Edad
    return Vector_X, Vector_Y

def entrenamiento_MSV(X, y):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Los datos de entrada X e y no deben estar vacíos.")
    if len(X.shape) != 2:
        raise ValueError("X debe ser un arreglo bidimensional.")

    n_muestras, n_caracteristicas = X.shape

    # Inicializar parámetros externos
    epocas = 1000
    lr = 0.01

    # Inicializar algunos parámetros internos
    w = np.zeros(n_caracteristicas) + 0.1
    b = 0.1

    vectores_soporte = []
    vectores_soporte_indices = []

    lamda = 1 / epocas
    for epoca in range(epocas):
        for i, x in enumerate(X):
            condicion_margen = y[i] * (np.dot(w, x) + b) >= 1
            if condicion_margen:
                # w no debería modificarse significativamente
                w -= lr * (2 * lamda * w)
            else:
                w -= lr * (2 * lamda * w - np.dot(y[i], x))
                b -= lr * lamda * y[i]
                vectores_soporte.append(x)
                vectores_soporte_indices.append(i)

    # Cálculo del margen del hiperplano
    M = 2 / np.linalg.norm(w)
    return w, b, vectores_soporte, vectores_soporte_indices


def prediccion(X_test, w, b):
    return np.sign(np.dot(X_test, w) + b)


# Entrenamiento
X,Y = LeerArchivo()
w, b, vectores_soporte, vectores_soporte_indices = entrenamiento_MSV(X, Y)

print("Modelo entrenado. Peso w:", w)
print("Intercepto b:", b)


# Datos de prueba
X_test = np.array([[3, 32, 7.75], [1, 49, 76], [1, 54, 51]])  # Ejemplo de datos
predict = prediccion(X_test, w, b)
print("Predicciones para los datos:", predict)



def entrenamiento_MSV(X, y):
    if len(X) == 0 or len(y) == 0:
        raise ValueError("Los datos de entrada X e y no deben estar vacíos.")
    if len(X.shape) != 2:
        raise ValueError("X debe ser un arreglo bidimensional.")

    n_muestras, n_caracteristicas = X.shape

    # Inicializar parámetros externos
    epocas = 1000
    lr = 0.01

    # Inicializar algunos parámetros internos
    w = np.zeros(n_caracteristicas) + 0.1
    b = 0.1

    vectores_soporte = []
    vectores_soporte_indices = []

    lamda = 1 / epocas
    for epoca in range(epocas):
        for i, x in enumerate(X):
            condicion_margen = y[i] * (np.dot(w, x) + b) >= 1
            if condicion_margen:
                # w no debería modificarse significativamente
                w -= lr * (2 * lamda * w)
            else:
                w -= lr * (2 * lamda * w - np.dot(y[i], x))
                b -= lr * lamda * y[i]
                vectores_soporte.append(x)
                vectores_soporte_indices.append(i)

    # Cálculo del margen del hiperplano
    M = 2 / np.linalg.norm(w)
    return w, b, vectores_soporte, vectores_soporte_indices


def prediccion(X_test, w, b):
    return np.sign(np.dot(X_test, w) + b)


# Entrenamiento
X,Y = LeerArchivo()
w, b, vectores_soporte, vectores_soporte_indices = entrenamiento_MSV(X, Y)

print("Modelo entrenado. Peso w:", w)
print("Intercepto b:", b)


# Datos de prueba
X_test = np.array([[3, 32, 7.75], [1, 49, 76], [2, 66, 10.05]])  # Ejemplo de datos
predict = prediccion(X_test, w, b)
print("Predicciones para los datos:", predict)


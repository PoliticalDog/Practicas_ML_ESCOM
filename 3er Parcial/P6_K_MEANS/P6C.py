import random
import numpy as np
import pandas as pd
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
        Vector_Y[i] = Datos.values[i][1]
        if Datos.values[i][4] == 'male':
            sexo = 1
        elif Datos.values[i][4] == 'female':
            sexo = 0
        Vector_X[i] = np.array([Datos.values[i][2], sexo, Datos.values[i][5]])  # [Clase, Sexo, Edad]

    Prom_Edad = np.nanmean(Vector_X[:, 2])
    Vector_X[np.isnan(Vector_X[:, 2]), 2] = Prom_Edad

    return Vector_X, Vector_Y

def inicializar_centroides(data, k):
    num_datos, _ = data.shape
    centroides = [data[random.randint(0, num_datos - 1)]]

    for _ in range(1, k):
        distancias = np.min(
            np.array([np.linalg.norm(data - c, axis=1)**2 for c in centroides]), axis=0
        )
        probabilidades = distancias / np.sum(distancias)
        nuevo_centroide_idx = np.random.choice(num_datos, p=probabilidades)
        centroides.append(data[nuevo_centroide_idx])

    return np.array(centroides)

def kmeans(data, k, epocas):
    centroides = inicializar_centroides(data, k)

    for _ in range(epocas):
        distancias = np.array([[np.linalg.norm(p - c) for c in centroides] for p in data])
        etiquetas = np.argmin(distancias, axis=1)

        nuevos_centroides = []
        for i in range(k):
            puntos_cluster = data[etiquetas == i]
            if len(puntos_cluster) > 0:
                nuevos_centroides.append(np.mean(puntos_cluster, axis=0))
            else:
                nuevos_centroides.append(centroides[i])
        centroides = np.array(nuevos_centroides)

    return etiquetas, centroides

# Leer datos desde el archivo CSV
X, Y = LeerArchivo()

# Definir número de clusters
k = 2 

# Aplicar K-Means
etiquetas, centroides = kmeans(X, k, epocas=100)
predicciones = etiquetas[:100]

# Mostrar las predicciones
for i, pred in enumerate(predicciones, 1):
    print(f"Dato {i+1} {X[i]}: Cluster {pred}")

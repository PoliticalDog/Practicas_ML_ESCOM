import matplotlib.pyplot as plt
import random
import numpy as np

def generar_clases():
    random.seed(42) 
    np.random.seed(42)
    centros = [
        (10, 10), (30, 30), (50, 50), (70, 70),
        (90, 90), (20, 80), (80, 20), (40, 60)
    ]
    datos = []
    for centro_x, centro_y in centros:
        puntos_x = np.random.normal(centro_x, 5, 20)
        puntos_y = np.random.normal(centro_y, 5, 20)
        datos.extend(zip(puntos_x, puntos_y))
    return np.array(datos)

def kmeans(data, k, epocas):
    num_datos, _ = data.shape
    indices_iniciales = random.sample(range(num_datos), k)
    centroides = data[indices_iniciales]

    for _ in range(epocas):
        # Asignar cada punto al centroide más cercano
        distancias = np.array([[np.linalg.norm(p - c) for c in centroides] for p in data])
        etiquetas = np.argmin(distancias, axis=1)

        # Actualizar centroides
        nuevos_centroides = []
        for i in range(k):
            puntos_cluster = data[etiquetas == i]
            if len(puntos_cluster) > 0:
                nuevos_centroides.append(np.mean(puntos_cluster, axis=0))
            else:
                nuevos_centroides.append(centroides[i])
        centroides = np.array(nuevos_centroides)

    return etiquetas, centroides

def graficar_clusters(data, etiquetas, centroides, k):
    colores = plt.cm.get_cmap("tab10", k)
    for i in range(k):
        puntos_cluster = data[etiquetas == i]
        plt.scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], color=colores(i), label=f"Cluster {i+1}")

    plt.scatter(centroides[:, 0], centroides[:, 1], color="black", marker="x", s=100, label="Centroides")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"K-Means con K={k}")
    plt.legend()
    plt.grid()
    plt.show()

# Generar datos
data = generar_clases()

# Ejecutar K-Means para diferentes valores de K
for k in [3, 5, 8, 10]:
    etiquetas, centroides = kmeans(data, k, epocas=100)
    graficar_clusters(data, etiquetas, centroides, k)

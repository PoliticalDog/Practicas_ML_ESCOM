import matplotlib.pyplot as plt
import random
import numpy as np

# Generar datos
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

# Inicializar centroides con K-Means++
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

# K-Means clásico
def kmeans(data, k, epocas):
    num_datos, _ = data.shape
    indices_iniciales = random.sample(range(num_datos), k)
    centroides = data[indices_iniciales]

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

# K-Means++
def kmeans_pp(data, k, epocas):
    num_datos, _ = data.shape
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

# Graficar ambas versiones de K-Means
def graficar_comparativa(data, etiquetas_kmeans, centroides_kmeans, etiquetas_kmeans_pp, centroides_kmeans_pp, k):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    colores = plt.cm.get_cmap("tab10", k)

    # Gráfica K-Means clásico
    for i in range(k):
        puntos_cluster = data[etiquetas_kmeans == i]
        axs[0].scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], color=colores(i), label=f"Cluster {i+1}")
    axs[0].scatter(centroides_kmeans[:, 0], centroides_kmeans[:, 1], color="black", marker="x", s=100, label="Centroides")
    axs[0].set_title(f"K-Means Clásico con K={k}")
    axs[0].set_xlabel("X")
    axs[0].set_ylabel("Y")
    axs[0].legend()
    axs[0].grid()

    # Gráfica K-Means++
    for i in range(k):
        puntos_cluster = data[etiquetas_kmeans_pp == i]
        axs[1].scatter(puntos_cluster[:, 0], puntos_cluster[:, 1], color=colores(i), label=f"Cluster {i+1}")
    axs[1].scatter(centroides_kmeans_pp[:, 0], centroides_kmeans_pp[:, 1], color="black", marker="x", s=100, label="Centroides")
    axs[1].set_title(f"K-Means++ con K={k}")
    axs[1].set_xlabel("X")
    axs[1].set_ylabel("Y")
    axs[1].legend()
    axs[1].grid()

    plt.tight_layout()
    plt.show()

# Generar datos
data = generar_clases()

for k in [3, 5, 8, 10]:
    etiquetas_kmeans, centroides_kmeans = kmeans(data, k, epocas=100)
    etiquetas_kmeans_pp, centroides_kmeans_pp = kmeans_pp(data, k, epocas=100)
    graficar_comparativa(data, etiquetas_kmeans, centroides_kmeans, etiquetas_kmeans_pp, centroides_kmeans_pp, k)

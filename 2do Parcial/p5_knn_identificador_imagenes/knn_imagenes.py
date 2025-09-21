import numpy as np
import matplotlib.pyplot as plt

# Definición de la base X
base_X = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Definición de la base B
base_B = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
])

# Definición de la base D
base_D = np.zeros((10, 10))
base_D[1:9, 1] = 1
base_D[2:8, 8] = 1
base_D[1, 2:8] = 1
base_D[8, 2:8] = 1

# Definición de la base E
base_E = np.zeros((10, 10))
base_E[1:8, 1] = 1
base_E[1, 2:8] = 1
base_E[4, 2:8] = 1
base_E[7, 2:8] = 1

# Generar patrones ruidosos
patterns_D = [base_D + np.random.uniform(-0.4, 0.4, base_D.shape) for _ in range(10)]
patterns_E = [base_E + np.random.uniform(-0.4, 0.4, base_E.shape) for _ in range(10)]
patterns_B = [base_B + np.random.uniform(-0.4, 0.4, base_B.shape) for _ in range(10)]
patterns_X = [base_X + np.random.uniform(-0.4, 0.4, base_X.shape) for _ in range(10)]

# Función para realizar SVD
def svd(matriz):
    U, S, VT = np.linalg.svd(matriz)
    return U, S, VT

# Función para visualizar patrones
def visualizar_patrones(patrones, titulo):
    num_patrones = len(patrones)
    fig, axes = plt.subplots(1, num_patrones, figsize=(12, 6))
    if num_patrones == 1:
        axes = [axes]
    for i in range(num_patrones):
        axes[i].imshow(patrones[i], cmap='gray')
        axes[i].axis('off')
    plt.suptitle(titulo)
    plt.show()

# Aplicar SVD a todos los patrones
dimensiones = 5
X = []
for patron in patterns_D + patterns_E + patterns_X + patterns_B:
    U, S, VT = svd(patron)
    X += [U[:, :dimensiones] @ np.diag(S[:dimensiones])]

# Etiquetas
Y = [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10

# Función distancia euclidiana
def distancia_euclidiana(X, x_test):
    distancias = [0] * len(X)
    for i in range(len(X)):
        distancias[i] = np.sqrt(np.sum((X[i] - x_test) ** 2))
    return distancias

# Ordenar lista con posiciones
def ordenar_lista_con_posiciones(lista):
    lista_con_indices = [(lista[i], i) for i in range(len(lista))]
    for i in range(len(lista_con_indices)):
        min_index = i
        for j in range(i + 1, len(lista_con_indices)):
            if lista_con_indices[j][0] < lista_con_indices[min_index][0]:
                min_index = j
        lista_con_indices[i], lista_con_indices[min_index] = lista_con_indices[min_index], lista_con_indices[i]
    return lista_con_indices

# Implementación del KNN sin librerías externas
def KNN(k, lista_con_indices_ordenados, Y):
    clase1 = clase2 = clase3 = clase4 = 0
    for j in range(k):
        index = lista_con_indices_ordenados[j][1]
        if Y[index] == 0:
            clase1 += 1
        elif Y[index] == 1:
            clase2 += 1
        elif Y[index] == 2:
            clase3 += 1
        elif Y[index] == 3:
            clase4 += 1

    if clase1 >= clase2 and clase1 >= clase3 and clase1 >= clase4:
        return 0
    elif clase2 >= clase1 and clase2 >= clase3 and clase2 >= clase4:
        return 1
    elif clase3 >= clase1 and clase3 >= clase2 and clase3 >= clase4:
        return 2
    else:
        return 3

# Visualización de los resultados
visualizar_patrones([base_D], "Patrón Base D")
visualizar_patrones([base_E], "Patrón Base E")
visualizar_patrones([base_B], "Patrón Base B")
visualizar_patrones([base_X], "Patrón Base X")
visualizar_patrones(patterns_D, "Patrones de la letra D")
visualizar_patrones(patterns_E, "Patrones de la letra E")
visualizar_patrones(patterns_B, "Patrones de la letra B")
visualizar_patrones(patterns_X, "Patrones de la letra X")

# Clasificación con KNN sobre patrones ruidosos
for i in range(4):
    U, S, _ = svd(patterns_D[i])
    test_D = U[:, :dimensiones] @ np.diag(S[:dimensiones])

    U, S, _ = svd(patterns_E[i])
    test_E = U[:, :dimensiones] @ np.diag(S[:dimensiones])

    U, S, _ = svd(patterns_B[i])
    test_B = U[:, :dimensiones] @ np.diag(S[:dimensiones])

    U, S, _ = svd(patterns_X[i])
    test_X = U[:, :dimensiones] @ np.diag(S[:dimensiones])

    # Distancias
    distancias_D = distancia_euclidiana(X, test_D)
    distancias_E = distancia_euclidiana(X, test_E)
    distancias_B = distancia_euclidiana(X, test_B)
    distancias_X = distancia_euclidiana(X, test_X)

    ordenados_D = ordenar_lista_con_posiciones(distancias_D)
    ordenados_E = ordenar_lista_con_posiciones(distancias_E)
    ordenados_B = ordenar_lista_con_posiciones(distancias_B)
    ordenados_X = ordenar_lista_con_posiciones(distancias_X)

    K = 6
    clase_D = KNN(K, ordenados_D, Y)
    clase_E = KNN(K, ordenados_E, Y)
    clase_B = KNN(K, ordenados_B, Y)
    clase_X = KNN(K, ordenados_X, Y)

    visualizar_patrones([patterns_D[i]], f"Patrón ruidoso D #{i+1} - Clasificado como {['D','E','B','X'][clase_D]}")
    visualizar_patrones([patterns_E[i]], f"Patrón ruidoso E #{i+1} - Clasificado como {['D','E','B','X'][clase_E]}")
    visualizar_patrones([patterns_B[i]], f"Patrón ruidoso B #{i+1} - Clasificado como {['D','E','B','X'][clase_B]}")
    visualizar_patrones([patterns_X[i]], f"Patrón ruidoso X #{i+1} - Clasificado como {['D','E','B','X'][clase_X]}")

import numpy as np
import matplotlib.pyplot as plt

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
patterns_D = [base_D + np.random.uniform(-0.1, 0.1, base_D.shape) for _ in range(20)]
patterns_E = [base_E + np.random.uniform(-0.1, 0.1, base_E.shape) for _ in range(20)]

# Agregar ruido a patrones individuales
noisy_pattern_D = [base_D + np.random.uniform(-0.1, 0.1, base_D.shape) for _ in range(2)]
noisy_pattern_E = [base_E + np.random.uniform(-0.1, 0.1, base_E.shape) for _ in range(2)]

# Función para realizar SVD
def svd(matriz):
    U, S, VT = np.linalg.svd(matriz)
    return U, S, VT

# Aplicar SVD a todos los patrones
dimensiones = 5
X = []
for patron in patterns_D + patterns_E:
    U, S, VT = svd(patron)
    X.append(U[:, :dimensiones] @ np.diag(S[:dimensiones]))

X = np.array(X).reshape(len(X), -1)  # Aplanar los datos
Y = np.array([1] * 20 + [-1] * 20)  # Etiquetas (+1 para D y -1 para E)

# SVM usando funciones
def entrenar_svm(X, Y, learning_rate=0.001, lambda_param=0.01, epochs=1000):
    num_samples, num_features = X.shape
    w = np.zeros(num_features)
    b = 0

    for _ in range(epochs):
        for i in range(num_samples):
            if Y[i] * (np.dot(X[i], w) - b) < 1:
                w -= learning_rate * (2 * lambda_param * w - np.dot(X[i], Y[i]))
                b -= learning_rate * Y[i]
            else:
                w -= learning_rate * (2 * lambda_param * w)

    return w, b

def predecir_svm(X, w, b):
    return np.sign(np.dot(X, w) - b)

# Entrenar el modelo
w, b = entrenar_svm(X, Y, learning_rate=0.001, lambda_param=0.01, epochs=1000)

# Visualización de patrones
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

#Visualización de los resultados
visualizar_patrones([base_D], 'Patrón Base D')
visualizar_patrones([base_E], 'Patrón Base E')
visualizar_patrones(patterns_D, "Patrones de la letra D")
visualizar_patrones(patterns_E, "Patrones de la letra E")

# Clasificación de patrones ruidosos
for i in range(2):
    # Aplicar SVD a los patrones ruidosos
    U_D, S_D, _ = svd(noisy_pattern_D[i])
    test_D = U_D[:, :dimensiones] @ np.diag(S_D[:dimensiones])
    test_D_flat = test_D.flatten()

    U_E, S_E, _ = svd(noisy_pattern_E[i])
    test_E = U_E[:, :dimensiones] @ np.diag(S_E[:dimensiones])
    test_E_flat = test_E.flatten()

    # Clasificar usando SVM
    clase_D = predecir_svm([test_D_flat], w, b)[0]
    clase_E = predecir_svm([test_E_flat], w, b)[0]

    visualizar_patrones([noisy_pattern_D[i]], f"Patrón ruidoso D #{i+1} - Clasificado como {'D' if clase_D == 1 else 'E'}")
    visualizar_patrones([noisy_pattern_E[i]], f"Patrón ruidoso E #{i+1} - Clasificado como {'D' if clase_E == 1 else 'E'}")
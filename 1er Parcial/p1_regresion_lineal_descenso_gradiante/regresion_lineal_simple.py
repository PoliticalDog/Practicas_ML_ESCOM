import numpy as np

# Definición de los datos de entrada
X = np.array([2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019], dtype=np.float64)
Yd = np.array([5.1, 5.4, 5.7, 6, 6.9, 7.5, 8.8, 9.8, 10.5, 9.5], dtype=np.float64)

# Normalización de X
X_min = X.min()
X_max = X.max()
X_norm = (X - X_min) / (X_max - X_min)

# Definición de los parámetros
a = 0.0  # Pendiente inicial
b = 0.0  # Intersección inicial
lr = 0.01  # Tasa de aprendizaje
epocas = 1000  # Número de iteraciones
m = len(Yd)  # Número de ejemplos

# Inicialización de las predicciones y el historial del costo
Yobt = np.zeros(m)
costo = []

for i in range(epocas):
    # Predicciones actuales
    Yobt = a * X_norm + b
    
    # Cálculo del error
    error = Yobt - Yd
    
    # Actualización de los parámetros usando descenso de gradiente
    a -= (lr / m) * np.dot(error, X_norm)
    b -= (lr / m) * np.sum(error)
    
    # Cálculo del Error Cuadrático Medio (ECM)
    ECM = (1 / (2 * m)) * np.dot(error, error)
    costo.append(ECM)

try:
     # Solicitar al usuario que ingrese un nuevo valor de X
    año = input("Ingrese un año a predecir: ")
    x = float(año)
        
    # Normalizar el nuevo X usando los mismos parámetros de normalización
    nuevo_X_norm = (x- X_min) / (X_max - X_min)
        
    # Calcular Y_pred utilizando los parámetros entrenados
    Y_pred_nuevo = a * nuevo_X_norm + b
        
    print(f"Predicción para el año: {x}: Y = {Y_pred_nuevo:.4f}")
except ValueError:
    print("Por favor, revise que haya ingresado correctamente el año")
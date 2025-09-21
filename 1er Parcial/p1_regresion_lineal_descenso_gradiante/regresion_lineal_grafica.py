import numpy as np
import matplotlib.pyplot as plt

# ========================================
# 1) DEFINICIÓN DE DATOS DE ENTRADA
# ========================================
X = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
Yd = np.array([2, 3, 5, 4, 7, 6, 10, 9, 13, 12])
Yinv = np.array([1.50, 2.70, 5.50, 4.50, 7.50, 6.50, 10.50, 9.50, 12.90, 12.20])

# ========================================
# 2) DEFINICIÓN DE PARÁMETROS
# ========================================
a = 1.19
b = -0.063
lr = 0.001  # Learning rate
m = len(Yd)  # Número de muestras
epocas = 1000

# Inicialización de arrays
Yobt = np.zeros(m)
ClaseDes = np.zeros(m)
ClaseCalc = np.zeros(m)

# ========================================
# 3) ENTRENAMIENTO DEL MODELO (DESCENSO DEL GRADIENTE)
# ========================================
for i in range(epocas):
    # Calcular predicciones
    Yobt = a * X + b
    
    # Actualizar parámetros usando descenso del gradiente
    a = a - (lr / m) * np.sum((Yobt - Yd) * X)
    b = b - (lr / m) * np.sum((Yobt - Yd))

# Calcular error cuadrático medio final
ECM = (1 / m) * np.sum((Yobt - Yd) ** 2)
print(f"ECM final: {ECM}")

# ========================================
# 4) CLASIFICACIÓN DE DATOS
# ========================================
# Clasificar datos deseados (Yd)
for i in range(len(Yd)):
    if (Yd[i] - Yobt[i]) < 0:
        ClaseDes[i] = 1
    else:
        ClaseDes[i] = 0

# Clasificar datos de validación (Yinv)
for i in range(len(Yinv)):
    if (Yinv[i] - Yobt[i]) < 0:
        ClaseCalc[i] = 1
    else:
        ClaseCalc[i] = 0

print(f"Clasificación de Yd: {ClaseDes}")
print(f"Clasificación de Yinv: {ClaseCalc}")

# ========================================
# 5) CÁLCULO DE MÉTRICAS DE EVALUACIÓN
# ========================================
# Inicializar contadores de matriz de confusión
vp = 0  # Verdaderos positivos
vn = 0  # Verdaderos negativos
fp = 0  # Falsos positivos
fn = 0  # Falsos negativos

# Verificar que las dimensiones coincidan
if len(Yd) == len(Yinv):
    for i in range(len(Yd)):
        if ClaseCalc[i] == 1 and ClaseDes[i] == 1:
            vp += 1
        elif ClaseCalc[i] == 0 and ClaseDes[i] == 0:
            vn += 1
        elif ClaseCalc[i] == 1 and ClaseDes[i] == 0:
            fp += 1
        elif ClaseCalc[i] == 0 and ClaseDes[i] == 1:
            fn += 1

# Calcular métricas
precision = vp / (vp + fp) if (vp + fp) > 0 else 0
exactitud = (vp + vn) / (fp + vp + fn + vn)
recall = vp / (vp + fn) if (vp + fn) > 0 else 0
f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"Vp={vp}, Vn={vn}, Fn={fn}, Fp={fp}")
print(f"Precisión={precision:.4f}, Exactitud={exactitud:.4f}, Recall={recall:.4f}, F1-Score={f1_score:.4f}")

# ========================================
# 6) CLASIFICACIÓN DE DATOS DE PRUEBA
# ========================================
Xtest = np.array([1.1, 2.4, 3.2, 9.5, 10.3])
Ytest = np.array([1.50, 2.70, 3.90, 11.10, 12.45])

# Clasificar datos de prueba
ClaseTest = np.zeros(len(Xtest))
for i in range(len(Xtest)):
    if (Ytest[i] - (a * Xtest[i] + b)) < 0:
        ClaseTest[i] = 1
    else:
        ClaseTest[i] = 0

print(f"Clasificación de YTest: {ClaseTest}")

# ========================================
# 7) VISUALIZACIÓN DE RESULTADOS
# ========================================
# Crear subplots
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

# Primer gráfico: Yd vs Yobt
ax1.plot(X, Yd, 'bo', label='Yd (Datos Deseados)', markersize=8)
ax1.plot(X, Yobt, 'r*-', label='Yobt (Datos Obtenidos)', markersize=8, linewidth=2)
ax1.set_title('Comparación entre Yd y Yobt', fontsize=14, fontweight='bold')
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# Segundo gráfico: Yinv vs Yobt
ax2.plot(X, Yinv, 'go', label='Yinv (Datos de Validación)', markersize=8)
ax2.plot(X, Yobt, 'r*-', label='Yobt (Datos Obtenidos)', markersize=8, linewidth=2)
ax2.set_title('Comparación entre Yinv y Yobt', fontsize=14, fontweight='bold')
ax2.set_xlabel('X', fontsize=12)
ax2.set_ylabel('Y', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# Tercer gráfico: Ytest vs Ypred_test
ax3.plot(Xtest, Ytest, 'mo', label='Ytest (Datos de Test)', markersize=8)
Ypred_test = a * Xtest + b
ax3.plot(Xtest, Ypred_test, 'c*-', label='Ypred_test (Predicciones)', markersize=8, linewidth=2)
ax3.plot(X, Yobt, 'r--', label='Línea de Regresión', alpha=0.7, linewidth=1)
ax3.set_title('Comparación entre Ytest y Predicciones', fontsize=14, fontweight='bold')
ax3.set_xlabel('X', fontsize=12)
ax3.set_ylabel('Y', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

# Ajustar espacios entre subplots
plt.tight_layout()
plt.show()

# ========================================
# 8) RESUMEN DE RESULTADOS
# ========================================
print("\n" + "="*50)
print("RESUMEN DE RESULTADOS")
print("="*50)
print(f"Parámetros finales del modelo:")
print(f"  a (pendiente): {a:.6f}")
print(f"  b (intercepto): {b:.6f}")
print(f"Error Cuadrático Medio: {ECM:.6f}")
print(f"Ecuación de la recta: y = {a:.4f}x + {b:.4f}")
print("="*50) 
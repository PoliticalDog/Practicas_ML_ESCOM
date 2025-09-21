import numpy as np
import time

#Función para calcular a y b
def RL_AP(a, b, lr):
    inicio = time.time()
    Yobt = np.zeros(m)
    for i in range(epocas):
        #Calculamos Yobt
        Yobt = a*X+b
        a = a - (lr/m)*np.sum((Yobt - Yd)*X)
        b = b - (lr/m)*np.sum((Yobt - Yd))
        ECM =(1/(2*m))*np.sum((Yobt-Yd)**2)
        #print(ECM)
    fin = time.time()
    print(f"Learning Rate: {lr}")
    print(f"Valor de a usando actualización de peso {a}")
    print(f"Valor de a usando actualización de peso: {b}")
    print(f"Tiempo de ejecución: {fin-inicio}")

##Regresion lineal via actualizacion de peso
#1)Definir datos de entradas
X = np.array([1,2,3,4,5,6,7,8,9,10])
Yd = np.array([0.8, 2.95, 2.3, 3.6, 5.2, 5.3, 6.1, 5.9, 7.6, 9])


#2) Definir parametros
a = 0.7
b = 0.9
lr1 = 0.01
lr2 = 0.03
lr3 = 0.05
m = len(Yd)
epocas = 1000000

#Calculo de a y b
RL_AP(a, b, lr1)
RL_AP(a, b, lr2)
RL_AP(a, b, lr3)

#Práctica 1, ejercicio 1
# Datos de entrada
X = np.array([1,2,3,4,5,6,7,8,9,10])
Yd = np.array([0.8, 2.95, 2.3, 3.6, 5.2, 5.3, 6.1, 5.9, 7.6, 9])
n = len(Yd)

inicio = time.time()
# Cálculo de las sumatorias
sum_X = np.sum(X)
sum_Y = np.sum(Yd)
sum_XY = np.sum(X * Yd)
sum_X_cuadrada = np.sum(X**2)

# Cálculo de la pendiente (m) y el intercepto (b) usando las fórmulas correctas
a_math = (n * sum_XY - sum_X * sum_Y) / (n * sum_X_cuadrada - sum_X**2)
b_math = (sum_Y - a_math * sum_X) / n
fin = time.time()

print(f"Valor de a usando formulas matematicas: {a_math}")
print(f"Valor de b usando formulas matematicas: {b_math}")
print(f"Tiempo de ejecución: {fin-inicio}")
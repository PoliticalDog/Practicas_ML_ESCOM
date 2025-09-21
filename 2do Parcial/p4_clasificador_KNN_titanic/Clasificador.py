import pandas as pd
import numpy as np
import os

def LeerArchivo():
    base_path = os.path.dirname(__file__)  # Carpeta donde está Clasificador.py
    file_path = os.path.join(base_path, "titanic.csv")
    Datos = pd.read_csv(file_path)
    #Datos = pd.read_csv("titanic.csv")
    Cantidad_Datos = len(Datos.values)
    #Cantidad_Datos = 10
    
    Vector_X = np.zeros((Cantidad_Datos,3))
    Vector_Y = np.zeros(Cantidad_Datos)

    #Crear vector de caracteristicas y vector Y
    # Caracteristica[Indice], Clase[2], Sexo[4], Edad[5]
    # Y en el indice [1]
    for i in range(Cantidad_Datos):
        #print("Nombre: {nombre} Clase: {clase} Sexo: {sexo} Edad: {edad}".format(nombre=Datos.values[i][3], clase=Datos.values[i][2], sexo=Datos.values[i][4], edad=Datos.values[i][5]))
        Vector_Y[i] = Datos.values[i][1]

        #Convertir string a valor numerico
        if Datos.values[i][4] == 'male':
            sexo = 1
        elif Datos.values[i][4] == 'female':
            sexo = 0
        Vector_X[i] =np.array([Datos.values[i][2], sexo, Datos.values[i][5]]) #Vector de caracteristicas [Clase, Sexo, Edad]

    #Calcular los promedios de cada X
    Prom_Clase = np.mean(Vector_X[:,0])
    Prom_Sexo = np.mean(Vector_X[:,1])
    Prom_Edad = np.nanmean(Vector_X[:,2])

    #Remplazar los valores nan en edad por el promedio de edad
    Vector_X[np.isnan(Vector_X[:, 2]), 2] = Prom_Edad

    return Vector_X, Vector_Y #Regresa el vector de caracteristicas con puros valores numericos

def DistanciaEuclidiana(X_Test, Vector_X):
    d = (Vector_X-X_Test)**2
    Distancias = np.zeros((len(d)))
    for i in range(len(d)): 
        Distancias[i] = np.sqrt(np.sum(d[i]))
    return Distancias

def QuicksortConIndice(Arreglo):
    def Quicksort(Arreglo, Indices):
        if len(Arreglo) <= 1:
            return Arreglo, Indices
        
        # Elegir un Pivote aleatorio
        PivoteIdx = np.random.randint(len(Arreglo))
        Pivote = Arreglo[PivoteIdx]

        # Separar elementos en Menores, Iguales y Mayores al Pivote
        Menor, Igual, Mayor = [], [], []
        MenorIdx, IgualIdx, MayorIdx = [], [], []

        for i, val in enumerate(Arreglo):
            if val < Pivote:
                Menor.append(val)
                MenorIdx.append(Indices[i])
            elif val == Pivote:
                Igual.append(val)
                IgualIdx.append(Indices[i])
            else:
                Mayor.append(val)
                MayorIdx.append(Indices[i])
        
        # Recursión en particiones
        OrdenadoMenor, OrdenadoMenorIdx = Quicksort(np.array(Menor), MenorIdx)
        OrdenadoMayor, OrdenadoMayorIdx = Quicksort(np.array(Mayor), MayorIdx)

        # Combinar resultados
        ArregloOrdenado = np.concatenate([OrdenadoMenor, Igual, OrdenadoMayor])
        IdxOrdenado = OrdenadoMenorIdx + IgualIdx + OrdenadoMayorIdx
        
        return ArregloOrdenado, IdxOrdenado

    # Índices iniciales
    IdxIniciales = list(range(len(Arreglo)))

    # Aplicar Quicksort
    ArregloOrdenado, IdxOrdenado = Quicksort(Arreglo, IdxIniciales)
    return ArregloOrdenado, IdxOrdenado

def KNN(K, Indices, Vector_Y):
    ContClase0 = 0
    ContClase1 = 0
    for i in range(0, K):
        if Vector_Y[Indices[i]] == 0:
            ContClase0 += 1
            #print("Muere")
        elif Vector_Y[Indices[i]] == 1:
            ContClase1 += 1
    print("Clase 0 = {}, Clase 1 = {}".format(ContClase0, ContClase1))
    if ContClase1 > ContClase0:
        return 1
    else:
        return 0
    


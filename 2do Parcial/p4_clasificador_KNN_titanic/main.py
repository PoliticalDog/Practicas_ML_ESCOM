import pandas as pd
import Clasificador
import tkinter as tk
from tkinter import ttk
import sys

Sexo = {
    "Mujer": 0,
    "Hombre": 1,
}

Clase = {
    "Baja": 1,
    "Media": 2,
    "Alta": 3
}

class RedirectText:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)  # Scroll automáticamente al final

    def flush(self):
        pass  # No se necesita implementar para este caso

class App:
    def __init__(self, root):
        self.VectorXOriginal, self.VectorYOriginal = Clasificador.LeerArchivo()

        self.root = root
        self.root.config(width=1200, height=500)
        self.root.title("Practica para saber si te mueres en el Titanic")
        self.NDatEnt = 0
        self.NDatTest = 0
        self.ClasePasajero = 1
        self.SexoPas = 0
        self.EdadPas = 0
        
        # Sección para el número de datos de entrenamiento
        self.EtiquetaDatEnt = tk.Label(self.root, text="Cantidad de datos de entrenamiento:")
        self.EtiquetaDatEnt.place(x=14, y=18)
        self.CajaDatEnt = tk.Entry(self.root, width=4)
        self.CajaDatEnt.insert(0, len(self.VectorXOriginal))
        self.CajaDatEnt.place(x=215, y=20)
        BotonCajaDatEnt = tk.Button(self.root, text="OK", command=self.GetDatEnt)
        BotonCajaDatEnt.place(x=245, y=18)

        # Sección para el número de datos de prueba
        self.EtiquetaDatTest = tk.Label(self.root, text="Cantidad de datos de prueba:")
        self.EtiquetaDatTest.place(x=55, y=48)
        self.CajaDatTest = tk.Entry(self.root, width=4)
        self.CajaDatTest.insert(0, 0)
        self.CajaDatTest.place(x=215, y=50)
        BotonDatTest = tk.Button(self.root, text="OK", command=self.GetDatTest)
        BotonDatTest.place(x=245, y=48)

        # Sección clasificar uno
        self.CajaK = tk.Entry(self.root, width=5)
        self.CajaK.insert(0, "K")
        self.CajaK.place(x=300, y=50)
        self.LisClase = ttk.Combobox(self.root, values=list(Clase.keys()), state="readonly")
        self.LisClase.place(x=300, y=20)
        self.LisClase.set("Clase")
        self.LisSexo = ttk.Combobox(self.root, values=list(Sexo.keys()), state="readonly")
        self.LisSexo.place(x=450, y=20)
        self.LisSexo.set("Sexo")
        self.CajaEdad = tk.Entry(self.root, width=5)
        self.CajaEdad.insert(0, "Edad")
        self.CajaEdad.place(x=600, y=21)
        BotonClasificarUno = tk.Button(self.root, text="Clasificar uno", command=self.ClasificarUno)
        BotonClasificarUno.place(x=400, y=50)

        # Clasificar varios
        BotonClasificarUno = tk.Button(self.root, text="Clasificar varios", command=self.ClasificarVarios)
        BotonClasificarUno.place(x=500, y=50)

        # Sección resultados
        self.ResultadoText = tk.Text(self.root, width=80, height=15, wrap=tk.WORD)
        self.ResultadoText.place(x=50, y=100)

        # Redirigir stdout a la caja de texto
        self.stdout_redirect = RedirectText(self.ResultadoText)
        sys.stdout = self.stdout_redirect

    def GetDatEnt(self):
        self.NDatEnt = int(self.CajaDatEnt.get())
        self.CajaDatTest.delete(0, tk.END)
        self.CajaDatTest.insert(0, len(self.VectorXOriginal) - self.NDatEnt)

    def GetDatTest(self):
        self.NDatTest = int(self.CajaDatTest.get())
        self.CajaDatEnt.delete(0, tk.END)
        self.CajaDatEnt.insert(0, len(self.VectorXOriginal) - self.NDatTest)

    def ClasificarUno(self):
        self.ResultadoText.delete("1.0", tk.END)
        self.NDatEnt = int(self.CajaDatEnt.get())
        self.NDatTest = int(self.CajaDatTest.get())
        self.ClasePasajero = Clase[self.LisClase.get()]
        self.SexoPas = Sexo[self.LisSexo.get()]
        self.EdadPas = int(self.CajaEdad.get())

        K = int(self.CajaK.get())
        X_Test = [self.ClasePasajero, self.SexoPas, self.EdadPas]
        X_Car = self.VectorXOriginal[0:self.NDatEnt]

        Distancias = Clasificador.DistanciaEuclidiana(X_Test, X_Car)
        DistanciasOrden, IdxOrden = Clasificador.QuicksortConIndice(Distancias)

        Resultado = Clasificador.KNN(K, IdxOrden, self.VectorYOriginal)
        print("X_test = {}, Y = {}".format(X_Test, Resultado))
        for i in range(K):
            print(f"Dis = {DistanciasOrden[i]:<10.4f} X = {str(self.VectorXOriginal[IdxOrden[i]]):<17} Y = {str(self.VectorYOriginal[IdxOrden[i]]):<15}")
    
    def ClasificarVarios(self):
        Datos = pd.read_csv("titanic.csv")
        self.ResultadoText.delete("1.0", tk.END)
        self.NDatEnt = int(self.CajaDatEnt.get())
        self.NDatTest = int(self.CajaDatTest.get())

        K = int(self.CajaK.get())
        X_Car = self.VectorXOriginal[0:self.NDatEnt]
        X_Varios = self.VectorXOriginal[self.NDatEnt:]

        for i in range(len(X_Varios)):
            Distancias = Clasificador.DistanciaEuclidiana(X_Varios[i], X_Car)
            DistanciasOrden, IdxOrden = Clasificador.QuicksortConIndice(Distancias)
            Resultado = Clasificador.KNN(K, IdxOrden, self.VectorYOriginal)
            #print("X_test = {}, Y = {}".format(X_Test, Resultado))
            print(Datos.values[self.NDatEnt+i][3])
            print(X_Varios[i], Resultado)
            print(" ")



if __name__ == '__main__':
    Ventana = tk.Tk()
    app = App(Ventana)
    Ventana.mainloop()

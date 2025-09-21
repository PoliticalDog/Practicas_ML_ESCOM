#import P2B
import funcion_sigmoide_stop_learnig as P2B
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def predecir_valor(x_val):
    x1, x2 = map(float, x_val.replace(",", " ").split())
    x_input = np.array([x1, x2])  # Agregar el bias si es necesario
    z_val = np.dot(x_input, P2B.theta)
    h_val = P2B.sigmoide(z_val)
    return h_val


# Crear la interfaz de tkinter y mostrar la gráfica en ella
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Predicción con Sigmoide para dos características")

        # Crear la figura para la gráfica de matplotlib
        self.fig, self.ax = plt.subplots()
        self.ax.plot(range(len(P2B.puntosJ)), P2B.puntosJ)
        self.ax.set_title('Función de Costo J vs Épocas')
        self.ax.set_xlabel('Épocas')
        self.ax.set_ylabel('Costo J')
        self.ax.grid(True)

        # Mostrar la gráfica en el widget tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # Crear el cuadro de entrada para pedir valores
        self.label = tk.Label(
            root,
            text="Ingrese dos valores de X separados por espacio o coma (e.g. 2, 3):"
        )
        self.label.pack(pady=10)

        self.entry = tk.Entry(root)
        self.entry.pack(pady=5)

        # Botón para hacer la predicción
        self.button = tk.Button(
            root,
            text="Predecir",
            command=self.realizar_prediccion
        )
        self.button.pack(pady=10)

    def realizar_prediccion(self):
        x_val = self.entry.get()
        prediccion = predecir_valor(x_val)
        print("Predicción:", prediccion)

        if prediccion < 0.5:
            messagebox.showinfo("Resultado", "Clase 1")
        else:
            messagebox.showinfo("Resultado", "Clase 0")


# Inicializar la app
root = tk.Tk()
app = App(root)
root.mainloop()

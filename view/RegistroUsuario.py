import sys
import os
import subprocess
# Agrega la ruta de la raíz del proyecto (un nivel arriba de View/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tkinter as tk
import customtkinter as ctk
from tkinter import messagebox
from model.usuario import Usuario as class_usuario
from controller.UsuarioController import registrar_usuario

# =========================
# Configuración del tema SENA
# =========================
ctk.set_appearance_mode("light")  # Modo claro (blanco)
SENA_VERDE = "#39A900"
TEXTO_OSCURO = "#333333"
FONDO_BLANCO = "#FFFFFF"
boton_activo = None


# =========================
# Funciones
# =========================
def registrar():
    nombre = entry_nombre.get().strip()
    documento = entry_documento.get().strip()
    telefono = entry_telefono.get().strip()
    correo = entry_correo.get().strip()
    
    if not nombre or not documento or not telefono or not correo:
        messagebox.showwarning("Campos vacíos", "Por favor, completa todos los campos.")
    else:
        usuario = class_usuario(nombre, documento, telefono, correo)
        if registrar_usuario(usuario):
            messagebox.showinfo("Registro exitoso", "Usuario registrado correctamente.")
            limpiar_campos()
        else:
            messagebox.showerror("Error", "No se pudo registrar el usuario.")

def limpiar_campos():
    entry_nombre.delete(0, ctk.END)
    entry_documento.delete(0, ctk.END)
    entry_telefono.delete(0, ctk.END)
    entry_correo.delete(0, ctk.END)

# =========================
# Ventana principal
# =========================
ventana = ctk.CTk()
ventana.title("Formulario de Registro de Usuario")
ventana.geometry("900x600+500+100")  # Aumentamos el ancho por el menú
ventana.configure(bg=FONDO_BLANCO)

ruta_base = os.path.dirname(os.path.abspath(__file__))  # Carpeta del archivo actual
ruta_logo = os.path.join(ruta_base, "logoSena.png")
foto = tk.PhotoImage(file=ruta_logo)
ventana.iconphoto(True, foto)

# =========================
# Frame contenedor principal (horizontal)
# =========================
main_container = ctk.CTkFrame(ventana, fg_color=FONDO_BLANCO)
main_container.pack(fill="both", expand=True)

# =========================
# Sidebar (menú lateral izquierdo)
# =========================
sidebar = ctk.CTkFrame(main_container, width=180, fg_color="#F0F0F0", corner_radius=0)
sidebar.pack(side="left", fill="y")

# Título del menú
menu_label = ctk.CTkLabel(
    sidebar, 
    text="Menú",
    font=ctk.CTkFont(size=18, weight="bold"),
    text_color=SENA_VERDE
)
menu_label.pack(pady=(20, 10))

def ir_a_page_principal():
    ventana.destroy()
    subprocess.Popen([sys.executable, os.path.abspath("Main.py")])

# Botones del menú
def ir_a_usuario():
    activar_boton(btn_usuario)
    ventana.destroy()  # Cierra esta ventana
    subprocess.Popen([sys.executable, os.path.abspath("RegistroUsuario.py")])  # Abre la pagina de usuarios

def ir_a_vehiculos():
    activar_boton(btn_vehiculo)
    ventana.destroy()  # Cierra esta ventana
    subprocess.Popen([sys.executable, os.path.abspath("RegistrarVehiculo.py")])  # Abre la pagina de vehiculos


btn_menu = ctk.CTkButton(
    sidebar,
    text="Pagina Principal",
    command=ir_a_page_principal,
    fg_color=SENA_VERDE,
    hover_color="#00304D",
    text_color="white",
    corner_radius=10,
    font=ctk.CTkFont(size=14, weight="bold"),
    width=150,
    height=40
)
btn_menu.pack(pady=10, padx=9)

btn_usuario = ctk.CTkButton(
    sidebar, 
    text="Registro Usuario", 
    command=ir_a_usuario,
    fg_color=SENA_VERDE,
    hover_color="#00304D",
    text_color="white",
    corner_radius=10,
    font=ctk.CTkFont(size=14, weight="bold"),
    width=150,
    height=40
)
btn_usuario.pack(pady=10, padx=9)  # <-- Espaciado lateral añadido

btn_vehiculo = ctk.CTkButton(
    sidebar, 
    text="Registro Vehículo", 
    command=ir_a_vehiculos,
    fg_color=SENA_VERDE,
    hover_color="#00304D",
    text_color="white",
    corner_radius=10,
    font=ctk.CTkFont(size=14, weight="bold"),
    width=150,
    height=40
)
btn_vehiculo.pack(pady=10, padx=9)  # <-- Espaciado lateral añadido


def activar_boton(boton):
    global boton_activo
    if boton_activo and boton_activo != boton:
        boton_activo.configure(fg_color=SENA_VERDE)
    boton.configure(fg_color="#007200")  # Verde más oscuro para indicar que está activo
    boton_activo = boton
    



# =========================
# Frame principal de contenido
# =========================
frame = ctk.CTkFrame(main_container, fg_color=FONDO_BLANCO)
frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

# =========================
# Título
# =========================
titulo = ctk.CTkLabel(
    frame, 
    text="Registro de Usuario", 
    font=ctk.CTkFont(family="Helvetica", size=22, weight="bold"),
    text_color=SENA_VERDE
)
titulo.pack(pady=10)

# =========================
# Frame para formulario
# =========================
form_frame = ctk.CTkFrame(frame, fg_color=FONDO_BLANCO)
form_frame.pack(fill="both", expand=True, padx=20, pady=10)
form_frame.columnconfigure(0, weight=1)
form_frame.columnconfigure(1, weight=1)

# =========================
# Función para crear campos con alineación
# =========================
def crear_campo_grid(frame, label_text, row):
    label = ctk.CTkLabel(
        frame, 
        text=label_text, 
        anchor="w", 
        text_color=TEXTO_OSCURO
    )
    label.grid(row=row, column=0, sticky="w", padx=20, pady=10)

    field = ctk.CTkEntry(
        frame, 
        width=300
    )
    field.grid(row=row, column=1, sticky="e", padx=20)

    return field

# =========================
# Campos del formulario
# =========================
entry_nombre = crear_campo_grid(form_frame, "Nombre:", 0)
entry_documento = crear_campo_grid(form_frame, "N° Documento:", 1)
entry_telefono = crear_campo_grid(form_frame, "Teléfono:", 2)
entry_correo = crear_campo_grid(form_frame, "Correo Electrónico:", 3)

# =========================
# Botones
# =========================
button_frame = ctk.CTkFrame(frame, fg_color="transparent")
button_frame.pack(pady=10)

registrar_btn = ctk.CTkButton(
    button_frame, 
    text="Registrar Usuario", 
    command=registrar,
    font=ctk.CTkFont(weight="bold"),
    height=40,
    width=200,
    fg_color=SENA_VERDE,
    hover_color="#00304D"
)
registrar_btn.pack(pady=8)

# =========================
# Pie de página
# =========================
footer = ctk.CTkLabel(
    frame, 
    text="Sistema de asistencia parqueadero - SENA", 
    font=ctk.CTkFont(size=10),
    text_color=TEXTO_OSCURO
)
footer.pack(pady=10)

# =========================
# Iniciar la app
# =========================
ventana.mainloop()

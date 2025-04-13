import sys
import os
import subprocess
# Agrega la ruta de la raíz del proyecto (un nivel arriba de View/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from controller.UsuarioController import consultar_usuarios
from controller.VehiculoController import registrar_Vehiculo
from model.vehiculo import vehiculo as class_vehiculo
import customtkinter as ctk
from tkinter import messagebox

# =========================
# Configuración del tema SENA (verde y blanco)
# =========================
ctk.set_appearance_mode("light")
SENA_VERDE = "#39A900"
TEXTO_OSCURO = "#333333"
FONDO_BLANCO = "#FFFFFF"
boton_activo = None

# =========================
# Obtener usuarios
# =========================
usuarios = consultar_usuarios()
usuarios_nombre = [usuario.nombre for usuario in usuarios]

def registrar_vehiculo():
    campos = {
        "Placa": entry_placa.get().strip(),
        "Marca": entry_marca.get().strip(),
        "Modelo": entry_modelo.get().strip(),
        "SOAT número": entry_soat.get().strip(),
        "SOAT vencimiento": entry_soat_fecha.get().strip(),
        "Tecno número": entry_tecno.get().strip(),
        "Tecno vencimiento": entry_tecno_fecha.get().strip(),
        "Usuario": combo_id_usuario.get().strip()
    }

    for nombre, valor in campos.items():
        if valor == "":
            messagebox.showerror("Campos vacíos", f"El campo '{nombre}' no puede estar vacío.")
            return

    id_Usuario = ""
    for usuario in usuarios:
        if usuario.nombre == combo_id_usuario.get():
            id_Usuario = usuario.idUsuario
            break

    nuevovehiculo = class_vehiculo(
        placa=entry_placa.get(),
        marca=entry_marca.get(),
        modelo=entry_modelo.get(),
        soat_numero=entry_soat.get(),
        soat_vencimiento=entry_soat_fecha.get(),
        tecno_numero=entry_tecno.get(),
        tecno_vencimiento=entry_tecno_fecha.get(),
        id_usuario=id_Usuario
    )

    if registrar_Vehiculo(nuevovehiculo):
        messagebox.showinfo("Registro exitoso", "Vehículo registrado correctamente.")
        entry_placa.delete(0, ctk.END)
        entry_marca.delete(0, ctk.END)
        entry_modelo.delete(0, ctk.END)
        entry_soat.delete(0, ctk.END)
        entry_soat_fecha.delete(0, ctk.END)
        entry_tecno.delete(0, ctk.END)
        entry_tecno_fecha.delete(0, ctk.END)
        combo_id_usuario.set("")
    else:
        messagebox.showerror("Error", "No se pudo registrar el vehículo.")

# =========================
# Ventana principal
# =========================
ventana = ctk.CTk()
ventana.title("Formulario Registro Vehículo")
ventana.geometry("900x600+500+100")  # Ancho aumentado para incluir el menú
ventana.configure(bg=FONDO_BLANCO)

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

# Funciones de navegación
def ir_a_page_principal():
    ventana.destroy()
    subprocess.Popen([sys.executable, os.path.abspath("Main.py")])

# Funciones de navegación
def ir_a_usuario():
    ventana.destroy()
    subprocess.Popen([sys.executable, os.path.abspath("RegistroUsuario.py")])

def ir_a_vehiculos():
    ventana.destroy()
    subprocess.Popen([sys.executable, os.path.abspath("RegistrarVehiculo.py")])



# Botones del menú
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
btn_usuario.pack(pady=10, padx=9)

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
btn_vehiculo.pack(pady=10, padx=9)

boton_activo = btn_vehiculo  # Marcar como activo por defecto

# =========================
# Frame principal (contenido)
# =========================
frame = ctk.CTkFrame(main_container, fg_color=FONDO_BLANCO)
frame.pack(side="left", fill="both", expand=True, padx=20, pady=20)

# =========================
# Título
# =========================
titulo = ctk.CTkLabel(
    frame,
    text="Registrar vehículo",
    font=ctk.CTkFont(family="Helvetica", size=20, weight="bold"),
    text_color=SENA_VERDE
)
titulo.pack(pady=10, padx=10)

# =========================
# Formulario
# =========================
form_frame = ctk.CTkFrame(frame, fg_color=FONDO_BLANCO)
form_frame.pack(fill="both", expand=True, padx=20, pady=20)

def crear_campo(frame, label_text, row, is_combobox=False):
    label = ctk.CTkLabel(frame, text=label_text, anchor="e", text_color=TEXTO_OSCURO)
    label.grid(row=row, column=0, sticky="e", padx=(20, 10), pady=10)

    if is_combobox:
        field = ctk.CTkComboBox(frame, values=usuarios_nombre, state="readonly", width=300)
    else:
        field = ctk.CTkEntry(frame, width=300)

    field.grid(row=row, column=1, sticky="w", padx=(10, 20), pady=10)
    return field

entry_placa = crear_campo(form_frame, "Placa:", 0)
entry_marca = crear_campo(form_frame, "Marca:", 1)
entry_modelo = crear_campo(form_frame, "Modelo (año):", 2)
entry_soat = crear_campo(form_frame, "SOAT número:", 3)
entry_soat_fecha = crear_campo(form_frame, "Fecha de vencimiento SOAT (YYYY-MM-DD):", 4)
entry_tecno = crear_campo(form_frame, "Tecno número:", 5)
entry_tecno_fecha = crear_campo(form_frame, "Fecha de vencimiento Tecno (YYYY-MM-DD):", 6)
combo_id_usuario = crear_campo(form_frame, "Usuario:", 7, is_combobox=True)

# =========================
# Botón registrar
# =========================
button_frame = ctk.CTkFrame(frame, fg_color="transparent")
button_frame.pack(fill="x", padx=20, pady=10)

registrar_btn = ctk.CTkButton(
    button_frame,
    text="Registrar Vehículo",
    command=registrar_vehiculo,
    font=ctk.CTkFont(family="Helvetica", weight="bold"),
    height=40,
    fg_color=SENA_VERDE,
    hover_color="#00304D"
)
registrar_btn.pack(pady=2)

# =========================
# Footer
# =========================
footer = ctk.CTkLabel(
    frame,
    text="Sistema de asistencia parqueadero - SENA",
    font=ctk.CTkFont(family="Helvetica", size=10),
    text_color=TEXTO_OSCURO
)
footer.pack()

ventana.mainloop()

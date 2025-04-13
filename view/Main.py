import customtkinter as ctk
import sys
import os
import subprocess
# Agrega la ruta de la raíz del proyecto (un nivel arriba de View/)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))




# Configuración inicial
ctk.set_appearance_mode("System")  # Puede ser "Dark", "Light" o "System"
ctk.set_default_color_theme("blue")  # Puedes cambiar el tema: "blue", "green", "dark-blue"

class ElegantApp(ctk.CTk):
    
        
    def __init__(self):
        super().__init__()
        self.title("Página principal")        
        self.geometry("900x600+500+100")  # Ancho aumentado para incluir el menú
        self.resizable(False, False)

        # Fondo
        self.configure(fg_color="#1e1e2f")  # Fondo oscuro elegante

        # Botón centrado
        self.button = ctk.CTkButton(
            master=self,
            text="Haz clic aquí",
            command=self.on_click,
            font=("Segoe UI", 18, "bold"),
            corner_radius=15,
            height=50,
            width=200,
        )
        self.button.place(relx=0.5, rely=0.5, anchor="center")

    def on_click(self):
        self.destroy()
        subprocess.Popen([sys.executable, os.path.abspath("RegistroUsuario.py")])

if __name__ == "__main__":
    app = ElegantApp()
    app.mainloop()

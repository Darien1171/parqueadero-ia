#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diálogo para registrar entrada de vehículos
"""
import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2

from modelos.vehiculo import Vehiculo
from modelos.usuario import Usuario

logger = logging.getLogger('parqueadero.ui.entrada_dialog')

class EntradaDialog:
    """
    Diálogo modal para registrar entrada de vehículos
    """
    
    def __init__(self, parent, placa_detectada="", imagen_path="", imagen_placa=None):
        """
        Inicializa el diálogo de entrada
        
        Args:
            parent: Ventana padre
            placa_detectada (str): Placa detectada automáticamente (puede ser vacía)
            imagen_path (str): Ruta a la imagen capturada
            imagen_placa (numpy.ndarray): Imagen recortada de la placa (puede ser None)
        """
        self.parent = parent
        self.placa_detectada = placa_detectada
        self.imagen_path = imagen_path
        self.imagen_placa = imagen_placa
        self.result = None
        
        # Crear diálogo
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Registro de Entrada")
        self.dialog.geometry("800x600")
        self.dialog.resizable(False, False)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        # Centrar en la pantalla
        self.dialog.update_idletasks()
        width = self.dialog.winfo_width()
        height = self.dialog.winfo_height()
        x = (self.dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (self.dialog.winfo_screenheight() // 2) - (height // 2)
        self.dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Crear interfaz
        self._create_widgets()
        
        # Buscar vehículo por placa
        self._buscar_vehiculo()
        
        # Esperar a que el diálogo se cierre
        self.dialog.wait_window()
    
    def _create_widgets(self):
        """Crear elementos de la interfaz"""
        # Marco principal
        main_frame = ttk.Frame(self.dialog, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Dividir en dos paneles: izquierdo para imagen, derecho para formulario
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))
        
        right_frame = ttk.Frame(main_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        # Panel de imagen
        self._create_image_panel(left_frame)
        
        # Panel de formulario
        self._create_form_panel(right_frame)
        
        # Botones de acción
        self._create_action_buttons(main_frame)
    
    def _create_image_panel(self, parent):
        """
        Crear panel para mostrar la imagen
        
        Args:
            parent: Widget padre
        """
        # Marco para imagen
        img_frame = ttk.LabelFrame(parent, text="Imagen Capturada")
        img_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Etiqueta para mostrar imagen
        self.img_label = ttk.Label(img_frame)
        self.img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Cargar y mostrar imagen
        if os.path.exists(self.imagen_path):
            self._load_and_display_image()
        else:
            self.img_label.config(text="No hay imagen disponible")
        
        # Si hay imagen de placa, mostrarla
        if self.imagen_placa is not None:
            # Marco para imagen de placa
            placa_frame = ttk.LabelFrame(parent, text="Placa Detectada")
            placa_frame.pack(fill=tk.X, padx=5, pady=5)
            
            # Etiqueta para mostrar imagen de placa
            self.placa_label = ttk.Label(placa_frame)
            self.placa_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Convertir imagen de placa para mostrar
            self._display_placa_image()
    
    def _load_and_display_image(self):
        """Cargar y mostrar imagen principal"""
        try:
            # Cargar imagen con PIL
            pil_img = Image.open(self.imagen_path)
            
            # Calcular tamaño para ajustar a la ventana (max 400x300)
            width, height = pil_img.size
            max_width = 380
            max_height = 300
            
            # Redimensionar manteniendo proporción
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convertir a formato Tkinter
            tk_img = ImageTk.PhotoImage(pil_img)
            
            # Mostrar en la etiqueta
            self.img_label.config(image=tk_img)
            self.img_label.image = tk_img  # Mantener referencia
            
        except Exception as e:
            logger.error(f"Error al cargar imagen: {e}")
            self.img_label.config(text=f"Error al cargar imagen: {e}")
    
    def _display_placa_image(self):
        """Mostrar imagen de la placa"""
        try:
            # Convertir BGR a RGB
            rgb_img = cv2.cvtColor(self.imagen_placa, cv2.COLOR_BGR2RGB)
            
            # Convertir a formato PIL
            pil_img = Image.fromarray(rgb_img)
            
            # Redimensionar si es necesario
            width, height = pil_img.size
            max_width = 200
            max_height = 80
            
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convertir a formato Tkinter
            tk_img = ImageTk.PhotoImage(pil_img)
            
            # Mostrar en la etiqueta
            self.placa_label.config(image=tk_img)
            self.placa_label.image = tk_img  # Mantener referencia
            
        except Exception as e:
            logger.error(f"Error al mostrar imagen de placa: {e}")
            self.placa_label.config(text=f"Error al mostrar placa: {e}")
    
    def _create_form_panel(self, parent):
        """
        Crear panel con formulario de datos
        
        Args:
            parent: Widget padre
        """
        # Marco para datos del vehículo
        vehicle_frame = ttk.LabelFrame(parent, text="Datos del Vehículo")
        vehicle_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Formulario de datos del vehículo
        # Placa
        ttk.Label(vehicle_frame, text="Placa:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.placa_var = tk.StringVar(value=self.placa_detectada)
        self.placa_entry = ttk.Entry(vehicle_frame, textvariable=self.placa_var)
        self.placa_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        self.placa_entry.bind("<KeyRelease>", self._on_placa_changed)
        
        # Botón de buscar
        ttk.Button(vehicle_frame, text="Buscar", command=self._buscar_vehiculo).grid(
            row=0, column=2, padx=5, pady=2
        )
        
        # Tipo de vehículo
        ttk.Label(vehicle_frame, text="Tipo:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.tipo_var = tk.StringVar()
        tipos = ["carro", "moto", "camión", "bus", "camioneta", "van"]
        self.tipo_combo = ttk.Combobox(vehicle_frame, textvariable=self.tipo_var, values=tipos)
        self.tipo_combo.grid(row=1, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        self.tipo_combo.current(0)
        
        # Marca
        ttk.Label(vehicle_frame, text="Marca:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.marca_var = tk.StringVar()
        marcas = ["Chevrolet", "Renault", "Mazda", "Nissan", "Toyota", "Kia", 
                "Hyundai", "Ford", "Volkswagen", "Mercedes-Benz", "BMW", "Honda",
                "Suzuki", "Yamaha", "Bajaj", "AKT", "Otro"]
        self.marca_combo = ttk.Combobox(vehicle_frame, textvariable=self.marca_var, values=marcas)
        self.marca_combo.grid(row=2, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        
        # Modelo
        ttk.Label(vehicle_frame, text="Modelo:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.modelo_var = tk.StringVar()
        self.modelo_entry = ttk.Entry(vehicle_frame, textvariable=self.modelo_var)
        self.modelo_entry.grid(row=3, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        
        # Color
        ttk.Label(vehicle_frame, text="Color:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.color_var = tk.StringVar()
        colores = ["blanco", "negro", "gris", "rojo", "azul", "verde", 
                "amarillo", "naranja", "marrón", "plata", "otro"]
        self.color_combo = ttk.Combobox(vehicle_frame, textvariable=self.color_var, values=colores)
        self.color_combo.grid(row=4, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=2)
        
        # Marco para datos del propietario
        owner_frame = ttk.LabelFrame(parent, text="Datos del Propietario")
        owner_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Nombre
        ttk.Label(owner_frame, text="Nombre:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.nombre_var = tk.StringVar()
        self.nombre_entry = ttk.Entry(owner_frame, textvariable=self.nombre_var)
        self.nombre_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Documento
        ttk.Label(owner_frame, text="Documento:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.documento_var = tk.StringVar()
        self.documento_entry = ttk.Entry(owner_frame, textvariable=self.documento_var)
        self.documento_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Teléfono
        ttk.Label(owner_frame, text="Teléfono:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.telefono_var = tk.StringVar()
        self.telefono_entry = ttk.Entry(owner_frame, textvariable=self.telefono_var)
        self.telefono_entry.grid(row=2, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Marco para observaciones
        obs_frame = ttk.LabelFrame(parent, text="Observaciones")
        obs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Campo de texto para observaciones
        self.observaciones_text = tk.Text(obs_frame, height=3, width=40)
        self.observaciones_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configurar redimensionamiento
        vehicle_frame.columnconfigure(1, weight=1)
        owner_frame.columnconfigure(1, weight=1)
    
    def _create_action_buttons(self, parent):
        """
        Crear botones de acción
        
        Args:
            parent: Widget padre
        """
        # Marco para botones
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, padx=5, pady=10)
        
        # Botón Cancelar
        ttk.Button(button_frame, text="Cancelar", command=self._on_cancel).pack(
            side=tk.RIGHT, padx=5
        )
        
        # Botón Registrar
        ttk.Button(button_frame, text="Registrar Entrada", command=self._on_submit).pack(
            side=tk.RIGHT, padx=5
        )
    
    def _on_placa_changed(self, event):
        """Manejador para cambios en campo de placa"""
        # Normalizar formato de placa
        placa = self.placa_var.get()
        if placa:
            placa_norm = Vehiculo.normalizar_placa(placa)
            if placa_norm != placa:
                self.placa_var.set(placa_norm)
    
    def _buscar_vehiculo(self):
        """Buscar vehículo por placa en la base de datos"""
        placa = self.placa_var.get()
        if not placa:
            return
        
        try:
            # Buscar vehículo
            vehiculo = Vehiculo().buscar_por_placa(placa)
            
            if vehiculo:
                # Llenar formulario con datos del vehículo
                self.tipo_var.set(vehiculo.tipo)
                self.marca_var.set(vehiculo.marca)
                self.modelo_var.set(vehiculo.modelo)
                self.color_var.set(vehiculo.color)
                
                # Si tiene propietario, buscar sus datos
                if vehiculo.id_propietario:
                    propietario = Usuario(vehiculo.id_propietario)
                    self.nombre_var.set(propietario.nombre)
                    self.documento_var.set(propietario.documento)
                    self.telefono_var.set(propietario.telefono)
                
                # Informar al usuario
                messagebox.showinfo(
                    "Vehículo Encontrado", 
                    f"Se encontró el vehículo con placa {placa} en la base de datos."
                )
            
        except Exception as e:
            logger.error(f"Error al buscar vehículo: {e}")
            messagebox.showerror("Error", f"Error al buscar vehículo: {e}")
    
    def _on_submit(self):
        """Manejador para botón Registrar"""
        # Validar datos mínimos
        placa = self.placa_var.get()
        
        if not placa:
            messagebox.showerror("Error", "Debe ingresar la placa del vehículo")
            return
        
        # Recopilar datos del formulario
        data = {
            'placa': placa,
            'tipo': self.tipo_var.get(),
            'marca': self.marca_var.get(),
            'modelo': self.modelo_var.get(),
            'color': self.color_var.get(),
            'propietario': self.nombre_var.get(),
            'documento': self.documento_var.get(),
            'telefono': self.telefono_var.get(),
            'observaciones': self.observaciones_text.get("1.0", tk.END).strip()
        }
        
        # Guardar resultado
        self.result = data
        
        # Cerrar diálogo
        self.dialog.destroy()
    
    def _on_cancel(self):
        """Manejador para botón Cancelar"""
        self.result = None
        self.dialog.destroy()


# Para pruebas independientes
if __name__ == "__main__":
    import sys
    import os
    
    # Añadir directorio raíz al path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear ventana de prueba
    root = tk.Tk()
    root.title("Prueba de Diálogo de Entrada")
    root.geometry("300x200")
    
    # Función para abrir diálogo
    def open_dialog():
        # Buscar una imagen existente para la prueba
        img_path = ""
        for test_path in [
            "imagenes/entrada/test.jpg",
            "test.jpg",
            os.path.join(os.path.dirname(__file__), "test.jpg")
        ]:
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        dialog = EntradaDialog(root, "ABC123", img_path)
        
        if dialog.result:
            print("Datos ingresados:")
            for key, value in dialog.result.items():
                print(f"  {key}: {value}")
        else:
            print("Diálogo cancelado")
    
    # Botón para abrir diálogo
    ttk.Button(root, text="Abrir Diálogo de Entrada", command=open_dialog).pack(
        padx=20, pady=20
    )
    
    # Iniciar bucle principal
    root.mainloop()

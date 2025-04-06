#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diálogo para registrar salida de vehículos
"""
import os
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
from datetime import datetime

from modelos.vehiculo import Vehiculo
from modelos.usuario import Usuario
from modelos.estado import Estado

logger = logging.getLogger('parqueadero.ui.salida_dialog')

class SalidaDialog:
    """
    Diálogo modal para registrar salida de vehículos
    """
    
    def __init__(self, parent, placa_detectada="", registro_entrada=None, imagen_path="", imagen_placa=None):
        """
        Inicializa el diálogo de salida
        
        Args:
            parent: Ventana padre
            placa_detectada (str): Placa detectada automáticamente (puede ser vacía)
            registro_entrada (Estado): Datos de la entrada del vehículo (puede ser None)
            imagen_path (str): Ruta a la imagen capturada
            imagen_placa (numpy.ndarray): Imagen recortada de la placa (puede ser None)
        """
        self.parent = parent
        self.placa_detectada = placa_detectada
        self.registro_entrada = registro_entrada
        self.imagen_path = imagen_path
        self.imagen_placa = imagen_placa
        self.result = None
        
        # Crear diálogo
        self.dialog = tk.Toplevel(parent)
        self.dialog.title("Registro de Salida")
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
        
        # Cargar datos del vehículo
        self._cargar_datos_vehiculo()
        
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
        
        # Panel de información
        self._create_info_panel(right_frame)
        
        # Botones de acción
        self._create_action_buttons(main_frame)
    
    def _create_image_panel(self, parent):
        """
        Crear panel para mostrar las imágenes
        
        Args:
            parent: Widget padre
        """
        # Panel superior: Imagen actual
        img_frame = ttk.LabelFrame(parent, text="Imagen Actual")
        img_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Etiqueta para mostrar imagen
        self.img_label = ttk.Label(img_frame)
        self.img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Cargar y mostrar imagen
        if os.path.exists(self.imagen_path):
            self._load_and_display_image(self.imagen_path, self.img_label)
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
        
        # Panel inferior: Imagen de entrada
        if self.registro_entrada and self.registro_entrada.imagen_entrada:
            entrada_frame = ttk.LabelFrame(parent, text="Imagen de Entrada")
            entrada_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Etiqueta para mostrar imagen de entrada
            self.entrada_label = ttk.Label(entrada_frame)
            self.entrada_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
            
            # Cargar y mostrar imagen de entrada
            self._load_and_display_image(self.registro_entrada.imagen_entrada, self.entrada_label)
    
    def _load_and_display_image(self, image_path, label):
        """
        Cargar y mostrar imagen en etiqueta
        
        Args:
            image_path (str): Ruta a la imagen
            label (ttk.Label): Etiqueta donde mostrar la imagen
        """
        try:
            # Verificar si la imagen existe
            if not os.path.exists(image_path):
                label.config(text=f"Imagen no encontrada: {image_path}")
                return
            
            # Cargar imagen con PIL
            pil_img = Image.open(image_path)
            
            # Calcular tamaño para ajustar a la ventana (max 380x250)
            width, height = pil_img.size
            max_width = 380
            max_height = 250
            
            # Redimensionar manteniendo proporción
            if width > max_width or height > max_height:
                ratio = min(max_width / width, max_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                pil_img = pil_img.resize((new_width, new_height), Image.LANCZOS)
            
            # Convertir a formato Tkinter
            tk_img = ImageTk.PhotoImage(pil_img)
            
            # Mostrar en la etiqueta
            label.config(image=tk_img)
            label.image = tk_img  # Mantener referencia
            
        except Exception as e:
            logger.error(f"Error al cargar imagen {image_path}: {e}")
            label.config(text=f"Error al cargar imagen: {e}")
    
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
    
    def _create_info_panel(self, parent):
        """
        Crear panel con información del vehículo
        
        Args:
            parent: Widget padre
        """
        # Marco para datos del vehículo
        self.vehicle_frame = ttk.LabelFrame(parent, text="Datos del Vehículo")
        self.vehicle_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Placa
        ttk.Label(self.vehicle_frame, text="Placa:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.placa_var = tk.StringVar(value=self.placa_detectada)
        self.placa_entry = ttk.Entry(self.vehicle_frame, textvariable=self.placa_var)
        self.placa_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=2)
        
        # Botón de buscar
        ttk.Button(self.vehicle_frame, text="Buscar", command=self._buscar_vehiculo).grid(
            row=0, column=2, padx=5, pady=2
        )
        
        # Tipo de vehículo
        ttk.Label(self.vehicle_frame, text="Tipo:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.tipo_var = tk.StringVar()
        self.tipo_label = ttk.Label(self.vehicle_frame, textvariable=self.tipo_var)
        self.tipo_label.grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Marca
        ttk.Label(self.vehicle_frame, text="Marca:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.marca_var = tk.StringVar()
        self.marca_label = ttk.Label(self.vehicle_frame, textvariable=self.marca_var)
        self.marca_label.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Color
        ttk.Label(self.vehicle_frame, text="Color:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.color_var = tk.StringVar()
        self.color_label = ttk.Label(self.vehicle_frame, textvariable=self.color_var)
        self.color_label.grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=5, pady=2)
        
        # Marco para datos de propietario
        self.propietario_frame = ttk.LabelFrame(parent, text="Datos del Propietario")
        self.propietario_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Nombre
        ttk.Label(self.propietario_frame, text="Nombre:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.nombre_var = tk.StringVar()
        self.nombre_label = ttk.Label(self.propietario_frame, textvariable=self.nombre_var)
        self.nombre_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Documento
        ttk.Label(self.propietario_frame, text="Documento:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.documento_var = tk.StringVar()
        self.documento_label = ttk.Label(self.propietario_frame, textvariable=self.documento_var)
        self.documento_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Marco para datos de estancia
        self.estancia_frame = ttk.LabelFrame(parent, text="Datos de Estancia")
        self.estancia_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Fecha de entrada
        ttk.Label(self.estancia_frame, text="Entrada:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.entrada_var = tk.StringVar()
        self.entrada_label = ttk.Label(self.estancia_frame, textvariable=self.entrada_var)
        self.entrada_label.grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Tiempo de estancia
        ttk.Label(self.estancia_frame, text="Tiempo:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.tiempo_var = tk.StringVar()
        self.tiempo_label = ttk.Label(self.estancia_frame, textvariable=self.tiempo_var, font=("Arial", 12, "bold"))
        self.tiempo_label.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Marco para observaciones
        obs_frame = ttk.LabelFrame(parent, text="Observaciones")
        obs_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Campo de texto para observaciones
        self.observaciones_text = tk.Text(obs_frame, height=3, width=40)
        self.observaciones_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configurar redimensionamiento
        self.vehicle_frame.columnconfigure(1, weight=1)
        self.propietario_frame.columnconfigure(1, weight=1)
        self.estancia_frame.columnconfigure(1, weight=1)
    
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
        
        # Botón Registrar Salida
        self.btn_registrar = ttk.Button(button_frame, text="Registrar Salida", command=self._on_submit)
        self.btn_registrar.pack(side=tk.RIGHT, padx=5)
        
        # Deshabilitar botón si no hay registro de entrada
        if not self.registro_entrada:
            self.btn_registrar.config(state=tk.DISABLED)
    
    def _cargar_datos_vehiculo(self):
        """Cargar datos del vehículo a partir del registro de entrada"""
        if not self.registro_entrada:
            # Si no hay registro, pero hay placa detectada, buscar
            if self.placa_detectada:
                self._buscar_vehiculo()
            return
        
        try:
            # Obtener datos del vehículo
            placa = self.registro_entrada.placa
            self.placa_var.set(placa)
            
            vehiculo = Vehiculo().buscar_por_placa(placa)
            if vehiculo:
                # Mostrar datos del vehículo
                self.tipo_var.set(vehiculo.tipo)
                self.marca_var.set(vehiculo.marca)
                self.color_var.set(vehiculo.color)
                
                # Mostrar datos del propietario
                if vehiculo.id_propietario:
                    propietario = Usuario(vehiculo.id_propietario)
                    self.nombre_var.set(propietario.nombre)
                    self.documento_var.set(propietario.documento)
            
            # Mostrar datos de estancia
            if self.registro_entrada.fecha_entrada:
                entrada_str = self.registro_entrada.fecha_entrada.strftime("%d/%m/%Y %H:%M:%S")
                self.entrada_var.set(entrada_str)
                
                # Calcular tiempo de estancia
                tiempo_estancia = datetime.now() - self.registro_entrada.fecha_entrada
                
                # Formatear en horas y minutos
                horas = int(tiempo_estancia.total_seconds() // 3600)
                minutos = int((tiempo_estancia.total_seconds() % 3600) // 60)
                self.tiempo_var.set(f"{horas}h {minutos}m")
            
            # Habilitar botón de registro
            self.btn_registrar.config(state=tk.NORMAL)
            
        except Exception as e:
            logger.error(f"Error al cargar datos del vehículo: {e}")
            messagebox.showerror("Error", f"Error al cargar datos del vehículo: {e}")
    
    def _buscar_vehiculo(self):
        """Buscar entrada activa para placa ingresada"""
        placa = self.placa_var.get()
        if not placa:
            messagebox.showerror("Error", "Debe ingresar una placa")
            return
        
        try:
            # Normalizar placa
            placa = Vehiculo.normalizar_placa(placa)
            self.placa_var.set(placa)
            
            # Buscar entrada activa
            estado = Estado()
            self.registro_entrada = estado.buscar_entrada_activa(placa)
            
            if self.registro_entrada:
                # Cargar datos del vehículo
                self._cargar_datos_vehiculo()
                
                messagebox.showinfo(
                    "Entrada Encontrada", 
                    f"Se encontró registro de entrada activo para la placa {placa}."
                )
            else:
                # Limpiar campos
                self.tipo_var.set("")
                self.marca_var.set("")
                self.color_var.set("")
                self.nombre_var.set("")
                self.documento_var.set("")
                self.entrada_var.set("")
                self.tiempo_var.set("")
                
                # Deshabilitar botón de registro
                self.btn_registrar.config(state=tk.DISABLED)
                
                messagebox.showwarning(
                    "Entrada No Encontrada", 
                    f"No se encontró registro de entrada activo para la placa {placa}."
                )
            
        except Exception as e:
            logger.error(f"Error al buscar entrada: {e}")
            messagebox.showerror("Error", f"Error al buscar entrada: {e}")
    
    def _on_submit(self):
        """Manejador para botón Registrar Salida"""
        if not self.registro_entrada:
            messagebox.showerror("Error", "No hay registro de entrada para este vehículo")
            return
        
        # Recopilar datos del formulario
        data = {
            'placa': self.placa_var.get(),
            'id_registro': self.registro_entrada.id,
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
    root.title("Prueba de Diálogo de Salida")
    root.geometry("300x200")
    
    # Función para abrir diálogo
    def open_dialog():
        # Buscar una imagen existente para la prueba
        img_path = ""
        for test_path in [
            "imagenes/salida/test.jpg",
            "test.jpg",
            os.path.join(os.path.dirname(__file__), "test.jpg")
        ]:
            if os.path.exists(test_path):
                img_path = test_path
                break
        
        # Para pruebas, crear un registro de entrada ficticio
        class MockRegistro:
            def __init__(self):
                self.id = 1
                self.placa = "ABC123"
                self.fecha_entrada = datetime.now()
                self.imagen_entrada = img_path
        
        # Abrir diálogo
        dialog = SalidaDialog(root, "ABC123", MockRegistro(), img_path)
        
        if dialog.result:
            print("Datos ingresados:")
            for key, value in dialog.result.items():
                print(f"  {key}: {value}")
        else:
            print("Diálogo cancelado")
    
    # Botón para abrir diálogo
    ttk.Button(root, text="Abrir Diálogo de Salida", command=open_dialog).pack(
        padx=20, pady=20
    )
    
    # Iniciar bucle principal
    root.mainloop()

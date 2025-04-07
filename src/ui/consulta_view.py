#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vista para consulta de vehículos en el parqueadero
"""
import logging
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import os
from PIL import Image, ImageTk

from modelos.estado import Estado
from modelos.vehiculo import Vehiculo

logger = logging.getLogger('parqueadero.ui.consulta_view')

class ConsultaView:
    """
    Ventana para consultar vehículos en el parqueadero
    """
    
    def __init__(self, parent):
        """
        Inicializa la ventana de consulta
        
        Args:
            parent: Ventana padre
        """
        self.parent = parent
        
        # Crear ventana
        self.window = tk.Toplevel(parent)
        self.window.title("Consulta de Vehículos en Parqueadero")
        self.window.geometry("900x600")
        self.window.transient(parent)
        
        # Centrar en la pantalla
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")
        
        # Crear interfaz
        self._create_widgets()
        
        # Cargar datos
        self._cargar_vehiculos()
    
    def _create_widgets(self):
        """Crear elementos de la interfaz"""
        # Marco principal
        main_frame = ttk.Frame(self.window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel superior para filtros
        filter_frame = ttk.LabelFrame(main_frame, text="Filtros")
        filter_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Entrada para buscar por placa
        ttk.Label(filter_frame, text="Buscar por placa:").grid(row=0, column=0, padx=5, pady=5)
        self.placa_var = tk.StringVar()
        self.placa_entry = ttk.Entry(filter_frame, textvariable=self.placa_var)
        self.placa_entry.grid(row=0, column=1, padx=5, pady=5)
        self.placa_entry.bind("<Return>", lambda e: self._filtrar_por_placa())
        
        # Botón para buscar
        ttk.Button(filter_frame, text="Buscar", command=self._filtrar_por_placa).grid(
            row=0, column=2, padx=5, pady=5
        )
        
        # Botón para mostrar todos
        ttk.Button(filter_frame, text="Mostrar Todos", command=self._cargar_vehiculos).grid(
            row=0, column=3, padx=5, pady=5
        )
        
        # Etiqueta de contador
        self.contador_var = tk.StringVar(value="Vehículos en parqueadero: 0")
        ttk.Label(filter_frame, textvariable=self.contador_var, font=("Arial", 10, "bold")).grid(
            row=0, column=4, padx=15, pady=5
        )
        
        # Panel principal divido en dos secciones
        paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Panel izquierdo: Tabla de vehículos
        self.table_frame = ttk.Frame(paned)
        paned.add(self.table_frame, weight=3)
        
        # Panel derecho: Detalles e imagen
        self.details_frame = ttk.Frame(paned)
        paned.add(self.details_frame, weight=2)
        
        # Crear tabla de vehículos
        self._create_vehicle_table()
        
        # Crear panel de detalles
        self._create_details_panel()
        
        # Panel inferior para botones de acción
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botón para cerrar
        ttk.Button(button_frame, text="Cerrar", command=self.window.destroy).pack(
            side=tk.RIGHT, padx=5, pady=5
        )
        
        # Botón para refrescar
        ttk.Button(button_frame, text="Refrescar", command=self._cargar_vehiculos).pack(
            side=tk.RIGHT, padx=5, pady=5
        )
    
    def _create_vehicle_table(self):
        """Crear tabla para mostrar vehículos"""
        # Marco para tabla con scrollbar
        table_container = ttk.Frame(self.table_frame)
        table_container.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar vertical
        scrollbar_y = ttk.Scrollbar(table_container)
        scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Scrollbar horizontal
        scrollbar_x = ttk.Scrollbar(table_container, orient=tk.HORIZONTAL)
        scrollbar_x.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Crear treeview (tabla)
        columns = ("placa", "tipo", "propietario", "entrada", "tiempo")
        self.table = ttk.Treeview(
            table_container, 
            columns=columns,
            show="headings",
            yscrollcommand=scrollbar_y.set,
            xscrollcommand=scrollbar_x.set
        )
        
        # Configurar scrollbars
        scrollbar_y.config(command=self.table.yview)
        scrollbar_x.config(command=self.table.xview)
        
        # Configurar columnas
        self.table.heading("placa", text="Placa")
        self.table.heading("tipo", text="Tipo")
        self.table.heading("propietario", text="Propietario")
        self.table.heading("entrada", text="Entrada")
        self.table.heading("tiempo", text="Tiempo")
        
        self.table.column("placa", width=100, anchor=tk.CENTER)
        self.table.column("tipo", width=100, anchor=tk.W)
        self.table.column("propietario", width=150, anchor=tk.W)
        self.table.column("entrada", width=150, anchor=tk.CENTER)
        self.table.column("tiempo", width=100, anchor=tk.CENTER)
        
        # Empaquetar tabla
        self.table.pack(fill=tk.BOTH, expand=True)
        
        # Vincular evento de selección
        self.table.bind("<<TreeviewSelect>>", self._on_select_vehicle)
    
    def _create_details_panel(self):
        """Crear panel para mostrar detalles del vehículo seleccionado"""
        # Marco para información
        info_frame = ttk.LabelFrame(self.details_frame, text="Información del Vehículo")
        info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Etiquetas para datos
        # Placa
        ttk.Label(info_frame, text="Placa:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.det_placa_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.det_placa_var, font=("Arial", 10, "bold")).grid(
            row=0, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        # Tipo
        ttk.Label(info_frame, text="Tipo:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.det_tipo_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.det_tipo_var).grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        # Marca
        ttk.Label(info_frame, text="Marca:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.det_marca_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.det_marca_var).grid(
            row=2, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        # Color
        ttk.Label(info_frame, text="Color:").grid(row=3, column=0, sticky=tk.W, padx=5, pady=2)
        self.det_color_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.det_color_var).grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        # Propietario
        ttk.Label(info_frame, text="Propietario:").grid(row=4, column=0, sticky=tk.W, padx=5, pady=2)
        self.det_propietario_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.det_propietario_var).grid(
            row=4, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        # Teléfono
        ttk.Label(info_frame, text="Teléfono:").grid(row=5, column=0, sticky=tk.W, padx=5, pady=2)
        self.det_telefono_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.det_telefono_var).grid(
            row=5, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        # Entrada
        ttk.Label(info_frame, text="Entrada:").grid(row=6, column=0, sticky=tk.W, padx=5, pady=2)
        self.det_entrada_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.det_entrada_var).grid(
            row=6, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        # Tiempo
        ttk.Label(info_frame, text="Tiempo:").grid(row=7, column=0, sticky=tk.W, padx=5, pady=2)
        self.det_tiempo_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.det_tiempo_var, font=("Arial", 12, "bold")).grid(
            row=7, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        # Observaciones
        ttk.Label(info_frame, text="Observaciones:").grid(row=8, column=0, sticky=tk.NW, padx=5, pady=2)
        self.det_observaciones_var = tk.StringVar()
        ttk.Label(info_frame, textvariable=self.det_observaciones_var, wraplength=250).grid(
            row=8, column=1, sticky=tk.W, padx=5, pady=2
        )
        
        # Configurar grid
        info_frame.columnconfigure(1, weight=1)
        
        # Marco para imagen
        img_frame = ttk.LabelFrame(self.details_frame, text="Imagen de Entrada")
        img_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Etiqueta para mostrar imagen
        self.img_label = ttk.Label(img_frame)
        self.img_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
    
    def _cargar_vehiculos(self):
        """Cargar lista de vehículos actualmente en el parqueadero"""
        try:
            # Limpiar tabla
            for item in self.table.get_children():
                self.table.delete(item)
            
            # Limpiar detalles
            self._limpiar_detalles()
            
            # Obtener vehículos en parqueadero
            estado = Estado()
            vehiculos = estado.obtener_vehiculos_en_parqueadero()
            
            # Mostrar contador
            self.contador_var.set(f"Vehículos en parqueadero: {len(vehiculos)}")
            
            # Llenar tabla
            for i, vehiculo in enumerate(vehiculos):
                # Formatear fecha de entrada
                entrada = vehiculo['fecha_entrada']
                if entrada:
                    entrada_str = entrada.strftime("%d/%m/%Y %H:%M")
                else:
                    entrada_str = "Desconocido"
                
                # Insertar en tabla
                self.table.insert(
                    "",
                    tk.END,
                    iid=str(i),
                    values=(
                        vehiculo['placa'],
                        vehiculo.get('tipo', ''),
                        vehiculo.get('propietario', ''),
                        entrada_str,
                        vehiculo.get('tiempo_estancia_str', '')
                    ),
                    tags=('vehiculo',)
                )
                
                # Almacenar datos adicionales
                self.table.item(str(i), tags=(str(vehiculo['id'])))
            
            # Alternar colores de filas
            for i, item in enumerate(self.table.get_children()):
                if i % 2 == 0:
                    self.table.item(item, tags=('even', self.table.item(item, 'tags')[0]))
                else:
                    self.table.item(item, tags=('odd', self.table.item(item, 'tags')[0]))
            
            # Configurar colores
            self.table.tag_configure('even', background='#f0f0f0')
            self.table.tag_configure('odd', background='#ffffff')
            
        except Exception as e:
            logger.error(f"Error al cargar vehículos: {e}")
            messagebox.showerror("Error", f"Error al cargar vehículos: {e}")
    
    def _filtrar_por_placa(self):
        """Filtrar vehículos por placa"""
        placa = self.placa_var.get()
        if not placa:
            self._cargar_vehiculos()
            return
        
        try:
            # Normalizar placa
            placa = Vehiculo.normalizar_placa(placa)
            
            # Limpiar tabla
            for item in self.table.get_children():
                self.table.delete(item)
            
            # Obtener vehículos en parqueadero
            estado = Estado()
            todos_vehiculos = estado.obtener_vehiculos_en_parqueadero()
            
            # Filtrar por placa
            vehiculos = [v for v in todos_vehiculos if placa.upper() in v['placa'].upper()]
            
            # Mostrar contador
            self.contador_var.set(f"Vehículos filtrados: {len(vehiculos)}")
            
            # Llenar tabla
            for i, vehiculo in enumerate(vehiculos):
                # Formatear fecha de entrada
                entrada = vehiculo['fecha_entrada']
                if entrada:
                    entrada_str = entrada.strftime("%d/%m/%Y %H:%M")
                else:
                    entrada_str = "Desconocido"
                
                # Insertar en tabla
                self.table.insert(
                    "",
                    tk.END,
                    iid=str(i),
                    values=(
                        vehiculo['placa'],
                        vehiculo.get('tipo', ''),
                        vehiculo.get('propietario', ''),
                        entrada_str,
                        vehiculo.get('tiempo_estancia_str', '')
                    )
                )
                
                # Almacenar datos adicionales
                self.table.item(str(i), tags=(str(vehiculo['id'])))
            
            # Alternar colores de filas
            for i, item in enumerate(self.table.get_children()):
                if i % 2 == 0:
                    self.table.item(item, tags=('even', self.table.item(item, 'tags')[0]))
                else:
                    self.table.item(item, tags=('odd', self.table.item(item, 'tags')[0]))
            
        except Exception as e:
            logger.error(f"Error al filtrar vehículos: {e}")
            messagebox.showerror("Error", f"Error al filtrar vehículos: {e}")
    
    def _on_select_vehicle(self, event):
        """Manejador para selección de vehículo en la tabla"""
        # Obtener ítem seleccionado
        selection = self.table.selection()
        if not selection:
            return
        
        try:
            # Obtener ID del registro
            item = self.table.item(selection[0])
            id_registro = int(item['tags'][0])
            
            # Cargar datos del registro
            registro = Estado(id_registro)
            if not registro.id:
                return
            
            # Mostrar datos básicos
            self.det_placa_var.set(registro.placa)
            
            # Mostrar fecha de entrada y tiempo
            if registro.fecha_entrada:
                entrada_str = registro.fecha_entrada.strftime("%d/%m/%Y %H:%M:%S")
                self.det_entrada_var.set(entrada_str)
                
                # Calcular tiempo de estancia
                tiempo_estancia = datetime.now() - registro.fecha_entrada
                
                # Formatear en horas y minutos
                horas = int(tiempo_estancia.total_seconds() // 3600)
                minutos = int((tiempo_estancia.total_seconds() % 3600) // 60)
                self.det_tiempo_var.set(f"{horas}h {minutos}m")
            else:
                self.det_entrada_var.set("Desconocido")
                self.det_tiempo_var.set("")
            
            # Mostrar observaciones
            self.det_observaciones_var.set(registro.observaciones_entrada)
            
            # Cargar datos del vehículo
            if registro.id_vehiculo:
                vehiculo = Vehiculo(registro.id_vehiculo)
                self.det_tipo_var.set(vehiculo.tipo)
                self.det_marca_var.set(vehiculo.marca)
                self.det_color_var.set(vehiculo.color)
                
                # Cargar datos del propietario
                if vehiculo.id_propietario:
                    propietario = vehiculo.obtener_propietario()
                    if propietario:
                        self.det_propietario_var.set(propietario['nombre'])
                        self.det_telefono_var.set(propietario['telefono'])
                else:
                    self.det_propietario_var.set("")
                    self.det_telefono_var.set("")
            else:
                self.det_tipo_var.set("")
                self.det_marca_var.set("")
                self.det_color_var.set("")
                self.det_propietario_var.set("")
                self.det_telefono_var.set("")
            
            # Mostrar imagen
            self._mostrar_imagen(registro.imagen_entrada)
            
        except Exception as e:
            logger.error(f"Error al cargar detalles: {e}")
            self._limpiar_detalles()
            self.det_placa_var.set(f"Error: {e}")
    
    def _mostrar_imagen(self, image_path):
        """
        Mostrar imagen en panel de detalles
        
        Args:
            image_path (str): Ruta a la imagen
        """
        try:
            if not image_path or not os.path.exists(image_path):
                self.img_label.config(text="No hay imagen disponible")
                return
            
            # Cargar imagen con PIL
            pil_img = Image.open(image_path)
            
            # Calcular tamaño para ajustar a la ventana (max 300x200)
            width, height = pil_img.size
            max_width = 300
            max_height = 200
            
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
            logger.error(f"Error al mostrar imagen {image_path}: {e}")
            self.img_label.config(text=f"Error al cargar imagen: {e}")
    
    def _limpiar_detalles(self):
        """Limpiar panel de detalles"""
        self.det_placa_var.set("")
        self.det_tipo_var.set("")
        self.det_marca_var.set("")
        self.det_color_var.set("")
        self.det_propietario_var.set("")
        self.det_telefono_var.set("")
        self.det_entrada_var.set("")
        self.det_tiempo_var.set("")
        self.det_observaciones_var.set("")
        self.img_label.config(image="", text="Seleccione un vehículo para ver su imagen")


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
    root.title("Prueba de Vista de Consulta")
    root.geometry("300x200")
    
    # Función para abrir vista
    def open_view():
        view = ConsultaView(root)
    
    # Botón para abrir vista
    ttk.Button(root, text="Abrir Vista de Consulta", command=open_view).pack(
        padx=20, pady=20
    )
    
    # Iniciar bucle principal
    root.mainloop()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Parqueadero con OpenALPR - Usando comando 'alpr' con configuración US
"""

import os
import sys
import time
import logging
import cv2
import numpy as np
import subprocess
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageTk

# Crear directorios necesarios
os.makedirs('logs', exist_ok=True)
os.makedirs('imagenes/entrada', exist_ok=True)
os.makedirs('imagenes/salida', exist_ok=True)
os.makedirs('imagenes/procesado', exist_ok=True)
os.makedirs('resultados', exist_ok=True)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'parqueadero_openalpr_{time.strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('parqueadero_openalpr')

# Detector de placas con OpenALPR (usando el comando)
class OpenALPRDetectorCmd:
    """Detector de placas usando el comando 'alpr' de OpenALPR"""
    
    def __init__(self, country="us", config_file=None, runtime_dir=None):
        """
        Inicializa el detector de placas usando el comando alpr de OpenALPR
        
        Args:
            country: Código de país para OpenALPR
            config_file: Ruta al archivo de configuración de OpenALPR
            runtime_dir: Ruta al directorio runtime_data de OpenALPR
        """
        try:
            # Verificar si el comando alpr está disponible
            result = subprocess.run(['which', 'alpr'], capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error("Comando 'alpr' no encontrado. Asegúrese de que OpenALPR está instalado.")
                self.loaded = False
                return
                
            # Configurar parámetros para el comando
            self.country = country
            self.config_file = config_file
            self.runtime_dir = runtime_dir
            
            # Parámetros predeterminados
            self.pattern = "aaa999"  # Patrón para placas formato AAA123
            self.topn = 5
            self.detect_region = False
            
            self.loaded = True
            logger.info(f"Detector OpenALPR (comando) inicializado correctamente con país '{country}'")
        except Exception as e:
            logger.error(f"Error al inicializar detector OpenALPR: {e}")
            self.loaded = False
    
    def set_pattern(self, pattern):
        """Establecer patrón de placa"""
        self.pattern = pattern
    
    def set_top_n(self, topn):
        """Establecer número de resultados"""
        self.topn = topn
    
    def set_detect_region(self, detect):
        """Establecer detección de región"""
        self.detect_region = detect
    
    def detectar_placa(self, imagen):
        """
        Detecta placa en una imagen usando el comando alpr
        
        Args:
            imagen: Imagen en formato numpy/OpenCV
            
        Returns:
            tuple: (texto_placa, confianza, imagen_placa)
        """
        if not self.loaded:
            logger.warning("Detector OpenALPR no inicializado correctamente")
            return None, 0.0, None
        
        try:
            # Guardar imagen temporalmente
            temp_img_path = "temp_alpr_input.jpg"
            cv2.imwrite(temp_img_path, imagen)
            
            # Construir comando
            cmd = ['alpr', '-c', self.country, '-j']
            
            # Agregar opciones adicionales si están configuradas
            if self.pattern:
                cmd.extend(['-p', self.pattern])
            
            if self.topn:
                cmd.extend(['-n', str(self.topn)])
            
            if self.config_file:
                cmd.extend(['--config', self.config_file])
            
            if self.runtime_dir:
                cmd.extend(['--runtime_dir', self.runtime_dir])
            
            # Añadir la ruta de la imagen
            cmd.append(temp_img_path)
            
            logger.info(f"Ejecutando comando: {' '.join(cmd)}")
            
            # Ejecutar comando
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Eliminar archivo temporal
            try:
                os.remove(temp_img_path)
            except:
                pass
            
            # Verificar si el comando se ejecutó correctamente
            if result.returncode != 0:
                logger.error(f"Error al ejecutar alpr: {result.stderr}")
                return None, 0.0, None
            
            # Parsear resultado (formato JSON)
            try:
                output = json.loads(result.stdout)
                
                # Verificar si se detectaron placas
                if output['results'] and len(output['results']) > 0:
                    # Obtener la mejor coincidencia
                    plate = output['results'][0]
                    placa_texto = plate['plate']
                    confianza = plate['confidence']
                    
                    # Extraer coordenadas de la placa
                    coords = plate['coordinates']
                    x_min = min(c['x'] for c in coords)
                    y_min = min(c['y'] for c in coords)
                    x_max = max(c['x'] for c in coords)
                    y_max = max(c['y'] for c in coords)
                    
                    # Asegurar que las coordenadas están dentro de la imagen
                    height, width = imagen.shape[:2]
                    x_min = max(0, x_min)
                    y_min = max(0, y_min)
                    x_max = min(width, x_max)
                    y_max = min(height, y_max)
                    
                    # Extraer la región de la placa
                    img_placa = imagen[y_min:y_max, x_min:x_max].copy() if y_max > y_min and x_max > x_min else None
                    
                    # Crear una versión con borde rojo
                    img_with_plate = imagen.copy()
                    cv2.rectangle(img_with_plate, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                    
                    # Ampliar un poco la región para mejor visualización
                    margin = 10
                    x_min_padded = max(0, x_min - margin)
                    y_min_padded = max(0, y_min - margin)
                    x_max_padded = min(width, x_max + margin)
                    y_max_padded = min(height, y_max + margin)
                    
                    # Extraer la región ampliada con el recuadro
                    img_placa_padded = img_with_plate[y_min_padded:y_max_padded, x_min_padded:x_max_padded].copy() if y_max_padded > y_min_padded and x_max_padded > x_min_padded else None
                    
                    logger.info(f"Placa detectada: {placa_texto} con confianza {confianza:.2f}")
                    return placa_texto, confianza, img_placa_padded if img_placa_padded is not None else img_placa
                else:
                    logger.info("No se detectaron placas en la imagen")
                    return None, 0.0, None
            except json.JSONDecodeError:
                logger.error(f"Error al parsear salida JSON de alpr: {result.stdout}")
                return None, 0.0, None
                
        except Exception as e:
            logger.error(f"Error al procesar imagen con alpr: {e}")
            return None, 0.0, None


class AnalisisImagenApp:
    """Aplicación para análisis de imágenes con OpenALPR"""
    
    def __init__(self, root):
        """
        Inicializa la aplicación de análisis de imágenes
        
        Args:
            root: Ventana principal de Tkinter
        """
        self.root = root
        self.root.title("Sistema de Parqueadero - Análisis de Imagen con OpenALPR")
        self.root.geometry("1280x720")
        
        # Inicializar variables
        self.image_path = None
        self.current_image = None
        self.detection_results = None
        
        # Configurar interfaz
        self._setup_ui()
        
        # Inicializar OpenALPR después de la UI para poder usar la consola
        self._init_openalpr()
        
        # Mensaje de bienvenida
        self._log_event("Sistema iniciado. Utiliza OpenALPR con configuración US. Por favor cargue una imagen.")
        
        logger.info("Aplicación de análisis de imágenes con OpenALPR iniciada correctamente")
    
    def _init_openalpr(self):
        """Inicializar OpenALPR para detección de placas"""
        try:
            # Inicializar con configuración de US
            self._log_event("Inicializando OpenALPR con país 'us'...")
            self.detector_placas = OpenALPRDetectorCmd(country="us")
            
            if self.detector_placas.loaded:
                self._log_event("OpenALPR inicializado correctamente")
                
                # Configurar para reconocer formato de placas AAA123
                self.detector_placas.set_pattern("aaa999")
                self._log_event("Patrón de placas configurado: aaa999")
            else:
                self._log_event("No se pudo inicializar OpenALPR")
                messagebox.showwarning("Advertencia", "No se pudo inicializar OpenALPR.\nVerifique que está instalado correctamente.")
            
        except Exception as e:
            logger.error(f"Error al inicializar OpenALPR: {e}")
            self._log_event(f"Error al inicializar OpenALPR: {e}")
            messagebox.showerror("Error", f"Error al inicializar OpenALPR: {e}")
            self.detector_placas = None
    
    def _setup_ui(self):
        """Configurar interfaz gráfica"""
        # Marco principal
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo para visualización de imágenes
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Panel para subir imágenes y visualizarlas
        self.image_frame = ttk.LabelFrame(self.left_frame, text="Imagen para Análisis")
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Etiqueta para mostrar la imagen
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Botones para operaciones con imágenes
        self.img_button_frame = ttk.Frame(self.left_frame)
        self.img_button_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botón para cargar imagen
        self.btn_cargar = ttk.Button(self.img_button_frame, text="Cargar Imagen", command=self._cargar_imagen)
        self.btn_cargar.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Botón para procesar imagen
        self.btn_procesar = ttk.Button(self.img_button_frame, text="Procesar Imagen", command=self._procesar_imagen)
        self.btn_procesar.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_procesar.config(state=tk.DISABLED)  # Deshabilitar hasta que se cargue una imagen
        
        # Botón para limpiar
        self.btn_limpiar = ttk.Button(self.img_button_frame, text="Limpiar", command=self._limpiar)
        self.btn_limpiar.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Panel derecho para resultados
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Panel para resultados de detección de placa
        self.detection_frame = ttk.LabelFrame(self.right_frame, text="Resultados de Detección con OpenALPR")
        self.detection_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Imagen de la placa detectada
        self.plate_frame = ttk.Frame(self.detection_frame)
        self.plate_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.plate_label = ttk.Label(self.plate_frame)
        self.plate_label.pack(side=tk.TOP, padx=5, pady=5)
        
        # Información de la placa
        self.info_frame = ttk.Frame(self.detection_frame)
        self.info_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Texto de la placa
        ttk.Label(self.info_frame, text="Placa detectada:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.placa_var = tk.StringVar()
        ttk.Label(self.info_frame, textvariable=self.placa_var, font=("Arial", 16, "bold")).grid(row=0, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Confianza de detección
        ttk.Label(self.info_frame, text="Confianza:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.confianza_var = tk.StringVar()
        ttk.Label(self.info_frame, textvariable=self.confianza_var).grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)
        
        # Botones de acción
        self.action_frame = ttk.Frame(self.right_frame)
        self.action_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Botón para ajustes de OpenALPR
        self.btn_ajustes = ttk.Button(self.action_frame, text="Ajustes de OpenALPR", command=self._mostrar_ajustes)
        self.btn_ajustes.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Botón para guardar resultado
        self.btn_guardar = ttk.Button(self.action_frame, text="Guardar Resultado", command=self._guardar_resultado)
        self.btn_guardar.pack(side=tk.LEFT, padx=5, pady=5)
        self.btn_guardar.config(state=tk.DISABLED)  # Deshabilitar hasta procesar una imagen
        
        # Consola de eventos
        self.console_frame = ttk.LabelFrame(self.right_frame, text="Consola de Eventos")
        self.console_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.console = tk.Text(self.console_frame, height=10, width=40)
        self.console.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.console.config(state=tk.DISABLED)
        
        # Barra de desplazamiento para consola
        self.scrollbar = ttk.Scrollbar(self.console, command=self.console.yview)
        self.console.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def _mostrar_ajustes(self):
        """Mostrar ventana de ajustes para OpenALPR"""
        # Crear ventana de diálogo
        dialog = tk.Toplevel(self.root)
        dialog.title("Ajustes de OpenALPR")
        dialog.geometry("500x400")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Asegurarse de que la ventana está en primer plano
        dialog.lift()
        
        # Configuración de país (solo US para evitar errores)
        ttk.Label(dialog, text="País:").grid(row=0, column=0, sticky=tk.W, padx=10, pady=10)
        pais_var = tk.StringVar(value="us")  # US por defecto
        ttk.Label(dialog, text="us (solo se soporta US)").grid(row=0, column=1, sticky=tk.W, padx=10, pady=10)
        
        # Patrón de placa
        ttk.Label(dialog, text="Patrón:").grid(row=1, column=0, sticky=tk.W, padx=10, pady=10)
        patron_var = tk.StringVar(value="aaa999")  # Patrón para AAA123
        patron_entry = ttk.Entry(dialog, textvariable=patron_var)
        patron_entry.grid(row=1, column=1, sticky=tk.W, padx=10, pady=10)
        
        ttk.Label(dialog, text="a=letra, 9=número").grid(row=1, column=2, sticky=tk.W, padx=5, pady=10)
        
        # Número de resultados
        ttk.Label(dialog, text="Número de resultados:").grid(row=2, column=0, sticky=tk.W, padx=10, pady=10)
        topn_var = tk.StringVar(value="5")
        topn_entry = ttk.Spinbox(dialog, from_=1, to=20, textvariable=topn_var)
        topn_entry.grid(row=2, column=1, sticky=tk.W, padx=10, pady=10)
        
        # Explicación
        ttk.Label(dialog, text="Nota: Los cambios se aplicarán solo para\nfuturos procesamientos de imágenes", 
                  justify=tk.LEFT).grid(row=5, column=0, columnspan=3, sticky=tk.W, padx=10, pady=20)
        
        # Función para aplicar cambios
        def aplicar_cambios():
            try:
                # Configurar opciones adicionales
                if self.detector_placas and self.detector_placas.loaded:
                    self.detector_placas.set_top_n(int(topn_var.get()))
                    
                    # Aplicar patrón si se proporciona
                    if patron_var.get():
                        self.detector_placas.set_pattern(patron_var.get())
                        self._log_event(f"Patrón de placa configurado: {patron_var.get()}")
                    
                    self._log_event(f"Configuración de OpenALPR actualizada: Patrón={patron_var.get()}, TopN={topn_var.get()}")
                    messagebox.showinfo("Éxito", "Configuración actualizada correctamente")
                    dialog.destroy()
                else:
                    messagebox.showerror("Error", "OpenALPR no está inicializado correctamente")
                
            except Exception as e:
                logger.error(f"Error al actualizar configuración: {e}")
                messagebox.showerror("Error", f"Error al actualizar configuración: {e}")
        
        # Botones
        button_frame = ttk.Frame(dialog)
        button_frame.grid(row=6, column=0, columnspan=3, pady=20)
        
        ttk.Button(button_frame, text="Aplicar", command=aplicar_cambios).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Cancelar", command=dialog.destroy).pack(side=tk.LEFT, padx=10)
    
    def _cargar_imagen(self):
        """Cargar imagen desde el sistema de archivos"""
        filetypes = [
            ("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp"),
            ("Todos los archivos", "*.*")
        ]
        
        filepath = filedialog.askopenfilename(
            title="Seleccionar imagen",
            filetypes=filetypes
        )
        
        if filepath:
            try:
                self.image_path = filepath
                self._log_event(f"Imagen cargada: {os.path.basename(filepath)}")
                
                # Cargar imagen con OpenCV
                self.current_image = cv2.imread(filepath)
                
                if self.current_image is None:
                    messagebox.showerror("Error", "No se pudo cargar la imagen seleccionada")
                    self._log_event("Error al cargar la imagen")
                    return
                
                # Mostrar imagen en la interfaz
                self._mostrar_imagen(self.current_image, self.image_label)
                
                # Habilitar botón de procesar
                self.btn_procesar.config(state=tk.NORMAL)
                
                # Limpiar resultados anteriores
                self._limpiar_resultados()
                
            except Exception as e:
                messagebox.showerror("Error", f"Error al cargar la imagen: {e}")
                self._log_event(f"Error al cargar la imagen: {e}")
    
    def _procesar_imagen(self):
        """Procesar la imagen cargada para detectar placa"""
        if self.current_image is None:
            messagebox.showerror("Error", "No hay imagen para procesar")
            return
        
        try:
            self._log_event("Procesando imagen con OpenALPR...")
            
            # Detectar placa
            if self.detector_placas and self.detector_placas.loaded:
                placa, confianza, img_placa = self.detector_placas.detectar_placa(self.current_image)
                
                if placa:
                    self._log_event(f"Placa detectada: {placa} (confianza: {confianza:.2f})")
                    
                    # Mostrar resultados
                    self.placa_var.set(placa)
                    self.confianza_var.set(f"{confianza:.2f}")
                    
                    # Mostrar imagen de la placa
                    if img_placa is not None:
                        self._mostrar_imagen(img_placa, self.plate_label)
                    
                    # Guardar resultados para registro posterior
                    self.detection_results = {
                        'placa': placa,
                        'confianza': confianza,
                        'img_placa': img_placa
                    }
                    
                    # Habilitar botón de guardar
                    self.btn_guardar.config(state=tk.NORMAL)
                else:
                    self._log_event("No se detectó placa en la imagen")
                    messagebox.showinfo("Resultado", "No se detectó placa en la imagen")
            else:
                self._log_event("OpenALPR no está disponible o no inicializado correctamente")
                messagebox.showwarning("Advertencia", "OpenALPR no está disponible o no inicializado correctamente")
                
        except Exception as e:
            messagebox.showerror("Error", f"Error al procesar la imagen: {e}")
            self._log_event(f"Error al procesar la imagen: {e}")
    
    def _guardar_resultado(self):
        """Guardar el resultado del análisis"""
        if not self.detection_results:
            messagebox.showerror("Error", "No hay resultados para guardar")
            return
        
        try:
            # Crear directorio de resultados si no existe
            os.makedirs("resultados", exist_ok=True)
            
            # Nombre de archivo con timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            placa = self.detection_results['placa']
            
            # Guardar imagen original con anotaciones
            resultado_img = self.current_image.copy()
            
            # Dibujar recuadro de placa si tenemos coordenadas
            if 'img_placa' in self.detection_results and self.detection_results['img_placa'] is not None:
                # En este caso, la imagen de la placa ya tiene el recuadro dibujado
                # porque viene directamente del método detectar_placa

                # Escribir información en la imagen
                texto = f"Placa: {placa} ({self.detection_results['confianza']:.2f})"
                cv2.putText(resultado_img, texto, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Guardar imagen con resultados
            output_path = os.path.join("resultados", f"placa_{placa}_{timestamp}.jpg")
            cv2.imwrite(output_path, resultado_img)
            
            # Guardar imagen recortada de la placa
            if 'img_placa' in self.detection_results and self.detection_results['img_placa'] is not None:
                plate_path = os.path.join("resultados", f"placa_{placa}_{timestamp}_recorte.jpg")
                cv2.imwrite(plate_path, self.detection_results['img_placa'])
            
            # Guardar datos de detección en archivo de texto
            info_path = os.path.join("resultados", f"placa_{placa}_{timestamp}_info.txt")
            with open(info_path, 'w') as f:
                f.write(f"Placa detectada: {placa}\n")
                f.write(f"Confianza: {self.detection_results['confianza']:.2f}\n")
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Imagen original: {self.image_path}\n")
            
            self._log_event(f"Resultados guardados en: {output_path}")
            messagebox.showinfo("Éxito", f"Resultados guardados en: {output_path}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al guardar resultados: {e}")
            self._log_event(f"Error al guardar resultados: {e}")
    
    def _limpiar(self):
        """Limpiar imagen y resultados"""
        self.image_path = None
        self.current_image = None
        self.detection_results = None
        
        # Limpiar visualización de imagen
        self.image_label.config(image="")
        
        # Limpiar resultados
        self._limpiar_resultados()
        
        # Deshabilitar botones
        self.btn_procesar.config(state=tk.DISABLED)
        self.btn_guardar.config(state=tk.DISABLED)
        
        self._log_event("Visualización limpiada")
    
    def _limpiar_resultados(self):
        """Limpiar resultados de detección"""
        self.placa_var.set("")
        self.confianza_var.set("")
        self.plate_label.config(image="")
        
        # Deshabilitar botón de guardar
        self.btn_guardar.config(state=tk.DISABLED)
    
    def _mostrar_imagen(self, imagen, label):
        """
        Mostrar imagen en una etiqueta
        
        Args:
            imagen: Imagen en formato OpenCV (numpy array)
            label: Etiqueta tkinter donde mostrar la imagen
        """
        try:
            # Convertir BGR a RGB
            if len(imagen.shape) == 3:
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            else:
                # Si es una imagen en escala de grises, convertir a RGB
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)
            
            # Determinar tamaño máximo
            panel_width = label.winfo_width()
            panel_height = label.winfo_height()
            
            # Si el panel no tiene tamaño aún, usar valores por defecto
            if panel_width <= 1:
                panel_width = 640 if label == self.image_label else 200
            
            if panel_height <= 1:
                panel_height = 480 if label == self.image_label else 100
            
            # Calcular tamaño para ajustar a la ventana
            height, width = imagen.shape[:2]
            
            # Redimensionar manteniendo proporción
            if width > panel_width or height > panel_height:
                ratio = min(panel_width / width, panel_height / height)
                new_width = int(width * ratio)
                new_height = int(height * ratio)
                imagen_rgb = cv2.resize(imagen_rgb, (new_width, new_height))
            
            # Convertir a formato PIL
            pil_img = Image.fromarray(imagen_rgb)
            
            # Convertir a formato Tkinter
            tk_img = ImageTk.PhotoImage(image=pil_img)
            
            # Mostrar en la etiqueta
            label.config(image=tk_img)
            label.image = tk_img  # Mantener referencia
            
        except Exception as e:
            logger.error(f"Error al mostrar imagen: {e}")
            label.config(text=f"Error al mostrar imagen: {e}")
    
    def _log_event(self, message):
        """
        Registrar evento en la consola de la UI
        
        Args:
            message: Mensaje a registrar
        """
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Agregar a la consola
        if hasattr(self, 'console') and self.console:
            self.console.config(state=tk.NORMAL)
            self.console.insert(tk.END, log_entry)
            self.console.see(tk.END)  # Desplazar al final
            self.console.config(state=tk.DISABLED)
        
        # Registrar en el logger
        logger.info(message)
    
    def shutdown(self):
        """Cerrar aplicación de forma ordenada"""
        logger.info("Cerrando aplicación de análisis con OpenALPR...")
        
        # No hay recursos específicos que liberar
        
        logger.info("Aplicación cerrada correctamente")

def main():
    """Función principal"""
    # Iniciar aplicación
    root = tk.Tk()
    app = AnalisisImagenApp(root)
    
    # Configurar cierre ordenado de la aplicación
    def on_closing():
        if messagebox.askokcancel("Salir", "¿Desea cerrar la aplicación?"):
            app.shutdown()
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Iniciar bucle principal de Tkinter
    root.mainloop()

if __name__ == "__main__":
    main()

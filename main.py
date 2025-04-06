#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sistema de Parqueadero con IA para Jetson Orin Nano
Archivo principal que inicializa y coordina todos los componentes
"""

import os
import sys
import time
import argparse
import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox
import cv2

# Agregar directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importar configuraciones
from config.settings import SETTINGS
from config.database import Database

# Importar componentes del sistema
from src.utils.ptz_camera import PTZCamera
from src.utils.gpio_sensor import GPIOSensor
from src.ai.detector_placas import DetectorPlacas
from src.ai.clasificador_vehiculos import ClasificadorVehiculos

# Importar modelos de datos
from modelos.vehiculo import Vehiculo
from modelos.usuario import Usuario
from modelos.estado import Estado

# Importar componentes de UI
from src.ui.ptz_controls import PTZControlFrame
from src.ui.entrada_dialog import EntradaDialog
from src.ui.salida_dialog import SalidaDialog
from src.ui.consulta_view import ConsultaView

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', f'parqueadero_{time.strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('parqueadero')

class ParqueaderoApp:
    """Aplicación principal del sistema de parqueadero con IA"""
    
    def __init__(self, root, args):
        """
        Inicializa la aplicación principal
        
        Args:
            root: Ventana principal de Tkinter
            args: Argumentos de línea de comandos
        """
        self.root = root
        self.args = args
        self.root.title("Sistema de Parqueadero con IA")
        self.root.geometry("1280x720")
        
        # Bandera para ejecución
        self.running = True
        
        # Crear directorios si no existen
        self._create_directories()
        
        # Inicializar componentes
        self._init_database()
        self._init_camera()
        self._init_ai_models()
        self._init_gpio_sensors()
        
        # Interfaz gráfica
        self._setup_ui()
        
        # Iniciar procesamiento
        self.process_thread = threading.Thread(target=self._process_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Programar actualización de UI
        self.root.after(100, self._update_ui)
        
        logger.info("Sistema de Parqueadero con IA iniciado correctamente")
    
    def _create_directories(self):
        """Crear directorios necesarios si no existen"""
        dirs = [
            'logs',
            'imagenes/entrada',
            'imagenes/salida',
            'imagenes/procesado'
        ]
        
        for directory in dirs:
            os.makedirs(directory, exist_ok=True)
    
    def _init_database(self):
        """Inicializar conexión a base de datos"""
        try:
            self.db = Database()
            logger.info("Conexión a base de datos establecida")
        except Exception as e:
            logger.error(f"Error al conectar a base de datos: {e}")
            messagebox.showerror("Error", f"No se pudo conectar a la base de datos: {e}")
            sys.exit(1)
    
    def _init_camera(self):
        """Inicializar cámara PTZ"""
        try:
            # Configurar cámara PTZ con los parámetros del archivo de configuración
            self.camera = PTZCamera(
                ip=SETTINGS['camera']['ip'],
                port=SETTINGS['camera']['port'],
                username=SETTINGS['camera']['username'],
                password=SETTINGS['camera']['password'],
                rtsp_url=SETTINGS['camera']['rtsp_url']
            )
            
            # Iniciar stream de video
            self.camera.start()
            logger.info("Cámara PTZ inicializada correctamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar cámara PTZ: {e}")
            messagebox.showerror("Error", f"No se pudo inicializar la cámara: {e}")
            self.camera = None
    
    def _init_ai_models(self):
        """Inicializar modelos de IA"""
        try:
            # Inicializar detector de placas
            self.detector_placas = DetectorPlacas(
                modelo_detector=SETTINGS['ai']['modelo_detector_placas'],
                modelo_ocr=SETTINGS['ai']['modelo_ocr_placas']
            )
            
            # Inicializar clasificador de vehículos (opcional)
            if SETTINGS['ai']['usar_clasificador_vehiculos']:
                self.clasificador_vehiculos = ClasificadorVehiculos(
                    modelo=SETTINGS['ai']['modelo_clasificador_vehiculos']
                )
            else:
                self.clasificador_vehiculos = None
                
            logger.info("Modelos de IA inicializados correctamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar modelos de IA: {e}")
            messagebox.showwarning("Advertencia", f"Error al cargar modelos de IA: {e}\nEl sistema funcionará en modo degradado.")
            self.detector_placas = None
            self.clasificador_vehiculos = None
    
    def _init_gpio_sensors(self):
        """Inicializar sensores GPIO para fotoceldas"""
        if not SETTINGS['gpio']['enabled']:
            logger.info("Sensores GPIO deshabilitados en configuración")
            self.sensor_entrada = None
            self.sensor_salida = None
            return
            
        try:
            # Inicializar sensor de entrada
            self.sensor_entrada = GPIOSensor(
                pin=SETTINGS['gpio']['pin_entrada'],
                callback=self._on_vehicle_detected_entrada,
                debounce_ms=SETTINGS['gpio']['debounce_ms']
            )
            self.sensor_entrada.start()
            
            # Inicializar sensor de salida (si está configurado)
            if SETTINGS['gpio']['pin_salida'] > 0:
                self.sensor_salida = GPIOSensor(
                    pin=SETTINGS['gpio']['pin_salida'],
                    callback=self._on_vehicle_detected_salida,
                    debounce_ms=SETTINGS['gpio']['debounce_ms']
                )
                self.sensor_salida.start()
            else:
                self.sensor_salida = None
                
            logger.info("Sensores GPIO inicializados correctamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar sensores GPIO: {e}")
            messagebox.showwarning("Advertencia", f"Error al inicializar sensores GPIO: {e}\nEl sistema funcionará sin detección automática.")
            self.sensor_entrada = None
            self.sensor_salida = None
    
    def _setup_ui(self):
        """Configurar interfaz gráfica"""
        # Crear marco principal
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Panel izquierdo para video y controles de cámara
        self.left_frame = ttk.Frame(self.main_frame)
        self.left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Panel de video
        self.video_frame = ttk.LabelFrame(self.left_frame, text="Cámara PTZ")
        self.video_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Controles de cámara PTZ
        if SETTINGS['ui']['show_ptz_controls'] and self.camera:
            self.ptz_controls = PTZControlFrame(self.left_frame, self.camera)
            self.ptz_controls.pack(fill=tk.X, padx=5, pady=5)
        
        # Panel derecho para operaciones
        self.right_frame = ttk.Frame(self.main_frame)
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        # Botones de operación
        self.op_frame = ttk.LabelFrame(self.right_frame, text="Operaciones")
        self.op_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.btn_entrada = ttk.Button(self.op_frame, text="Registrar Entrada", command=self._on_entrada_manual)
        self.btn_entrada.pack(fill=tk.X, padx=5, pady=2)
        
        self.btn_salida = ttk.Button(self.op_frame, text="Registrar Salida", command=self._on_salida_manual)
        self.btn_salida.pack(fill=tk.X, padx=5, pady=2)
        
        self.btn_consulta = ttk.Button(self.op_frame, text="Consultar Vehículos", command=self._on_consulta)
        self.btn_consulta.pack(fill=tk.X, padx=5, pady=2)
        
        # Panel de estado
        self.status_frame = ttk.LabelFrame(self.right_frame, text="Estado del Sistema")
        self.status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Estado de cámara
        self.lbl_camera_status = ttk.Label(self.status_frame, text="Cámara: Inicializando...")
        self.lbl_camera_status.pack(anchor=tk.W, padx=5, pady=2)
        
        # Estado de IA
        self.lbl_ai_status = ttk.Label(self.status_frame, text="IA: Inicializando...")
        self.lbl_ai_status.pack(anchor=tk.W, padx=5, pady=2)
        
        # Estado de sensores
        self.lbl_sensor_entrada = ttk.Label(self.status_frame, text="Sensor Entrada: Desconocido")
        self.lbl_sensor_entrada.pack(anchor=tk.W, padx=5, pady=2)
        
        if self.sensor_salida:
            self.lbl_sensor_salida = ttk.Label(self.status_frame, text="Sensor Salida: Desconocido")
            self.lbl_sensor_salida.pack(anchor=tk.W, padx=5, pady=2)
        
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
    
    def _on_vehicle_detected_entrada(self, is_present):
        """
        Callback para cuando se detecta un vehículo en la entrada
        
        Args:
            is_present: True si se detecta vehículo, False si no
        """
        if is_present:
            self._log_event("Vehículo detectado en la entrada")
            
            # Mover cámara a posición de entrada
            if self.camera:
                self._log_event("Moviendo cámara a posición de entrada")
                self.camera.go_to_preset(SETTINGS['camera']['preset_entrada'])
                
                # Esperar a que la cámara se posicione
                time.sleep(1)
                
                # Capturar imagen
                self._procesar_entrada()
        else:
            self._log_event("Vehículo ya no está presente en la entrada")
    
    def _on_vehicle_detected_salida(self, is_present):
        """
        Callback para cuando se detecta un vehículo en la salida
        
        Args:
            is_present: True si se detecta vehículo, False si no
        """
        if is_present:
            self._log_event("Vehículo detectado en la salida")
            
            # Mover cámara a posición de salida
            if self.camera:
                self._log_event("Moviendo cámara a posición de salida")
                self.camera.go_to_preset(SETTINGS['camera']['preset_salida'])
                
                # Esperar a que la cámara se posicione
                time.sleep(1)
                
                # Capturar imagen
                self._procesar_salida()
        else:
            self._log_event("Vehículo ya no está presente en la salida")
    
    def _on_entrada_manual(self):
        """Manejador para entrada manual de vehículo"""
        # Mover cámara a posición de entrada
        if self.camera:
            self._log_event("Moviendo cámara a posición de entrada")
            self.camera.go_to_preset(SETTINGS['camera']['preset_entrada'])
            
            # Esperar a que la cámara se posicione
            time.sleep(1)
        
        # Capturar imagen actual
        if self.camera:
            frame = self.camera.get_frame()
            if frame is not None:
                # Detectar placa
                if self.detector_placas:
                    self._log_event("Analizando imagen para detectar placa...")
                    placa, confianza, img_placa = self.detector_placas.detectar_placa(frame)
                    
                    if placa:
                        self._log_event(f"Placa detectada: {placa} (confianza: {confianza:.2f})")
                    else:
                        placa = ""
                        self._log_event("No se detectó placa")
                else:
                    placa = ""
                    img_placa = None
                
                # Guardar imagen de entrada
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join("imagenes", "entrada", f"entrada_{timestamp}.jpg")
                cv2.imwrite(img_path, frame)
                
                # Abrir diálogo de entrada
                dialog = EntradaDialog(self.root, placa, img_path, img_placa)
                if dialog.result:
                    # Registro exitoso
                    self._log_event(f"Registro de entrada exitoso: {dialog.result['placa']}")
                    
                    # Guardar en base de datos
                    try:
                        vehiculo = Vehiculo()
                        # Primero ver si el vehículo ya existe
                        existe = vehiculo.buscar_por_placa(dialog.result['placa'])
                        
                        if not existe:
                            # Crear nuevo registro de vehículo
                            vehiculo.placa = dialog.result['placa']
                            vehiculo.tipo = dialog.result['tipo']
                            vehiculo.color = dialog.result['color']
                            vehiculo.marca = dialog.result['marca']
                            vehiculo.modelo = dialog.result['modelo']
                            
                            # Si hay propietario, asignarlo o crearlo
                            if dialog.result.get('propietario'):
                                usuario = Usuario()
                                usuario.nombre = dialog.result['propietario']
                                usuario.documento = dialog.result.get('documento', '')
                                usuario.telefono = dialog.result.get('telefono', '')
                                id_usuario = usuario.guardar()
                                vehiculo.id_propietario = id_usuario
                            
                            vehiculo.guardar()
                        
                        # Registrar entrada
                        estado = Estado()
                        estado.registrar_entrada(
                            dialog.result['placa'],
                            img_path,
                            dialog.result.get('observaciones', '')
                        )
                        
                        self._log_event("Entrada registrada en base de datos")
                        messagebox.showinfo("Éxito", "Entrada registrada correctamente")
                    except Exception as e:
                        logger.error(f"Error al registrar entrada: {e}")
                        self._log_event(f"Error al registrar entrada: {e}")
                        messagebox.showerror("Error", f"Error al registrar entrada: {e}")
            else:
                self._log_event("No se pudo capturar imagen de la cámara")
                messagebox.showerror("Error", "No se pudo capturar imagen de la cámara")
        else:
            self._log_event("La cámara no está disponible")
            messagebox.showerror("Error", "La cámara no está disponible")
    
    def _on_salida_manual(self):
        """Manejador para salida manual de vehículo"""
        # Mover cámara a posición de salida
        if self.camera:
            self._log_event("Moviendo cámara a posición de salida")
            self.camera.go_to_preset(SETTINGS['camera']['preset_salida'])
            
            # Esperar a que la cámara se posicione
            time.sleep(1)
        
        # Capturar imagen actual
        if self.camera:
            frame = self.camera.get_frame()
            if frame is not None:
                # Detectar placa
                if self.detector_placas:
                    self._log_event("Analizando imagen para detectar placa...")
                    placa, confianza, img_placa = self.detector_placas.detectar_placa(frame)
                    
                    if placa:
                        self._log_event(f"Placa detectada: {placa} (confianza: {confianza:.2f})")
                    else:
                        placa = ""
                        self._log_event("No se detectó placa")
                else:
                    placa = ""
                    img_placa = None
                
                # Guardar imagen de salida
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join("imagenes", "salida", f"salida_{timestamp}.jpg")
                cv2.imwrite(img_path, frame)
                
                # Verificar si hay registro de entrada para esta placa
                if placa:
                    estado = Estado()
                    registro_entrada = estado.buscar_entrada_activa(placa)
                    
                    if registro_entrada:
                        # Abrir diálogo de salida con datos del vehículo
                        dialog = SalidaDialog(self.root, placa, registro_entrada, img_path, img_placa)
                        if dialog.result:
                            # Registro exitoso
                            self._log_event(f"Registro de salida exitoso: {placa}")
                            
                            # Registrar salida
                            try:
                                estado.registrar_salida(
                                    placa,
                                    img_path,
                                    dialog.result.get('observaciones', '')
                                )
                                
                                self._log_event("Salida registrada en base de datos")
                                messagebox.showinfo("Éxito", "Salida registrada correctamente")
                            except Exception as e:
                                logger.error(f"Error al registrar salida: {e}")
                                self._log_event(f"Error al registrar salida: {e}")
                                messagebox.showerror("Error", f"Error al registrar salida: {e}")
                    else:
                        self._log_event(f"No hay registro de entrada para la placa {placa}")
                        messagebox.showwarning("Advertencia", f"No hay registro de entrada para la placa {placa}")
                else:
                    # Permitir buscar placa manualmente
                    dialog = SalidaDialog(self.root, "", None, img_path, None)
                    if dialog.result:
                        # Registro exitoso
                        self._log_event(f"Registro de salida exitoso: {dialog.result['placa']}")
                        
                        # Registrar salida
                        try:
                            estado = Estado()
                            estado.registrar_salida(
                                dialog.result['placa'],
                                img_path,
                                dialog.result.get('observaciones', '')
                            )
                            
                            self._log_event("Salida registrada en base de datos")
                            messagebox.showinfo("Éxito", "Salida registrada correctamente")
                        except Exception as e:
                            logger.error(f"Error al registrar salida: {e}")
                            self._log_event(f"Error al registrar salida: {e}")
                            messagebox.showerror("Error", f"Error al registrar salida: {e}")
            else:
                self._log_event("No se pudo capturar imagen de la cámara")
                messagebox.showerror("Error", "No se pudo capturar imagen de la cámara")
        else:
            self._log_event("La cámara no está disponible")
            messagebox.showerror("Error", "La cámara no está disponible")
    
    def _on_consulta(self):
        """Manejador para consulta de vehículos"""
        # Abrir ventana de consulta
        ConsultaView(self.root)
    
    def _procesar_entrada(self):
        """Procesar entrada automática de vehículo"""
        if self.camera:
            frame = self.camera.get_frame()
            if frame is not None:
                # Detectar placa
                if self.detector_placas:
                    self._log_event("Analizando imagen para detectar placa...")
                    placa, confianza, img_placa = self.detector_placas.detectar_placa(frame)
                    
                    if placa and confianza > SETTINGS['ai']['umbral_confianza']:
                        self._log_event(f"Placa detectada: {placa} (confianza: {confianza:.2f})")
                        
                        # Guardar imagen de entrada
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        img_path = os.path.join("imagenes", "entrada", f"entrada_{timestamp}.jpg")
                        cv2.imwrite(img_path, frame)
                        
                        # Verificar si el vehículo ya existe
                        vehiculo = Vehiculo()
                        existe = vehiculo.buscar_por_placa(placa)
                        
                        if existe:
                            # Registrar entrada automáticamente
                            try:
                                estado = Estado()
                                estado.registrar_entrada(placa, img_path, "Entrada automática por fotocelda")
                                self._log_event(f"Entrada automática registrada para vehículo: {placa}")
                            except Exception as e:
                                logger.error(f"Error al registrar entrada automática: {e}")
                                self._log_event(f"Error al registrar entrada automática: {e}")
                        else:
                            # Abrir diálogo para registro manual
                            self._log_event("Vehículo no registrado, abriendo diálogo manual")
                            dialog = EntradaDialog(self.root, placa, img_path, img_placa)
                            if dialog.result:
                                # Registro manual exitoso
                                self._log_event(f"Registro de entrada exitoso: {dialog.result['placa']}")
                                
                                # Guardar en base de datos
                                try:
                                    vehiculo = Vehiculo()
                                    vehiculo.placa = dialog.result['placa']
                                    vehiculo.tipo = dialog.result['tipo']
                                    vehiculo.color = dialog.result['color']
                                    vehiculo.marca = dialog.result['marca']
                                    vehiculo.modelo = dialog.result['modelo']
                                    
                                    # Si hay propietario, asignarlo o crearlo
                                    if dialog.result.get('propietario'):
                                        usuario = Usuario()
                                        usuario.nombre = dialog.result['propietario']
                                        usuario.documento = dialog.result.get('documento', '')
                                        usuario.telefono = dialog.result.get('telefono', '')
                                        id_usuario = usuario.guardar()
                                        vehiculo.id_propietario = id_usuario
                                    
                                    vehiculo.guardar()
                                    
                                    # Registrar entrada
                                    estado = Estado()
                                    estado.registrar_entrada(
                                        dialog.result['placa'],
                                        img_path,
                                        dialog.result.get('observaciones', '')
                                    )
                                    
                                    self._log_event("Entrada registrada en base de datos")
                                except Exception as e:
                                    logger.error(f"Error al registrar entrada: {e}")
                                    self._log_event(f"Error al registrar entrada: {e}")
                    else:
                        if placa:
                            self._log_event(f"Placa detectada con baja confianza: {placa} ({confianza:.2f})")
                        else:
                            self._log_event("No se detectó ninguna placa")
                else:
                    self._log_event("Detector de placas no disponible")
            else:
                self._log_event("No se pudo capturar imagen de la cámara")
        else:
            self._log_event("La cámara no está disponible")
    
    def _procesar_salida(self):
        """Procesar salida automática de vehículo"""
        if self.camera:
            frame = self.camera.get_frame()
            if frame is not None:
                # Detectar placa
                if self.detector_placas:
                    self._log_event("Analizando imagen para detectar placa...")
                    placa, confianza, img_placa = self.detector_placas.detectar_placa(frame)
                    
                    if placa and confianza > SETTINGS['ai']['umbral_confianza']:
                        self._log_event(f"Placa detectada: {placa} (confianza: {confianza:.2f})")
                        
                        # Guardar imagen de salida
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        img_path = os.path.join("imagenes", "salida", f"salida_{timestamp}.jpg")
                        cv2.imwrite(img_path, frame)
                        
                        # Verificar si hay registro de entrada para esta placa
                        estado = Estado()
                        registro_entrada = estado.buscar_entrada_activa(placa)
                        
                        if registro_entrada:
                            # Registrar salida automáticamente
                            try:
                                estado.registrar_salida(placa, img_path, "Salida automática por fotocelda")
                                self._log_event(f"Salida automática registrada para vehículo: {placa}")
                            except Exception as e:
                                logger.error(f"Error al registrar salida automática: {e}")
                                self._log_event(f"Error al registrar salida automática: {e}")
                        else:
                            self._log_event(f"No hay registro de entrada para la placa {placa}")
                    else:
                        if placa:
                            self._log_event(f"Placa detectada con baja confianza: {placa} ({confianza:.2f})")
                        else:
                            self._log_event("No se detectó ninguna placa")
                else:
                    self._log_event("Detector de placas no disponible")
            else:
                self._log_event("No se pudo capturar imagen de la cámara")
        else:
            self._log_event("La cámara no está disponible")
    
    def _process_loop(self):
        """Bucle principal de procesamiento en hilo separado"""
        while self.running:
            # Actualizar estado de la cámara
            if self.camera:
                self.camera.update()
            
            # Dormir para evitar uso excesivo de CPU
            time.sleep(0.1)
    
    def _update_ui(self):
        """Actualizar interfaz de usuario"""
        if not self.running:
            return
            
        # Actualizar imagen de la cámara
        if self.camera and self.camera.is_connected():
            frame = self.camera.get_frame()
            if frame is not None:
                # Convertir frame a formato compatible con Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w = frame.shape[:2]
                
                # Redimensionar si es necesario para caber en la UI
                max_w = 800
                max_h = 600
                if w > max_w or h > max_h:
                    scale = min(max_w / w, max_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
                
                # Mostrar en la UI
                from PIL import Image, ImageTk
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(image=img)
                self.video_label.img_tk = img_tk  # Guardar referencia para evitar que el recolector de basura la elimine
                self.video_label.configure(image=img_tk)
        
        # Actualizar estados en la UI
        if self.camera:
            status = "Conectada" if self.camera.is_connected() else "Desconectada"
            self.lbl_camera_status.config(text=f"Cámara: {status}")
        else:
            self.lbl_camera_status.config(text="Cámara: No disponible")
        
        if self.detector_placas:
            self.lbl_ai_status.config(text="IA: Activa")
        else:
            self.lbl_ai_status.config(text="IA: No disponible")
        
        if self.sensor_entrada:
            estado = "Activo" if self.sensor_entrada.is_running else "Inactivo"
            estado_deteccion = "Vehículo Detectado" if self.sensor_entrada.read_state() else "Sin Vehículo"
            self.lbl_sensor_entrada.config(text=f"Sensor Entrada: {estado} - {estado_deteccion}")
        else:
            self.lbl_sensor_entrada.config(text="Sensor Entrada: No disponible")
        
        if self.sensor_salida:
            estado = "Activo" if self.sensor_salida.is_running else "Inactivo"
            estado_deteccion = "Vehículo Detectado" if self.sensor_salida.read_state() else "Sin Vehículo"
            self.lbl_sensor_salida.config(text=f"Sensor Salida: {estado} - {estado_deteccion}")
        
        # Programar próxima actualización
        self.root.after(100, self._update_ui)
    
    def _log_event(self, message):
        """
        Registrar evento en la consola de la UI
        
        Args:
            message: Mensaje a registrar
        """
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        
        # Agregar a la consola
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, log_entry)
        self.console.see(tk.END)  # Desplazar al final
        self.console.config(state=tk.DISABLED)
        
        # Registrar en el logger
        logger.info(message)
    
    def shutdown(self):
        """Cerrar aplicación de forma ordenada"""
        logger.info("Cerrando aplicación...")
        self.running = False
        
        # Detener componentes
        if self.camera:
            self.camera.stop()
        
        if self.sensor_entrada:
            self.sensor_entrada.stop()
        
        if self.sensor_salida:
            self.sensor_salida.stop()
        
        logger.info("Aplicación cerrada correctamente")


def test_camaras(args):
    """Función para probar cámaras"""
    from src.utils.ptz_camera import PTZCamera
    
    print("=== Prueba de cámaras ===")
    try:
        print(f"Conectando a cámara PTZ en {SETTINGS['camera']['ip']}...")
        
        camera = PTZCamera(
            ip=SETTINGS['camera']['ip'],
            port=SETTINGS['camera']['port'],
            username=SETTINGS['camera']['username'],
            password=SETTINGS['camera']['password'],
            rtsp_url=SETTINGS['camera']['rtsp_url']
        )
        
        camera.start()
        
        print("Cámara conectada. Mostrando stream de video (presione ESC para salir)...")
        
        cv2.namedWindow("Test Cámara PTZ", cv2.WINDOW_NORMAL)
        
        start_time = time.time()
        frames = 0
        
        while True:
            frame = camera.get_frame()
            
            if frame is not None:
                frames += 1
                
                # Mostrar FPS
                elapsed_time = time.time() - start_time
                if elapsed_time >= 1.0:
                    fps = frames / elapsed_time
                    print(f"FPS: {fps:.2f}")
                    frames = 0
                    start_time = time.time()
                
                # Mostrar frame
                cv2.imshow("Test Cámara PTZ", frame)
            
            # Salir con ESC
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
        
        cv2.destroyAllWindows()
        camera.stop()
        print("Prueba de cámara finalizada")
        
    except Exception as e:
        print(f"Error al probar cámara: {e}")


def test_gpio(args):
    """Función para probar GPIO"""
    from src.utils.gpio_sensor import GPIOSensor
    
    print("=== Prueba de sensores GPIO ===")
    try:
        # Callback de prueba
        def on_sensor_change(is_present):
            if is_present:
                print("Vehículo detectado")
            else:
                print("Vehículo retirado")
        
        print(f"Inicializando sensor GPIO en pin {SETTINGS['gpio']['pin_entrada']}...")
        
        sensor = GPIOSensor(
            pin=SETTINGS['gpio']['pin_entrada'],
            callback=on_sensor_change,
            debounce_ms=SETTINGS['gpio']['debounce_ms']
        )
        
        sensor.start()
        
        print("Sensor GPIO inicializado. Monitoreo activo (presione Ctrl+C para salir)...")
        print("Información: Interrumpa el haz de la fotocelda para probar la detección")
        
        try:
            while True:
                state = sensor.read_state()
                status = "Vehículo Detectado" if state else "Sin Vehículo"
                print(f"\rEstado actual: {status}", end="", flush=True)
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\nPrueba de GPIO finalizada por el usuario")
        
        sensor.stop()
        
    except Exception as e:
        print(f"Error al probar GPIO: {e}")


def test_ia(args):
    """Función para probar modelos de IA"""
    from src.ai.detector_placas import DetectorPlacas
    import glob
    
    print("=== Prueba de modelos de IA ===")
    try:
        print("Inicializando detector de placas...")
        
        detector = DetectorPlacas(
            modelo_detector=SETTINGS['ai']['modelo_detector_placas'],
            modelo_ocr=SETTINGS['ai']['modelo_ocr_placas']
        )
        
        print("Detector inicializado.")
        
        # Buscar imágenes de prueba
        imagenes = []
        for ext in ['jpg', 'jpeg', 'png']:
            imagenes.extend(glob.glob(f"imagenes/entrada/*.{ext}"))
            imagenes.extend(glob.glob(f"imagenes/salida/*.{ext}"))
        
        if not imagenes:
            print("No se encontraron imágenes de prueba. Capturando de la cámara...")
            
            # Inicializar cámara
            from src.utils.ptz_camera import PTZCamera
            
            camera = PTZCamera(
                ip=SETTINGS['camera']['ip'],
                port=SETTINGS['camera']['port'],
                username=SETTINGS['camera']['username'],
                password=SETTINGS['camera']['password'],
                rtsp_url=SETTINGS['camera']['rtsp_url']
            )
            
            camera.start()
            
            # Capturar frame
            print("Esperando captura de imagen...")
            time.sleep(2)  # Dar tiempo a la cámara para inicializarse
            
            frame = camera.get_frame()
            if frame is not None:
                test_path = "imagenes/test_ia.jpg"
                cv2.imwrite(test_path, frame)
                imagenes = [test_path]
                print(f"Imagen capturada y guardada en {test_path}")
            else:
                print("No se pudo capturar imagen de la cámara")
                return
            
            camera.stop()
        
        # Procesar imágenes
        print(f"Procesando {len(imagenes)} imágenes...")
        
        for img_path in imagenes:
            print(f"\nAnalizando imagen: {img_path}")
            
            img = cv2.imread(img_path)
            if img is None:
                print(f"No se pudo cargar la imagen {img_path}")
                continue
            
            # Detectar placa
            placa, confianza, img_placa = detector.detectar_placa(img)
            
            if placa:
                print(f"Placa detectada: {placa} (confianza: {confianza:.2f})")
                
                # Mostrar imagen original y recorte de placa
                cv2.namedWindow("Imagen Original", cv2.WINDOW_NORMAL)
                cv2.imshow("Imagen Original", img)
                
                if img_placa is not None:
                    cv2.namedWindow("Placa Detectada", cv2.WINDOW_NORMAL)
                    cv2.imshow("Placa Detectada", img_placa)
                
                print("Presione cualquier tecla para continuar o ESC para salir...")
                key = cv2.waitKey(0) & 0xFF
                cv2.destroyAllWindows()
                
                if key == 27:  # ESC
                    break
            else:
                print(f"No se detectó placa en la imagen {img_path}")
        
        print("Prueba de IA finalizada")
        
    except Exception as e:
        print(f"Error al probar modelos de IA: {e}")


def optimizar_rendimiento(args):
    """Función para optimizar rendimiento de la Jetson"""
    print("=== Optimización de rendimiento de Jetson ===")
    try:
        # Configurar modo de máximo rendimiento
        print("Configurando modo de máximo rendimiento...")
        os.system("sudo nvpmodel -m 0")
        
        # Establecer frecuencias máximas
        print("Estableciendo frecuencias máximas...")
        os.system("sudo jetson_clocks")
        
        print("Optimización completada. La Jetson está configurada para máximo rendimiento.")
        print("Nota: Esta configuración aumentará el consumo de energía y temperatura.")
        
    except Exception as e:
        print(f"Error al optimizar rendimiento: {e}")


def main():
    """Función principal"""
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Sistema de Parqueadero con IA para Jetson Orin Nano")
    
    # Argumentos para modos de prueba
    parser.add_argument("--test-camaras", action="store_true", help="Probar conexión con cámaras")
    parser.add_argument("--test-gpio", action="store_true", help="Probar sensores GPIO")
    parser.add_argument("--test-ia", action="store_true", help="Probar modelos de IA")
    parser.add_argument("--optimizar", action="store_true", help="Optimizar rendimiento de la Jetson")
    
    args = parser.parse_args()
    
    # Ejecutar en modo de prueba si se especifica
    if args.test_camaras:
        test_camaras(args)
        return
    
    if args.test_gpio:
        test_gpio(args)
        return
    
    if args.test_ia:
        test_ia(args)
        return
    
    if args.optimizar:
        optimizar_rendimiento(args)
        return
    
    # Ejecutar aplicación principal
    root = tk.Tk()
    app = ParqueaderoApp(root, args)
    
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

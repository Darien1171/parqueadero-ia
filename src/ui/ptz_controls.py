#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Panel de control para cámara PTZ con botones direccionales, zoom y presets
"""
import logging
import threading
import tkinter as tk
from tkinter import ttk

logger = logging.getLogger('parqueadero.ui.ptz_controls')

class PTZControlFrame(ttk.Frame):
    """
    Marco con controles para mover cámara PTZ y activar presets
    """
    
    def __init__(self, parent, camera, *args, **kwargs):
        """
        Inicializa panel de control PTZ
        
        Args:
            parent: Widget padre
            camera: Instancia de PTZCamera
        """
        super().__init__(parent, *args, **kwargs)
        
        self.camera = camera
        
        # Verificar si la cámara tiene capacidades PTZ
        self.ptz_available = self.camera and self.camera.has_ptz()
        
        # Configurar interfaz
        self._create_widgets()
        
        # Cargar presets disponibles
        self._load_presets()
    
    def _create_widgets(self):
        """Crear widgets del panel de control"""
        # Título del marco
        self.title_label = ttk.Label(self, text="Control PTZ", font=("Arial", 12, "bold"))
        self.title_label.grid(row=0, column=0, columnspan=3, pady=5)
        
        # Panel principal para controles PTZ
        self.control_frame = ttk.LabelFrame(self, text="Movimiento")
        self.control_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        
        # Panel para controles de presets
        self.preset_frame = ttk.LabelFrame(self, text="Presets")
        self.preset_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # Panel para controles de zoom
        self.zoom_frame = ttk.LabelFrame(self, text="Zoom")
        self.zoom_frame.grid(row=1, column=2, padx=5, pady=5, sticky="nsew")
        
        # Configurar distribución de celdas
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self.rowconfigure(1, weight=1)
        
        # Crear controles de movimiento
        self._create_movement_controls()
        
        # Crear controles de presets
        self._create_preset_controls()
        
        # Crear controles de zoom
        self._create_zoom_controls()
        
        # Mensaje de estado
        self.status_label = ttk.Label(self, text="Listo")
        self.status_label.grid(row=2, column=0, columnspan=3, pady=5)
        
        # Deshabilitar controles si no hay PTZ
        if not self.ptz_available:
            self._disable_controls()
            self.status_label.config(text="Control PTZ no disponible")
    
    def _create_movement_controls(self):
        """Crear botones para controlar movimiento de la cámara"""
        # Botón arriba
        self.btn_up = ttk.Button(
            self.control_frame, 
            text="▲", 
            width=3,
            command=lambda: self._move_camera(0, 0.5, 0)
        )
        self.btn_up.grid(row=0, column=1, padx=2, pady=2)
        
        # Botón abajo
        self.btn_down = ttk.Button(
            self.control_frame, 
            text="▼", 
            width=3,
            command=lambda: self._move_camera(0, -0.5, 0)
        )
        self.btn_down.grid(row=2, column=1, padx=2, pady=2)
        
        # Botón izquierda
        self.btn_left = ttk.Button(
            self.control_frame, 
            text="◀", 
            width=3,
            command=lambda: self._move_camera(-0.5, 0, 0)
        )
        self.btn_left.grid(row=1, column=0, padx=2, pady=2)
        
        # Botón derecha
        self.btn_right = ttk.Button(
            self.control_frame, 
            text="▶", 
            width=3,
            command=lambda: self._move_camera(0.5, 0, 0)
        )
        self.btn_right.grid(row=1, column=2, padx=2, pady=2)
        
        # Botón centro (parar)
        self.btn_stop = ttk.Button(
            self.control_frame, 
            text="■", 
            width=3,
            command=self._stop_camera
        )
        self.btn_stop.grid(row=1, column=1, padx=2, pady=2)
        
        # Botón de inicio
        self.btn_home = ttk.Button(
            self.control_frame, 
            text="Inicio", 
            command=self._goto_home
        )
        self.btn_home.grid(row=3, column=0, columnspan=3, padx=2, pady=5)
    
    def _create_preset_controls(self):
        """Crear botones para controlar presets de la cámara"""
        # Espacio para botones de presets (se cargan dinámicamente)
        self.preset_buttons = []
        
        # Preset entrada
        self.btn_preset_entrada = ttk.Button(
            self.preset_frame, 
            text="Entrada", 
            command=lambda: self._goto_preset(1)
        )
        self.btn_preset_entrada.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        self.preset_buttons.append(self.btn_preset_entrada)
        
        # Preset salida
        self.btn_preset_salida = ttk.Button(
            self.preset_frame, 
            text="Salida", 
            command=lambda: self._goto_preset(2)
        )
        self.btn_preset_salida.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        self.preset_buttons.append(self.btn_preset_salida)
        
        # Preset general
        self.btn_preset_general = ttk.Button(
            self.preset_frame, 
            text="General", 
            command=lambda: self._goto_preset(3)
        )
        self.btn_preset_general.grid(row=2, column=0, padx=2, pady=2, sticky="ew")
        self.preset_buttons.append(self.btn_preset_general)
        
        # Espacio adicional para más presets
        for i in range(3, 6):
            btn = ttk.Button(
                self.preset_frame, 
                text=f"Preset {i+1}", 
                command=lambda idx=i+1: self._goto_preset(idx)
            )
            btn.grid(row=i, column=0, padx=2, pady=2, sticky="ew")
            self.preset_buttons.append(btn)
    
    def _create_zoom_controls(self):
        """Crear controles para zoom de la cámara"""
        # Botón zoom in
        self.btn_zoom_in = ttk.Button(
            self.zoom_frame, 
            text="Zoom +", 
            command=lambda: self._zoom_camera(0.5)
        )
        self.btn_zoom_in.grid(row=0, column=0, padx=2, pady=2, sticky="ew")
        
        # Botón zoom out
        self.btn_zoom_out = ttk.Button(
            self.zoom_frame, 
            text="Zoom -", 
            command=lambda: self._zoom_camera(-0.5)
        )
        self.btn_zoom_out.grid(row=1, column=0, padx=2, pady=2, sticky="ew")
        
        # Botón zoom stop
        self.btn_zoom_stop = ttk.Button(
            self.zoom_frame, 
            text="Detener Zoom", 
            command=self._stop_camera
        )
        self.btn_zoom_stop.grid(row=2, column=0, padx=2, pady=2, sticky="ew")
    
    def _load_presets(self):
        """Cargar presets disponibles en la cámara"""
        if not self.ptz_available:
            return
        
        try:
            # Obtener presets disponibles
            presets = self.camera.get_presets()
            
            # Actualizar botones con nombres de presets
            for i, preset in enumerate(presets):
                if i < len(self.preset_buttons):
                    # Actualizar texto del botón
                    if preset['name']:
                        self.preset_buttons[i].config(text=preset['name'])
                    else:
                        self.preset_buttons[i].config(text=f"Preset {preset['index']}")
                    
                    # Actualizar comando
                    self.preset_buttons[i].config(
                        command=lambda idx=preset['index']: self._goto_preset(idx)
                    )
                    
                    # Habilitar botón
                    self.preset_buttons[i].config(state=tk.NORMAL)
            
            # Deshabilitar botones sin preset asignado
            for i in range(len(presets), len(self.preset_buttons)):
                self.preset_buttons[i].config(state=tk.DISABLED)
                
        except Exception as e:
            logger.error(f"Error al cargar presets: {e}")
            self.status_label.config(text="Error al cargar presets")
    
    def _disable_controls(self):
        """Deshabilitar todos los controles"""
        # Deshabilitar botones de movimiento
        for btn in [self.btn_up, self.btn_down, self.btn_left, self.btn_right, self.btn_stop, self.btn_home]:
            btn.config(state=tk.DISABLED)
        
        # Deshabilitar botones de presets
        for btn in self.preset_buttons:
            btn.config(state=tk.DISABLED)
        
        # Deshabilitar botones de zoom
        for btn in [self.btn_zoom_in, self.btn_zoom_out, self.btn_zoom_stop]:
            btn.config(state=tk.DISABLED)
    
    def _move_camera(self, pan, tilt, zoom):
        """
        Mover la cámara
        
        Args:
            pan (float): Velocidad horizontal (-1.0 a 1.0)
            tilt (float): Velocidad vertical (-1.0 a 1.0)
            zoom (float): Velocidad de zoom (-1.0 a 1.0)
        """
        if not self.ptz_available or not self.camera:
            return
        
        try:
            # Ejecutar en hilo separado para no bloquear la UI
            threading.Thread(target=self._move_camera_thread, args=(pan, tilt, zoom)).start()
            
            # Actualizar estado
            direction = ""
            if pan > 0:
                direction = "derecha"
            elif pan < 0:
                direction = "izquierda"
            elif tilt > 0:
                direction = "arriba"
            elif tilt < 0:
                direction = "abajo"
                
            self.status_label.config(text=f"Moviendo hacia {direction}")
            
        except Exception as e:
            logger.error(f"Error al mover cámara: {e}")
            self.status_label.config(text="Error al mover cámara")
    
    def _move_camera_thread(self, pan, tilt, zoom):
        """Método que se ejecuta en hilo separado para mover la cámara"""
        try:
            self.camera.move_continuous(pan, tilt, zoom)
        except Exception as e:
            logger.error(f"Error en hilo de movimiento: {e}")
    
    def _stop_camera(self):
        """Detener movimiento de la cámara"""
        if not self.ptz_available or not self.camera:
            return
        
        try:
            # Ejecutar en hilo separado para no bloquear la UI
            threading.Thread(target=self._stop_camera_thread).start()
            
            # Actualizar estado
            self.status_label.config(text="Movimiento detenido")
            
        except Exception as e:
            logger.error(f"Error al detener cámara: {e}")
            self.status_label.config(text="Error al detener cámara")
    
    def _stop_camera_thread(self):
        """Método que se ejecuta en hilo separado para detener la cámara"""
        try:
            self.camera.stop_movement()
        except Exception as e:
            logger.error(f"Error en hilo de detención: {e}")
    
    def _goto_preset(self, preset_index):
        """
        Mover la cámara a un preset
        
        Args:
            preset_index (int): Índice del preset
        """
        if not self.ptz_available or not self.camera:
            return
        
        try:
            # Ejecutar en hilo separado para no bloquear la UI
            threading.Thread(target=self._goto_preset_thread, args=(preset_index,)).start()
            
            # Actualizar estado
            self.status_label.config(text=f"Moviendo a preset {preset_index}")
            
        except Exception as e:
            logger.error(f"Error al ir a preset {preset_index}: {e}")
            self.status_label.config(text=f"Error al ir a preset {preset_index}")
    
    def _goto_preset_thread(self, preset_index):
        """Método que se ejecuta en hilo separado para ir a un preset"""
        try:
            self.camera.go_to_preset(preset_index)
        except Exception as e:
            logger.error(f"Error en hilo de preset: {e}")
    
    def _goto_home(self):
        """Mover la cámara a la posición home"""
        if not self.ptz_available or not self.camera:
            return
        
        try:
            # Ejecutar en hilo separado para no bloquear la UI
            threading.Thread(target=self._goto_home_thread).start()
            
            # Actualizar estado
            self.status_label.config(text="Moviendo a posición inicial")
            
        except Exception as e:
            logger.error(f"Error al ir a posición home: {e}")
            self.status_label.config(text="Error al ir a posición inicial")
    
    def _goto_home_thread(self):
        """Método que se ejecuta en hilo separado para ir a posición home"""
        try:
            self.camera.go_to_home_position()
        except Exception as e:
            logger.error(f"Error en hilo de home: {e}")
    
    def _zoom_camera(self, zoom):
        """
        Controlar zoom de la cámara
        
        Args:
            zoom (float): Velocidad de zoom (-1.0 a 1.0)
        """
        if not self.ptz_available or not self.camera:
            return
        
        try:
            # Ejecutar en hilo separado para no bloquear la UI
            threading.Thread(target=self._move_camera_thread, args=(0, 0, zoom)).start()
            
            # Actualizar estado
            if zoom > 0:
                self.status_label.config(text="Zoom in")
            else:
                self.status_label.config(text="Zoom out")
            
        except Exception as e:
            logger.error(f"Error al controlar zoom: {e}")
            self.status_label.config(text="Error al controlar zoom")


# Para pruebas independientes
if __name__ == "__main__":
    import sys
    import os
    
    # Añadir directorio raíz al path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    # Importar módulo de cámara
    from src.utils.ptz_camera import PTZCamera
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Crear ventana de prueba
    root = tk.Tk()
    root.title("Prueba de Controles PTZ")
    root.geometry("800x600")
    
    # Crear instancia de cámara
    camera = PTZCamera(
        ip="192.168.1.100",  # Cambiar por IP de cámara real para pruebas
        username="admin",
        password="admin"
    )
    
    # Intentar iniciar cámara
    try:
        camera.start()
        print("Cámara iniciada correctamente")
    except Exception as e:
        print(f"Error al iniciar cámara: {e}")
    
    # Crear frame para controles PTZ
    ptz_frame = PTZControlFrame(root, camera)
    ptz_frame.pack(padx=10, pady=10, fill=tk.X)
    
    # Función para cerrar ventana
    def on_closing():
        if camera:
            camera.stop()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Iniciar bucle principal
    root.mainloop()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para gestión de cámaras PTZ IP mediante RTSP y ONVIF
"""
import os
import cv2
import time
import logging
import threading
import requests
import urllib.parse
from urllib.parse import urlparse
from datetime import datetime
import numpy as np
from requests.auth import HTTPDigestAuth, HTTPBasicAuth
from zeep import Client
from zeep.wsse.username import UsernameToken
import onvif
from onvif import ONVIFCamera

logger = logging.getLogger('parqueadero.camera')

class PTZCamera:
    """Clase para gestionar cámaras PTZ IP mediante RTSP y ONVIF"""
    
    def __init__(self, ip, port=80, username='admin', password='admin', rtsp_url=None):
        """
        Inicializa la cámara PTZ
        
        Args:
            ip (str): Dirección IP de la cámara
            port (int): Puerto HTTP de la cámara (normalmente 80)
            username (str): Nombre de usuario para autenticación
            password (str): Contraseña para autenticación
            rtsp_url (str): URL RTSP personalizada. Si es None, se genera a partir de los otros parámetros
        """
        self.ip = ip
        self.port = port
        self.username = username
        self.password = password
        
        # URL RTSP para streaming de video
        if rtsp_url is None:
            self.rtsp_url = f"rtsp://{username}:{password}@{ip}:554/Streaming/Channels/101"
        else:
            # Sustituir variables en URL
            self.rtsp_url = rtsp_url.format(
                ip=ip,
                username=urllib.parse.quote(username),
                password=urllib.parse.quote(password)
            )
        
        # Estado de la cámara
        self.connected = False
        self.ptz_available = False
        self.video_capture = None
        self.last_frame = None
        self.last_frame_time = 0
        
        # Mutex para sincronización
        self.frame_lock = threading.Lock()
        
        # Thread para capturar frames
        self.capture_thread = None
        self.running = False
        
        # Cliente ONVIF para control PTZ
        self.onvif_client = None
        self.ptz_service = None
        self.media_service = None
        self.imaging_service = None
        self.profile = None
        
        logger.info(f"Cámara PTZ inicializada con IP {ip}")
    
    def connect_onvif(self):
        """Conectar a cámara mediante protocolo ONVIF para control PTZ"""
        try:
            # Inicializar cliente ONVIF
            self.onvif_client = ONVIFCamera(
                self.ip,
                self.port,
                self.username,
                self.password
            )
            
            # Obtener servicios PTZ y Media
            self.ptz_service = self.onvif_client.create_ptz_service()
            self.media_service = self.onvif_client.create_media_service()
            
            # Intentar obtener servicio de Imaging
            try:
                self.imaging_service = self.onvif_client.create_imaging_service()
            except:
                logger.warning("El servicio de Imaging no está disponible en esta cámara")
                self.imaging_service = None
            
            # Obtener el perfil de medios (normalmente el primero)
            profiles = self.media_service.GetProfiles()
            if profiles:
                self.profile = profiles[0]
                logger.info(f"Perfil de medios seleccionado: {self.profile.Name}")
            else:
                logger.warning("No se encontraron perfiles de medios")
                return False
            
            # Verificar capacidades PTZ
            request = self.ptz_service.create_type('GetConfigurations')
            ptz_configs = self.ptz_service.GetConfigurations(request)
            
            if ptz_configs:
                self.ptz_available = True
                logger.info("Control PTZ disponible")
            else:
                logger.warning("Control PTZ no disponible para esta cámara")
                self.ptz_available = False
            
            return True
            
        except Exception as e:
            logger.error(f"Error al conectar con ONVIF: {e}")
            self.ptz_available = False
            return False
    
    def connect_rtsp(self):
        """Conectar al stream RTSP de la cámara"""
        try:
            # Para mejorar el rendimiento en Jetson, usar GStreamer
            if self._is_jetson():
                # Pipeline Gstreamer optimizado para Jetson
                gst_pipeline = (
                    f'rtspsrc location="{self.rtsp_url}" latency=0 ! '
                    f'rtph264depay ! h264parse ! omxh264dec ! '
                    f'videoconvert ! appsink max-buffers=1 drop=true'
                )
                self.video_capture = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
            else:
                # Inicializar captura de video con OpenCV estándar
                self.video_capture = cv2.VideoCapture(self.rtsp_url)
            
            # Verificar si se pudo conectar
            if self.video_capture.isOpened():
                logger.info(f"Conectado al stream RTSP: {self.rtsp_url}")
                
                # Configurar propiedades del video capture
                self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Capturar un frame para verificar que funciona
                ret, frame = self.video_capture.read()
                if ret:
                    with self.frame_lock:
                        self.last_frame = frame
                        self.last_frame_time = time.time()
                    
                    self.connected = True
                    return True
                else:
                    logger.error("No se pudo leer frame de la cámara")
                    self.video_capture.release()
                    self.video_capture = None
                    return False
            else:
                logger.error(f"No se pudo conectar al stream RTSP: {self.rtsp_url}")
                return False
                
        except Exception as e:
            logger.error(f"Error al conectar con RTSP: {e}")
            if self.video_capture:
                self.video_capture.release()
                self.video_capture = None
            return False
    
    def start(self):
        """Iniciar conexión con la cámara y captura de video"""
        # Conectar a ONVIF para control PTZ
        onvif_status = self.connect_onvif()
        if not onvif_status:
            logger.warning("No se pudo conectar a ONVIF. Control PTZ no disponible.")
        
        # Conectar a RTSP para video
        rtsp_status = self.connect_rtsp()
        if not rtsp_status:
            logger.error("No se pudo conectar al stream RTSP.")
            return False
        
        # Iniciar thread de captura
        self.running = True
        self.capture_thread = threading.Thread(target=self._capture_loop)
        self.capture_thread.daemon = True
        self.capture_thread.start()
        
        logger.info("Cámara PTZ iniciada correctamente")
        return True
    
    def stop(self):
        """Detener conexión con la cámara y liberar recursos"""
        # Detener thread de captura
        self.running = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
        
        # Liberar VideoCapture
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        # Reiniciar estado
        self.connected = False
        
        logger.info("Cámara PTZ detenida")
    
    def _capture_loop(self):
        """Loop en thread separado para capturar frames continuamente"""
        consecutive_errors = 0
        
        while self.running:
            try:
                if self.video_capture and self.video_capture.isOpened():
                    ret, frame = self.video_capture.read()
                    
                    if ret:
                        with self.frame_lock:
                            self.last_frame = frame
                            self.last_frame_time = time.time()
                        
                        consecutive_errors = 0
                    else:
                        consecutive_errors += 1
                        logger.warning(f"Error al capturar frame (intento {consecutive_errors})")
                        
                        # Si hay muchos errores seguidos, intentar reconectar
                        if consecutive_errors >= 5:
                            logger.error("Demasiados errores consecutivos. Reconectando...")
                            self.reconnect()
                            consecutive_errors = 0
                else:
                    # Si no hay VideoCapture o está cerrado, intentar reconectar
                    logger.warning("VideoCapture no disponible. Reconectando...")
                    self.reconnect()
                    consecutive_errors = 0
                
                # Pequeña espera para no saturar CPU
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                logger.error(f"Error en thread de captura: {e}")
                consecutive_errors += 1
                
                # Si hay muchos errores seguidos, intentar reconectar
                if consecutive_errors >= 5:
                    logger.error("Demasiados errores. Reconectando...")
                    self.reconnect()
                    consecutive_errors = 0
                
                time.sleep(1.0)  # Esperar más tiempo tras un error
    
    def reconnect(self):
        """Reintentar conexión con la cámara"""
        logger.info("Intentando reconectar con la cámara...")
        
        # Liberar recursos actuales
        if self.video_capture:
            self.video_capture.release()
            self.video_capture = None
        
        # Esperar un momento antes de reconectar
        time.sleep(2.0)
        
        # Reconectar RTSP
        rtsp_status = self.connect_rtsp()
        
        # Si falló RTSP, intentar también reconectar ONVIF
        if not rtsp_status or not self.ptz_available:
            onvif_status = self.connect_onvif()
            if onvif_status:
                logger.info("Reconexión ONVIF exitosa")
            else:
                logger.warning("Reconexión ONVIF fallida")
        
        if rtsp_status:
            logger.info("Reconexión RTSP exitosa")
            self.connected = True
        else:
            logger.error("Reconexión fallida")
            self.connected = False
    
    def update(self):
        """
        Actualizar estado de la cámara
        Llamar periódicamente desde el hilo principal
        """
        # Verificar si la conexión está activa
        if self.video_capture and not self.video_capture.isOpened():
            logger.warning("VideoCapture cerrado. Intentando reconectar...")
            self.reconnect()
        
        # Verificar si el último frame es reciente
        if self.last_frame_time > 0:
            time_since_last_frame = time.time() - self.last_frame_time
            
            # Si han pasado más de 5 segundos sin recibir frames, reconectar
            if time_since_last_frame > 5.0:
                logger.warning(f"No se han recibido frames en {time_since_last_frame:.1f} segundos. Reconectando...")
                self.reconnect()
    
    def get_frame(self):
        """
        Obtener el último frame capturado
        
        Returns:
            numpy.ndarray: Último frame capturado o None si no hay
        """
        with self.frame_lock:
            if self.last_frame is not None:
                # Devolver una copia para evitar problemas de concurrencia
                return self.last_frame.copy()
        return None
    
    def is_connected(self):
        """
        Verificar si la cámara está conectada
        
        Returns:
            bool: True si está conectada, False si no
        """
        return self.connected
    
    def has_ptz(self):
        """
        Verificar si la cámara tiene capacidades PTZ
        
        Returns:
            bool: True si tiene PTZ, False si no
        """
        return self.ptz_available
    
    def move_continuous(self, pan=0.0, tilt=0.0, zoom=0.0):
        """
        Mover la cámara continuamente en dirección específica
        
        Args:
            pan (float): Velocidad de movimiento horizontal (-1.0 a 1.0)
            tilt (float): Velocidad de movimiento vertical (-1.0 a 1.0)
            zoom (float): Velocidad de zoom (-1.0 a 1.0)
            
        Returns:
            bool: True si el comando se envió correctamente, False si no
        """
        if not self.ptz_available or not self.profile:
            logger.warning("PTZ no disponible")
            return False
        
        try:
            # Crear request para movimiento continuo
            request = self.ptz_service.create_type('ContinuousMove')
            request.ProfileToken = self.profile.token
            
            # Configurar velocidades
            request.Velocity = self.ptz_service.GetStatus({'ProfileToken': self.profile.token}).Position
            request.Velocity.PanTilt.x = float(pan)
            request.Velocity.PanTilt.y = float(tilt)
            request.Velocity.Zoom.x = float(zoom)
            
            # Enviar comando
            self.ptz_service.ContinuousMove(request)
            logger.debug(f"Movimiento continuo: pan={pan}, tilt={tilt}, zoom={zoom}")
            return True
            
        except Exception as e:
            logger.error(f"Error al mover cámara continuamente: {e}")
            return False
    
    def move_absolute(self, pan=0.0, tilt=0.0, zoom=0.0):
        """
        Mover la cámara a una posición absoluta
        
        Args:
            pan (float): Posición horizontal (-1.0 a 1.0)
            tilt (float): Posición vertical (-1.0 a 1.0)
            zoom (float): Posición de zoom (0.0 a 1.0)
            
        Returns:
            bool: True si el comando se envió correctamente, False si no
        """
        if not self.ptz_available or not self.profile:
            logger.warning("PTZ no disponible")
            return False
        
        try:
            # Crear request para movimiento absoluto
            request = self.ptz_service.create_type('AbsoluteMove')
            request.ProfileToken = self.profile.token
            
            # Configurar posiciones
            request.Position = self.ptz_service.create_type('PTZVector')
            request.Position.PanTilt = self.ptz_service.create_type('Vector2D')
            request.Position.Zoom = self.ptz_service.create_type('Vector1D')
            
            request.Position.PanTilt.x = float(pan)
            request.Position.PanTilt.y = float(tilt)
            request.Position.Zoom.x = float(zoom)
            
            # Enviar comando
            self.ptz_service.AbsoluteMove(request)
            logger.debug(f"Movimiento absoluto: pan={pan}, tilt={tilt}, zoom={zoom}")
            return True
            
        except Exception as e:
            logger.error(f"Error al mover cámara a posición absoluta: {e}")
            return False
    
    def move_relative(self, pan=0.0, tilt=0.0, zoom=0.0):
        """
        Mover la cámara relativamente a su posición actual
        
        Args:
            pan (float): Desplazamiento horizontal (-1.0 a 1.0)
            tilt (float): Desplazamiento vertical (-1.0 a 1.0)
            zoom (float): Desplazamiento de zoom (-1.0 a 1.0)
            
        Returns:
            bool: True si el comando se envió correctamente, False si no
        """
        if not self.ptz_available or not self.profile:
            logger.warning("PTZ no disponible")
            return False
        
        try:
            # Crear request para movimiento relativo
            request = self.ptz_service.create_type('RelativeMove')
            request.ProfileToken = self.profile.token
            
            # Configurar desplazamientos
            request.Translation = self.ptz_service.create_type('PTZVector')
            request.Translation.PanTilt = self.ptz_service.create_type('Vector2D')
            request.Translation.Zoom = self.ptz_service.create_type('Vector1D')
            
            request.Translation.PanTilt.x = float(pan)
            request.Translation.PanTilt.y = float(tilt)
            request.Translation.Zoom.x = float(zoom)
            
            # Enviar comando
            self.ptz_service.RelativeMove(request)
            logger.debug(f"Movimiento relativo: pan={pan}, tilt={tilt}, zoom={zoom}")
            return True
            
        except Exception as e:
            logger.error(f"Error al mover cámara relativamente: {e}")
            return False
    
    def stop_movement(self):
        """
        Detener cualquier movimiento de la cámara
        
        Returns:
            bool: True si el comando se envió correctamente, False si no
        """
        if not self.ptz_available or not self.profile:
            logger.warning("PTZ no disponible")
            return False
        
        try:
            # Crear request para detener movimiento
            request = self.ptz_service.create_type('Stop')
            request.ProfileToken = self.profile.token
            
            # Detener todos los movimientos (pan, tilt, zoom)
            request.PanTilt = True
            request.Zoom = True
            
            # Enviar comando
            self.ptz_service.Stop(request)
            logger.debug("Movimiento detenido")
            return True
            
        except Exception as e:
            logger.error(f"Error al detener movimiento: {e}")
            return False
    
    def go_to_preset(self, preset_token):
        """
        Mover la cámara a una posición predefinida
        
        Args:
            preset_token (int or str): Token o número del preset
            
        Returns:
            bool: True si el comando se envió correctamente, False si no
        """
        if not self.ptz_available or not self.profile:
            logger.warning("PTZ no disponible")
            return False
        
        try:
            # Obtener presets disponibles
            presets = self.ptz_service.GetPresets({'ProfileToken': self.profile.token})
            
            # Buscar preset por token o número (índice + 1)
            target_preset = None
            
            if isinstance(preset_token, int):
                # Buscar por número de preset (índice + 1)
                if 1 <= preset_token <= len(presets):
                    target_preset = presets[preset_token - 1]
                else:
                    logger.warning(f"Preset número {preset_token} no encontrado")
                    return False
            else:
                # Buscar por token (string)
                for preset in presets:
                    if preset.token == preset_token:
                        target_preset = preset
                        break
                
                if target_preset is None:
                    logger.warning(f"Preset con token {preset_token} no encontrado")
                    return False
            
            # Crear request para ir a preset
            request = self.ptz_service.create_type('GotoPreset')
            request.ProfileToken = self.profile.token
            request.PresetToken = target_preset.token
            
            # Enviar comando
            self.ptz_service.GotoPreset(request)
            logger.info(f"Moviendo a preset {preset_token}")
            return True
            
        except Exception as e:
            logger.error(f"Error al ir a preset: {e}")
            return False
    
    def get_presets(self):
        """
        Obtener lista de presets disponibles
        
        Returns:
            list: Lista de presets disponibles
        """
        if not self.ptz_available or not self.profile:
            logger.warning("PTZ no disponible")
            return []
        
        try:
            # Obtener presets disponibles
            presets = self.ptz_service.GetPresets({'ProfileToken': self.profile.token})
            
            # Convertir a formato más simple
            result = []
            for i, preset in enumerate(presets):
                result.append({
                    'index': i + 1,
                    'token': preset.token,
                    'name': preset.Name if hasattr(preset, 'Name') else f"Preset {i+1}"
                })
            
            return result
            
        except Exception as e:
            logger.error(f"Error al obtener presets: {e}")
            return []
    
    def set_home_position(self):
        """
        Establecer la posición actual como posición Home
        
        Returns:
            bool: True si el comando se envió correctamente, False si no
        """
        if not self.ptz_available or not self.profile:
            logger.warning("PTZ no disponible")
            return False
        
        try:
            request = self.ptz_service.create_type('SetHomePosition')
            request.ProfileToken = self.profile.token
            
            self.ptz_service.SetHomePosition(request)
            logger.info("Posición Home establecida")
            return True
            
        except Exception as e:
            logger.error(f"Error al establecer posición Home: {e}")
            return False
    
    def go_to_home_position(self):
        """
        Mover la cámara a la posición Home
        
        Returns:
            bool: True si el comando se envió correctamente, False si no
        """
        if not self.ptz_available or not self.profile:
            logger.warning("PTZ no disponible")
            return False
        
        try:
            request = self.ptz_service.create_type('GotoHomePosition')
            request.ProfileToken = self.profile.token
            
            self.ptz_service.GotoHomePosition(request)
            logger.info("Moviendo a posición Home")
            return True
            
        except Exception as e:
            logger.error(f"Error al ir a posición Home: {e}")
            return False
    
    def capture_image(self, save_path=None):
        """
        Capturar imagen actual y opcionalmente guardarla
        
        Args:
            save_path (str, optional): Ruta donde guardar la imagen
            
        Returns:
            numpy.ndarray: Imagen capturada o None si no se pudo capturar
        """
        frame = self.get_frame()
        
        if frame is not None and save_path:
            try:
                # Crear directorio si no existe
                os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
                
                # Guardar imagen
                cv2.imwrite(save_path, frame)
                logger.info(f"Imagen guardada en {save_path}")
            except Exception as e:
                logger.error(f"Error al guardar imagen: {e}")
        
        return frame
    
    def _is_jetson(self):
        """
        Detectar si estamos ejecutando en una Jetson
        
        Returns:
            bool: True si estamos en una Jetson, False si no
        """
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read()
                return 'NVIDIA Jetson' in model
        except:
            return False


# Para ejecutar pruebas directamente
if __name__ == "__main__":
    import argparse
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Prueba de cámara PTZ")
    parser.add_argument("--ip", default="192.168.1.100", help="Dirección IP de la cámara")
    parser.add_argument("--port", type=int, default=80, help="Puerto HTTP de la cámara")
    parser.add_argument("--user", default="admin", help="Nombre de usuario")
    parser.add_argument("--password", default="admin", help="Contraseña")
    parser.add_argument("--rtsp", help="URL RTSP personalizada")
    
    args = parser.parse_args()
    
    # Inicializar cámara PTZ
    camera = PTZCamera(
        ip=args.ip,
        port=args.port,
        username=args.user,
        password=args.password,
        rtsp_url=args.rtsp
    )
    
    # Iniciar cámara
    if camera.start():
        print("Cámara iniciada correctamente")
        
        # Mostrar presets disponibles
        if camera.has_ptz():
            presets = camera.get_presets()
            print(f"Presets disponibles: {len(presets)}")
            for preset in presets:
                print(f"  {preset['index']}: {preset['name']} (token: {preset['token']})")
        
        # Mostrar video
        print("Mostrando video de la cámara (ESC para salir, teclas 1-9 para ir a presets)")
        cv2.namedWindow("PTZ Camera", cv2.WINDOW_NORMAL)
        
        while True:
            frame = camera.get_frame()
            
            if frame is not None:
                cv2.imshow("PTZ Camera", frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # ESC para salir
            if key == 27:
                break
            
            # Movimiento con teclas de flecha
            elif key == ord('w'):  # Arriba
                camera.move_continuous(0, 0.3, 0)
            elif key == ord('s'):  # Abajo
                camera.move_continuous(0, -0.3, 0)
            elif key == ord('a'):  # Izquierda
                camera.move_continuous(-0.3, 0, 0)
            elif key == ord('d'):  # Derecha
                camera.move_continuous(0.3, 0, 0)
            elif key == ord('q'):  # Zoom in
                camera.move_continuous(0, 0, 0.3)
            elif key == ord('e'):  # Zoom out
                camera.move_continuous(0, 0, -0.3)
            elif key == ord(' '):  # Detener
                camera.stop_movement()
            
            # Ir a presets con teclas numéricas
            elif ord('1') <= key <= ord('9'):
                preset_num = key - ord('0')
                camera.go_to_preset(preset_num)
            
            # Capturar imagen con 'c'
            elif key == ord('c'):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                camera.capture_image(f"captura_{timestamp}.jpg")
                print(f"Imagen capturada: captura_{timestamp}.jpg")
        
        cv2.destroyAllWindows()
        
        # Detener cámara
        camera.stop()
        print("Cámara detenida")
    else:
        print("No se pudo iniciar la cámara")

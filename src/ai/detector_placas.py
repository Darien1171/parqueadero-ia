#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo avanzado para detección automática de placas vehiculares
utilizando NVIDIA DeepStream para aceleración por hardware
"""
import os
import sys
import gi
import logging
import numpy as np
import cv2
import time
import json
import pyds
import threading
from datetime import datetime

# Configurar paths necesarios para DeepStream
sys.path.append('/opt/nvidia/deepstream/deepstream/lib')

# Inicializar GStreamer
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GLib, GstRtspServer

# Inicializar GStreamer
Gst.init(None)

logger = logging.getLogger('parqueadero.detector_placas')

class DetectorPlacas:
    """
    Clase para detección automática de placas vehiculares con DeepStream
    """
    
    def __init__(self, modelo_detector=None, modelo_ocr=None):
        """
        Inicializa el detector con DeepStream
        
        Args:
            modelo_detector: Ruta al config file del modelo detector de placas (TensorRT)
            modelo_ocr: Ruta al config file del modelo OCR (TensorRT)
        """
        self.modelo_detector_path = modelo_detector
        self.modelo_ocr_path = modelo_ocr
        self.pipeline = None
        self.loop = None
        self.bus = None
        
        # Variables para comunicación entre hilos
        self.mutex = threading.Lock()
        self.detection_results = []
        self.frame_number = 0
        self.current_frame = None
        self.latest_results = None
        
        # Parámetros para validación de placas
        self.min_confidence = 0.7
        self.min_plate_chars = 5
        
        # Inicializar pipeline
        self._init_pipeline()
        
        logger.info("Detector de placas con DeepStream inicializado")
    
    def _init_pipeline(self):
        """Inicializar pipeline DeepStream"""
        try:
            # Verificar que las rutas de configuración existen
            if not os.path.exists(self.modelo_detector_path):
                logger.error(f"No se encontró el archivo de configuración: {self.modelo_detector_path}")
                self._setup_fallback_mode()
                return
            
            # Configurar pipeline básico para procesamiento de una imagen
            self.pipeline = Gst.Pipeline()
            
            # Crear elementos
            self.source = Gst.ElementFactory.make("appsrc", "source")
            convert1 = Gst.ElementFactory.make("videoconvert", "convert1")
            nvvidconv1 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv1")
            caps_filter = Gst.ElementFactory.make("capsfilter", "caps_filter")
            streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
            pgie = Gst.ElementFactory.make("nvinfer", "primary-nvinference-engine")
            tracker = Gst.ElementFactory.make("nvtracker", "tracker")
            nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv2")
            nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
            convert2 = Gst.ElementFactory.make("videoconvert", "convert2")
            self.sink = Gst.ElementFactory.make("appsink", "sink")
            
            # Verificar que se crearon correctamente
            if (not self.source or not convert1 or not nvvidconv1 or not caps_filter or
                not streammux or not pgie or not tracker or not nvvidconv2 or
                not nvosd or not convert2 or not self.sink):
                logger.error("No se pudieron crear todos los elementos del pipeline")
                self._setup_fallback_mode()
                return
            
            # Configurar appsrc
            self.source.set_property("is-live", True)
            self.source.set_property("format", Gst.Format.TIME)
            
            # Configurar capsfilter
            caps = Gst.Caps.from_string("video/x-raw,format=NV12")
            caps_filter.set_property("caps", caps)
            
            # Configurar streammux
            streammux.set_property("batch-size", 1)
            streammux.set_property("width", 1280)
            streammux.set_property("height", 720)
            streammux.set_property("batched-push-timeout", 4000000)
            streammux.set_property("live-source", 1)
            
            # Configurar detector primario (nvinfer)
            pgie.set_property("config-file-path", self.modelo_detector_path)
            
            # Configurar tracker
            config_file_path = "dstest2_tracker_config.txt"  # Ruta predeterminada
            if os.path.exists(config_file_path):
                tracker.set_property("ll-config-file", config_file_path)
            
            # Configurar appsink
            self.sink.set_property("emit-signals", True)
            self.sink.set_property("max-buffers", 1)
            self.sink.set_property("drop", True)
            self.sink.set_property("sync", False)
            
            # Agregar elementos al pipeline
            self.pipeline.add(self.source)
            self.pipeline.add(convert1)
            self.pipeline.add(nvvidconv1)
            self.pipeline.add(caps_filter)
            self.pipeline.add(streammux)
            self.pipeline.add(pgie)
            self.pipeline.add(tracker)
            self.pipeline.add(nvvidconv2)
            self.pipeline.add(nvosd)
            self.pipeline.add(convert2)
            self.pipeline.add(self.sink)
            
            # Enlazar elementos
            self.source.link(convert1)
            convert1.link(nvvidconv1)
            nvvidconv1.link(caps_filter)
            caps_filter.link(streammux)
            streammux.link(pgie)
            pgie.link(tracker)
            tracker.link(nvvidconv2)
            nvvidconv2.link(nvosd)
            nvosd.link(convert2)
            convert2.link(self.sink)
            
            # Configurar callbacks
            self.sink.connect("new-sample", self._on_new_sample)
            
            # Configurar bus de mensajes
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            self.bus.connect("message", self._on_bus_message)
            
            # Iniciar pipeline en modo PAUSED
            ret = self.pipeline.set_state(Gst.State.PAUSED)
            if ret == Gst.StateChangeReturn.SUCCESS:
                logger.info("Pipeline DeepStream en estado PAUSED")
            else:
                logger.error("No se pudo cambiar el estado del pipeline a PAUSED")
                self._setup_fallback_mode()
                return
            
            # Crear loop para procesamiento asíncrono
            self.loop = GLib.MainLoop()
            self.thread = threading.Thread(target=self.loop.run)
            self.thread.daemon = True
            self.thread.start()
            
            logger.info("Pipeline DeepStream inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error al inicializar DeepStream: {e}")
            # Configurar modo fallback si DeepStream no está disponible
            self._setup_fallback_mode()
    
    def _setup_fallback_mode(self):
        """Configurar modo de respaldo usando OpenCV"""
        logger.info("Configurando modo de respaldo con OpenCV")
        
        try:
            # Intentar cargar clasificador para placas
            cascade_paths = [
                os.path.join('src', 'ai', 'cascades', 'haarcascade_russian_plate_number.xml'),
                os.path.join(os.path.dirname(__file__), 'cascades', 'haarcascade_russian_plate_number.xml'),
                '/usr/share/opencv4/haarcascades/haarcascade_russian_plate_number.xml',
                cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            ]
            
            self.plate_cascade = None
            for path in cascade_paths:
                if os.path.exists(path):
                    self.plate_cascade = cv2.CascadeClassifier(path)
                    logger.info(f"Clasificador de placas cargado desde: {path}")
                    break
                    
            if self.plate_cascade is None:
                logger.warning("No se encontró clasificador para placas")
            
            # Verificar disponibilidad de Tesseract
            try:
                import pytesseract
                self.tesseract_available = True
            except ImportError:
                self.tesseract_available = False
                logger.warning("Tesseract OCR no está disponible")
                
        except Exception as e:
            logger.error(f"Error al configurar modo fallback: {e}")
            self.plate_cascade = None
            self.tesseract_available = False
    
    def _on_bus_message(self, bus, message):
        """Callback para mensajes del bus de GStreamer"""
        t = message.type
        if t == Gst.MessageType.EOS:
            logger.info("End-of-stream recibido")
            if self.loop and self.loop.is_running():
                self.loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Error en pipeline: {err.message}")
            if self.loop and self.loop.is_running():
                self.loop.quit()
    
    def _on_new_sample(self, sink):
        """Callback para recibir nuevos samples del appsink"""
        try:
            sample = sink.emit("pull-sample")
            if sample:
                buf = sample.get_buffer()
                caps = sample.get_caps()
                
                # Obtener metadatos del frame
                batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(buf))
                l_frame = batch_meta.frame_meta_list
                
                # Procesar metadatos
                while l_frame:
                    frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
                    frame_number = frame_meta.frame_num
                    
                    # Procesar objetos detectados
                    l_obj = frame_meta.obj_meta_list
                    plate_results = []
                    
                    while l_obj:
                        obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                        
                        # Si es una detección de placa
                        if obj_meta.class_id == 0:  # Asumiendo que class_id 0 es placa
                            confidence = obj_meta.confidence
                            
                            # Extraer texto de la placa (si está disponible)
                            # Esto dependerá de cómo esté configurado el modelo OCR
                            plate_text = ""
                            if obj_meta.text_params:
                                plate_text = obj_meta.text_params.display_text
                            
                            # Extraer coordenadas
                            left = obj_meta.rect_params.left
                            top = obj_meta.rect_params.top
                            width = obj_meta.rect_params.width
                            height = obj_meta.rect_params.height
                            
                            # Guardar resultado
                            plate_results.append({
                                'text': plate_text,
                                'confidence': confidence,
                                'coords': (left, top, width, height)
                            })
                        
                        # Siguiente objeto
                        l_obj = l_obj.next
                    
                    # Guardar resultados para este frame
                    with self.mutex:
                        self.detection_results = plate_results
                        self.frame_number = frame_number
                    
                    # Siguiente frame
                    l_frame = l_frame.next
                
                return Gst.FlowReturn.OK
            
        except Exception as e:
            logger.error(f"Error en callback de new-sample: {e}")
        
        return Gst.FlowReturn.ERROR
    
    def detectar_placa(self, imagen):
        """
        Detecta una placa en la imagen utilizando DeepStream
        
        Args:
            imagen (numpy.ndarray): Imagen BGR de OpenCV
            
        Returns:
            tuple: (texto_placa, confianza, imagen_placa)
        """
        start_time = time.time()
        
        # Verificar si la imagen es válida
        if imagen is None or imagen.size == 0:
            logger.warning("Imagen inválida recibida")
            return None, 0.0, None
        
        # Guardar imagen actual
        self.current_frame = imagen.copy()
        
        # Procesar con DeepStream o Fallback según disponibilidad
        if self.pipeline:
            resultado = self._procesar_con_deepstream(imagen)
        else:
            resultado = self._procesar_con_fallback(imagen)
        
        if resultado:
            texto, confianza, img_placa = resultado
            elapsed_time = time.time() - start_time
            logger.info(f"Placa detectada: {texto} (confianza: {confianza:.2f}, tiempo: {elapsed_time:.3f}s)")
            return texto, confianza, img_placa
        else:
            elapsed_time = time.time() - start_time
            logger.warning(f"No se detectó placa (tiempo: {elapsed_time:.3f}s)")
            return None, 0.0, None
    
    def _procesar_con_deepstream(self, imagen):
        """
        Procesa la imagen usando el pipeline DeepStream
        
        Args:
            imagen (numpy.ndarray): Imagen BGR de OpenCV
            
        Returns:
            tuple: (texto_placa, confianza, imagen_placa) o None si no se detectó
        """
        try:
            # Redimensionar imagen si es necesario
            height, width = imagen.shape[:2]
            
            # Crear buffer para enviar al appsrc
            size = width * height * 3  # BGR
            data = imagen.tobytes()
            
            # Crear GstBuffer
            buf = Gst.Buffer.new_allocate(None, size, None)
            buf.fill(0, data)
            
            # Configurar timestamp
            buf.pts = buf.dts = 0
            buf.duration = 1000000000  # 1 segundo en nanosegundos
            
            # Enviar buffer al appsrc
            ret = self.source.emit("push-buffer", buf)
            if ret != Gst.FlowReturn.OK:
                logger.error(f"Error al enviar buffer: {ret}")
                return None
            
            # Esperar un breve momento para que se procese el frame
            time.sleep(0.1)
            
            # Obtener resultados
            with self.mutex:
                results = self.detection_results.copy()
            
            # Procesar resultados
            if results:
                # Ordenar por confianza, de mayor a menor
                results.sort(key=lambda x: x['confidence'], reverse=True)
                
                # Tomar el resultado con mayor confianza
                mejor_resultado = results[0]
                texto = mejor_resultado['text']
                confianza = mejor_resultado['confidence']
                coords = mejor_resultado['coords']
                
                # Si no hay texto pero sí hay coordenadas, intentar OCR
                if not texto and coords:
                    left, top, width, height = coords
                    # Extraer región de interés (ROI)
                    roi = imagen[int(top):int(top+height), int(left):int(left+width)]
                    texto = self._realizar_ocr(roi)
                    
                # Normalizar formato de placa
                if texto:
                    texto = self._normalizar_placa(texto)
                
                # Extraer imagen de la placa
                if coords:
                    left, top, width, height = coords
                    imagen_placa = imagen[int(top):int(top+height), int(left):int(left+width)].copy()
                else:
                    imagen_placa = None
                
                return texto, confianza, imagen_placa
                
            return None
            
        except Exception as e:
            logger.error(f"Error en procesamiento DeepStream: {e}")
            return self._procesar_con_fallback(imagen)
    
    def _procesar_con_fallback(self, imagen):
        """
        Procesa la imagen usando OpenCV cuando DeepStream no está disponible
        
        Args:
            imagen (numpy.ndarray): Imagen BGR de OpenCV
            
        Returns:
            tuple: (texto_placa, confianza, imagen_placa) o None si no se detectó
        """
        try:
            # Verificar si el cascade classifier está disponible
            if self.plate_cascade is None:
                logger.warning("Clasificador de placas no disponible en modo fallback")
                return None
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
            # Detectar placas potenciales
            plates = self.plate_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            mejor_placa = None
            mejor_confianza = 0.0
            mejor_imagen = None
            
            for (x, y, w, h) in plates:
                # Extraer región de interés (ROI)
                roi = gray[y:y+h, x:x+w]
                
                # Preprocesar para OCR
                roi_procesado = self._preprocesar_para_ocr(roi)
                
                # Realizar OCR
                texto = self._realizar_ocr(roi_procesado)
                
                # Validar si parece una placa
                if texto and self._validar_formato_placa(texto):
                    # Calcular una confianza simulada
                    confianza = 0.8  # Valor fijo para modo fallback
                    
                    # Normalizar formato
                    texto_normalizado = self._normalizar_placa(texto)
                    
                    # Guardar si es mejor que el anterior
                    if confianza > mejor_confianza:
                        mejor_placa = texto_normalizado
                        mejor_confianza = confianza
                        mejor_imagen = imagen[y:y+h, x:x+w].copy()
            
            if mejor_placa:
                return mejor_placa, mejor_confianza, mejor_imagen
            
            return None
            
        except Exception as e:
            logger.error(f"Error en procesamiento fallback: {e}")
            return None
    
    def _preprocesar_para_ocr(self, imagen):
        """
        Preprocesa una imagen para mejorar OCR
        
        Args:
            imagen (numpy.ndarray): Imagen en escala de grises
            
        Returns:
            numpy.ndarray: Imagen preprocesada
        """
        try:
            # Verificar tamaño mínimo
            if imagen.shape[0] < 10 or imagen.shape[1] < 20:
                return imagen
            
            # Asegurarnos que está en escala de grises
            if len(imagen.shape) > 2:
                gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                gray = imagen.copy()
            
            # Redimensionar a altura estándar
            height, width = gray.shape
            target_height = 50
            scale_factor = target_height / height
            resized = cv2.resize(gray, (int(width * scale_factor), target_height))
            
            # Ecualización de histograma
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            equalized = clahe.apply(resized)
            
            # Filtro gaussiano
            blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
            
            # Umbralización Otsu
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            # Operaciones morfológicas para limpiar ruido
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            # Inversión para texto negro en fondo blanco (mejor para OCR)
            inverted = cv2.bitwise_not(morph)
            
            return inverted
            
        except Exception as e:
            logger.warning(f"Error en preprocesamiento: {e}")
            return imagen
    
    def _realizar_ocr(self, imagen):
        """
        Realiza OCR en la imagen de la placa
        
        Args:
            imagen (numpy.ndarray): Imagen preprocesada
            
        Returns:
            str: Texto reconocido o None si no se pudo reconocer
        """
        # Verificar si Tesseract está disponible
        if not self.tesseract_available:
            logger.warning("Tesseract no disponible para OCR")
            return None
            
        try:
            import pytesseract
            
            # Configuración optimizada para placas
            config = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            texto = pytesseract.image_to_string(imagen, config=config).strip()
            
            return texto
            
        except Exception as e:
            logger.warning(f"Error en OCR: {e}")
            return None
    
    def _validar_formato_placa(self, texto):
        """
        Validar si un texto tiene formato de placa vehicular
        
        Args:
            texto (str): Texto a validar
            
        Returns:
            bool: True si parece placa, False si no
        """
        if not texto:
            return False
        
        # Eliminar espacios y convertir a mayúsculas
        texto = texto.strip().upper()
        
        # Eliminar caracteres no alfanuméricos excepto guion
        texto_limpio = ''.join(c for c in texto if c.isalnum() or c == '-')
        
        # Verificar longitud mínima
        if len(texto_limpio) < self.min_plate_chars:
            return False
        
        # Verificar que tenga al menos una letra y un número
        tiene_letra = any(c.isalpha() for c in texto_limpio)
        tiene_numero = any(c.isdigit() for c in texto_limpio)
        
        # Verificar patrones comunes de placas
        # Patrón 1: 3 letras seguidas de 3 números (AAA-123, AAA123)
        patron1 = all(c.isalpha() for c in texto_limpio[:3]) and all(c.isdigit() for c in texto_limpio[-3:])
        
        # Patrón 2: Letra(s) al principio, números en medio, letra(s) al final
        patron2 = (texto_limpio[0].isalpha() and 
                  any(c.isdigit() for c in texto_limpio[1:-1]) and 
                  texto_limpio[-1].isalpha())
        
        # Patrón 3: Números al principio, letras al final
        patron3 = all(c.isdigit() for c in texto_limpio[:3]) and all(c.isalpha() for c in texto_limpio[-2:])
        
        return (tiene_letra and tiene_numero) and (patron1 or patron2 or patron3)
    
    def _normalizar_placa(self, texto):
        """
        Normalizar formato de placa
        
        Args:
            texto (str): Texto de placa
            
        Returns:
            str: Placa normalizada
        """
        if not texto:
            return None
        
        # Eliminar espacios y convertir a mayúsculas
        texto = texto.strip().upper()
        
        # Eliminar caracteres no alfanuméricos excepto guion
        texto = ''.join(c for c in texto if c.isalnum() or c == '-')
        
        # Formato estándar para placas colombianas: 3 letras, guion, 3 números (placas colombianas)
        if len(texto) == 6:
            # Si tiene 6 caracteres sin guion, insertar guion después de los primeros 3
            if texto[:3].isalpha() and texto[3:].isdigit():
                texto = texto[:3] + '-' + texto[3:]
        
        return texto
    
    def __del__(self):
        """Destructor para liberar recursos"""
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        if self.loop and self.loop.is_running():
            self.loop.quit()
            
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1.0)


# Para ejecutar pruebas directamente
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ruta a archivos de configuración para pruebas
    CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'config', 'deepstream')
    
    # Verificar argumentos de línea de comandos
    import argparse
    
    parser = argparse.ArgumentParser(description="Prueba de detector de placas con DeepStream")
    parser.add_argument("--imagen", help="Ruta a la imagen para analizar")
    parser.add_argument("--modelo", help="Ruta al archivo de configuración del modelo")
    
    args = parser.parse_args()
    
    # Buscar rutas por defecto si no se especifican
    if not args.modelo:
        # Buscar archivos de configuración
        modelo_path = os.path.join(CONFIG_DIR, "plate_detector_config.txt")
        
        if not os.path.exists(modelo_path):
            print(f"No se encontró el archivo de configuración: {modelo_path}")
            print("Por favor especifique la ruta con --modelo")
            sys.exit(1)
    else:
        modelo_path = args.modelo
    
    # Inicializar detector
    detector = DetectorPlacas(modelo_detector=modelo_path)
    
    # Procesar imagen
    if args.imagen:
        if not os.path.exists(args.imagen):
            print(f"No se encuentra la imagen: {args.imagen}")
            sys.exit(1)
        
        imagen = cv2.imread(args.imagen)
        if imagen is None:
            print(f"No se pudo cargar la imagen: {args.imagen}")
            sys.exit(1)
        
        print(f"Procesando imagen: {args.imagen}")
        
        # Detectar placa
        placa, confianza, imagen_placa = detector.detectar_placa(imagen)
        
        if placa:
            print(f"Placa detectada: {placa} (confianza: {confianza:.2f})")
            
            # Mostrar resultado
            result_img = imagen.copy()
            cv2.putText(result_img, f"Placa: {placa}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(result_img, f"Confianza: {confianza:.2f}", (10, 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Mostrar la imagen original y la placa recortada
            cv2.imshow("Imagen original", result_img)
            
            if imagen_placa is not None:
                cv2.imshow("Placa detectada", imagen_placa)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("No se detectó ninguna placa")
    else:
        print("No se especificó imagen. Usando webcam...")
        
        # Abrir webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("No se pudo abrir la webcam")
            sys.exit(1)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar frame")
                break
            
            # Detectar placa
            placa, confianza, imagen_placa = detector.detectar_placa(frame)
            
            # Mostrar resultado
            if placa:
                cv2.putText(frame, f"Placa: {placa}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Confianza: {confianza:.2f}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "No se detectó placa", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            cv2.imshow("Detector de placas con DeepStream", frame)
            
            # Mostrar placa recortada si se detectó
            if imagen_placa is not None:
                cv2.imshow("Placa detectada", imagen_placa)
            
            # Salir con ESC
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
        
        cap.release()
        cv2.destroyAllWindows()

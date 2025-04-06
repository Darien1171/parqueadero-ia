#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para detección y reconocimiento de placas vehiculares
utilizando modelos de deep learning optimizados con TensorRT
"""
import os
import cv2
import time
import logging
import numpy as np
from pathlib import Path
import pytesseract

logger = logging.getLogger('parqueadero.detector_placas')

class DetectorPlacas:
    """
    Clase para detectar y reconocer placas vehiculares
    utilizando modelos de deep learning optimizados con TensorRT
    """
    
    def __init__(self, modelo_detector, modelo_ocr):
        """
        Inicializa el detector de placas
        
        Args:
            modelo_detector (str): Ruta al modelo TensorRT para detección de placas
            modelo_ocr (str): Ruta al modelo TensorRT para OCR
        """
        self.modelo_detector = modelo_detector
        self.modelo_ocr = modelo_ocr
        
        # Inicializar modelos
        self.detector_loaded = False
        self.ocr_loaded = False
        self.detector = None
        self.ocr = None
        
        # Cargar modelos
        self._load_models()
        
        logger.info("Detector de placas inicializado")
    
    def _load_models(self):
        """Cargar modelos de detección y OCR"""
        try:
            # Verificar disponibilidad de TensorRT
            self._check_tensorrt()
            
            # Cargar modelo detector de placas
            if os.path.exists(self.modelo_detector):
                self.detector = self._load_tensorrt_model(self.modelo_detector)
                self.detector_loaded = True
                logger.info(f"Modelo detector de placas cargado desde {self.modelo_detector}")
            else:
                logger.warning(f"Modelo detector no encontrado en {self.modelo_detector}")
                self.detector_loaded = False
            
            # Cargar modelo OCR
            if os.path.exists(self.modelo_ocr):
                self.ocr = self._load_tensorrt_model(self.modelo_ocr)
                self.ocr_loaded = True
                logger.info(f"Modelo OCR cargado desde {self.modelo_ocr}")
            else:
                logger.warning(f"Modelo OCR no encontrado en {self.modelo_ocr}")
                self.ocr_loaded = False
            
        except Exception as e:
            logger.error(f"Error al cargar modelos: {e}")
            # Intentar cargar modo fallback si los modelos no están disponibles
            self._setup_fallback_mode()
    
    def _check_tensorrt(self):
        """Verificar si TensorRT está disponible"""
        try:
            import tensorrt as trt
            logger.info(f"TensorRT disponible: versión {trt.__version__}")
        except ImportError:
            logger.warning("TensorRT no está disponible en el sistema")
            raise ImportError("TensorRT no disponible")
    
    def _load_tensorrt_model(self, model_path):
        """
        Cargar modelo TensorRT
        
        Args:
            model_path (str): Ruta al archivo .trt
            
        Returns:
            object: Modelo TensorRT cargado
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            # Crear runtime TensorRT
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)
            
            # Cargar modelo
            with open(model_path, 'rb') as f:
                serialized_engine = f.read()
            
            engine = runtime.deserialize_cuda_engine(serialized_engine)
            context = engine.create_execution_context()
            
            # Crear buffers para entrada/salida
            input_shape = engine.get_binding_shape(0)
            output_shape = engine.get_binding_shape(1)
            
            # Crear modelo con información necesaria
            model = {
                'engine': engine,
                'context': context,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'bindings': [],
                'input_alloc': None,
                'output_alloc': None
            }
            
            # Configurar memoria para entrada/salida
            for binding in range(engine.num_bindings):
                size = trt.volume(engine.get_binding_shape(binding)) * \
                       engine.max_batch_size
                dtype = trt.nptype(engine.get_binding_dtype(binding))
                
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                # Append to model info
                model['bindings'].append(int(device_mem))
                
                if engine.binding_is_input(binding):
                    model['input_alloc'] = {'host': host_mem, 'device': device_mem}
                else:
                    model['output_alloc'] = {'host': host_mem, 'device': device_mem}
            
            return model
            
        except Exception as e:
            logger.error(f"Error al cargar modelo TensorRT: {e}")
            raise
    
    def _setup_fallback_mode(self):
        """Configurar modo fallback usando OpenCV y Tesseract"""
        logger.info("Configurando modo fallback con OpenCV y Tesseract")
        
        # Verificar instalación de Tesseract
        try:
            pytesseract.get_tesseract_version()
            logger.info("Tesseract instalado correctamente")
        except Exception as e:
            logger.error(f"Error con Tesseract: {e}")
        
        # Cargar clasificador Haar para detección de placas
        cascades_dir = Path(__file__).parent / 'cascades'
        cascade_file = cascades_dir / 'haarcascade_russian_plate_number.xml'
        
        if not cascade_file.exists():
            logger.warning(f"Archivo de cascada no encontrado en {cascade_file}")
            
            # Intentar cargar desde ubicación alternativa
            cascade_file = Path('/usr/share/opencv4/haarcascades/haarcascade_russian_plate_number.xml')
            if not cascade_file.exists():
                logger.error("No se pudo encontrar un clasificador Haar para placas")
                return
        
        try:
            self.plate_cascade = cv2.CascadeClassifier(str(cascade_file))
            logger.info(f"Clasificador Haar cargado desde {cascade_file}")
        except Exception as e:
            logger.error(f"Error al cargar clasificador Haar: {e}")
    
    def detectar_placa(self, imagen):
        """
        Detectar y reconocer placa en una imagen
        
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
        
        # Detectar placa
        if self.detector_loaded:
            # Usar modelo TensorRT
            regions = self._detect_plate_regions_tensorrt(imagen)
        else:
            # Usar modo fallback con OpenCV
            regions = self._detect_plate_regions_fallback(imagen)
        
        # Si no se detectaron placas
        if not regions:
            logger.debug("No se detectaron placas en la imagen")
            return None, 0.0, None
        
        # Ordenar regiones por confianza
        regions.sort(key=lambda x: x[1], reverse=True)
        
        # Tomar la región con mayor confianza
        best_region = regions[0]
        x1, y1, x2, y2 = best_region[0]
        confianza_deteccion = best_region[1]
        
        # Recortar la región de la placa
        img_placa = imagen[y1:y2, x1:x2].copy()
        
        # Reconocer texto
        if self.ocr_loaded:
            # Usar modelo OCR TensorRT
            texto_placa, confianza_ocr = self._recognize_plate_text_tensorrt(img_placa)
        else:
            # Usar Tesseract como fallback
            texto_placa, confianza_ocr = self._recognize_plate_text_fallback(img_placa)
        
        # Calcular confianza combinada
        if texto_placa:
            confianza = (confianza_deteccion + confianza_ocr) / 2
        else:
            confianza = 0.0
        
        elapsed_time = time.time() - start_time
        logger.debug(f"Tiempo de procesamiento: {elapsed_time:.3f}s, Texto: {texto_placa}, Confianza: {confianza:.2f}")
        
        return texto_placa, confianza, img_placa
    
    def _detect_plate_regions_tensorrt(self, imagen):
        """
        Detectar regiones de placas usando modelo TensorRT
        
        Args:
            imagen (numpy.ndarray): Imagen BGR de OpenCV
            
        Returns:
            list: Lista de tuplas (bbox, confianza)
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            
            # Redimensionar imagen para entrada del modelo
            input_shape = self.detector['input_shape']
            orig_height, orig_width = imagen.shape[:2]
            
            # Preprocesar imagen para el modelo
            input_height, input_width = input_shape[2], input_shape[3]
            preprocessed = cv2.resize(imagen, (input_width, input_height))
            preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            preprocessed = preprocessed.transpose((2, 0, 1))  # HWC -> CHW
            preprocessed = preprocessed.astype(np.float32) / 255.0
            preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
            
            # Copiar a memoria page-locked
            np.copyto(self.detector['input_alloc']['host'], preprocessed.ravel())
            
            # Transferir a GPU
            cuda.memcpy_htod(self.detector['input_alloc']['device'], 
                            self.detector['input_alloc']['host'])
            
            # Ejecutar inferencia
            self.detector['context'].execute_v2(self.detector['bindings'])
            
            # Transferir de GPU a CPU
            cuda.memcpy_dtoh(self.detector['output_alloc']['host'], 
                            self.detector['output_alloc']['device'])
            
            # Procesar salida
            output = self.detector['output_alloc']['host'].reshape(self.detector['output_shape'])
            
            # Interpretar salida (específico para YOLOv5)
            # El formato puede variar según el modelo, ajustar según sea necesario
            detections = self._process_yolo_output(output, (orig_height, orig_width), 
                                                threshold=0.5)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error en detección con TensorRT: {e}")
            return []
    
    def _process_yolo_output(self, output, original_shape, threshold=0.5):
        """
        Procesar salida del modelo YOLOv5
        
        Args:
            output (numpy.ndarray): Salida del modelo
            original_shape (tuple): (altura, ancho) original de la imagen
            threshold (float): Umbral de confianza
            
        Returns:
            list: Lista de tuplas (bbox, confianza)
        """
        # Este procesamiento es específico para la salida de YOLOv5
        # Convertir de coordenadas normalizadas a pixeles
        orig_height, orig_width = original_shape
        
        # Extraer predicciones
        predictions = output[0]  # Primera (y única) instancia del batch
        
        # Lista para almacenar las detecciones
        detections = []
        
        # Recorrer predicciones
        for pred in predictions:
            confidence = pred[4]  # Confianza de la detección
            
            # Filtrar detecciones por confianza
            if confidence < threshold:
                continue
            
            # Verificar si es una placa (clase 0 en este modelo)
            class_scores = pred[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id]
            
            if class_id != 0 or class_confidence < threshold:
                continue
            
            # Convertir coordenadas de centro/ancho/alto a esquinas
            x_center, y_center, width, height = pred[:4]
            
            # Convertir a coordenadas absolutas
            x1 = int((x_center - width/2) * orig_width)
            y1 = int((y_center - height/2) * orig_height)
            x2 = int((x_center + width/2) * orig_width)
            y2 = int((y_center + height/2) * orig_height)
            
            # Asegurar que las coordenadas estén dentro de la imagen
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(orig_width, x2)
            y2 = min(orig_height, y2)
            
            # Calcular confianza total
            total_confidence = confidence * class_confidence
            
            # Agregar a detecciones
            detections.append(((x1, y1, x2, y2), total_confidence))
        
        return detections
    
    def _detect_plate_regions_fallback(self, imagen):
        """
        Detectar regiones de placas usando OpenCV como fallback
        
        Args:
            imagen (numpy.ndarray): Imagen BGR de OpenCV
            
        Returns:
            list: Lista de tuplas (bbox, confianza)
        """
        try:
            # Convertir a escala de grises
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
            # Aplicar ecualización de histograma para mejorar contraste
            gray = cv2.equalizeHist(gray)
            
            # Detectar placas usando clasificador Haar
            if hasattr(self, 'plate_cascade'):
                plates = self.plate_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=5,
                    minSize=(60, 20),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                
                # Convertir a formato (x1, y1, x2, y2)
                detections = []
                for (x, y, w, h) in plates:
                    # Asignar una confianza fija (0.7) para detecciones con Haar
                    detections.append(((x, y, x+w, y+h), 0.7))
                
                return detections
            
            # Si el clasificador Haar no está disponible, intentar con detección de contornos
            # Aplicar umbral adaptativo
            thresh = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Buscar contornos
            contours, _ = cv2.findContours(
                thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Filtrar contornos por forma
            detections = []
            for contour in contours:
                # Obtener rectángulo
                x, y, w, h = cv2.boundingRect(contour)
                
                # Filtrar por tamaño y relación de aspecto
                aspect_ratio = w / float(h)
                area = w * h
                
                # Placas típicas tienen relación de aspecto entre 2 y 5
                # y ocupan una parte razonable de la imagen
                img_area = imagen.shape[0] * imagen.shape[1]
                area_ratio = area / img_area
                
                if 2.0 <= aspect_ratio <= 6.0 and 0.01 <= area_ratio <= 0.1:
                    # Asignar confianza basada en qué tan cerca está la relación de aspecto de 3:1
                    # (relación común para placas)
                    confidence = 1.0 - min(abs(3.0 - aspect_ratio) / 3.0, 0.5)
                    
                    # Agregar margen
                    margin_x = int(w * 0.05)
                    margin_y = int(h * 0.1)
                    
                    x1 = max(0, x - margin_x)
                    y1 = max(0, y - margin_y)
                    x2 = min(imagen.shape[1], x + w + margin_x)
                    y2 = min(imagen.shape[0], y + h + margin_y)
                    
                    detections.append(((x1, y1, x2, y2), confidence))
            
            return detections
            
        except Exception as e:
            logger.error(f"Error en detección fallback: {e}")
            return []
    
    def _recognize_plate_text_tensorrt(self, img_placa):
        """
        Reconocer texto de placa usando modelo OCR con TensorRT
        
        Args:
            img_placa (numpy.ndarray): Imagen recortada de la placa
            
        Returns:
            tuple: (texto_placa, confianza)
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            
            # Redimensionar imagen para entrada del modelo OCR
            input_shape = self.ocr['input_shape']
            
            # Preprocesar imagen para el modelo OCR
            input_height, input_width = input_shape[2], input_shape[3]
            preprocessed = cv2.resize(img_placa, (input_width, input_height))
            preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            preprocessed = preprocessed.transpose((2, 0, 1))  # HWC -> CHW
            preprocessed = preprocessed.astype(np.float32) / 255.0
            preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
            
            # Copiar a memoria page-locked
            np.copyto(self.ocr['input_alloc']['host'], preprocessed.ravel())
            
            # Transferir a GPU
            cuda.memcpy_htod(self.ocr['input_alloc']['device'], 
                            self.ocr['input_alloc']['host'])
            
            # Ejecutar inferencia
            self.ocr['context'].execute_v2(self.ocr['bindings'])
            
            # Transferir de GPU a CPU
            cuda.memcpy_dtoh(self.ocr['output_alloc']['host'], 
                            self.ocr['output_alloc']['device'])
            
            # Procesar salida
            output = self.ocr['output_alloc']['host'].reshape(self.ocr['output_shape'])
            
            # Decodificar texto 
            # El formato de salida depende del modelo OCR específico
            text, confidence = self._decode_ocr_output(output)
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"Error en OCR con TensorRT: {e}")
            return None, 0.0
    
    def _decode_ocr_output(self, output):
        """
        Decodificar salida del modelo OCR
        
        Args:
            output (numpy.ndarray): Salida del modelo OCR
            
        Returns:
            tuple: (texto, confianza)
        """
        # Este método debe adaptarse al formato de salida específico del modelo OCR
        # A continuación se muestra un ejemplo genérico
        
        # Ejemplo para un modelo que produce una secuencia de caracteres y scores
        char_map = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-."
        
        # Decodificar secuencia (depende del formato de salida del modelo)
        predictions = output[0]  # Primera (y única) instancia del batch
        
        # Suponiendo que cada columna es un caracter y las filas son las probabilidades
        char_indices = np.argmax(predictions, axis=1)
        confidences = np.max(predictions, axis=1)
        
        # Filtrar caracteres repetidos (CTC decoding simplified)
        text = ""
        prev_char = None
        avg_confidence = 0.0
        count = 0
        
        for idx, conf in zip(char_indices, confidences):
            if idx < len(char_map) and idx > 0:  # Ignorar índice 0 (blank)
                char = char_map[idx]
                if char != prev_char:  # Evitar repeticiones
                    text += char
                    avg_confidence += conf
                    count += 1
                prev_char = char
        
        # Calcular confianza promedio
        confidence = avg_confidence / max(1, count)
        
        # Limpiar el texto (formato específico para placas colombianas)
        text = self._format_plate_text(text)
        
        return text, confidence
    
    def _recognize_plate_text_fallback(self, img_placa):
        """
        Reconocer texto de placa usando Tesseract como fallback
        
        Args:
            img_placa (numpy.ndarray): Imagen recortada de la placa
            
        Returns:
            tuple: (texto_placa, confianza)
        """
        try:
            # Verificar tamaño mínimo
            if img_placa.shape[0] < 20 or img_placa.shape[1] < 50:
                logger.warning("Imagen de placa demasiado pequeña para OCR")
                return None, 0.0
            
            # Preprocesar imagen para mejorar OCR
            # Redimensionar para mejor OCR
            # Una altura de 50-60 pixeles suele dar buenos resultados con Tesseract
            scale_factor = 60.0 / img_placa.shape[0]
            width = int(img_placa.shape[1] * scale_factor)
            img_proc = cv2.resize(img_placa, (width, 60))
            
            # Convertir a escala de grises
            gray = cv2.cvtColor(img_proc, cv2.COLOR_BGR2GRAY)
            
            # Aplicar ecualización de histograma
            gray = cv2.equalizeHist(gray)
            
            # Aplicar filtro Gaussiano para reducir ruido
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Binarizar la imagen (umbral adaptativo)
            binary = cv2.adaptiveThreshold(
                blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 11, 2
            )
            
            # Dilatación para conectar componentes
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            dilated = cv2.dilate(binary, kernel, iterations=1)
            
            # Erosión para eliminar pequeños ruidos
            eroded = cv2.erode(dilated, kernel, iterations=1)
            
            # Inversión para texto negro sobre fondo blanco (mejor para Tesseract)
            inverted = cv2.bitwise_not(eroded)
            
            # Configuraciones para Tesseract
            config = f'--psm 7 --oem 1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
            
            # Realizar OCR con Tesseract
            text = pytesseract.image_to_string(inverted, config=config).strip()
            
            # Obtener datos adicionales para calcular confianza
            data = pytesseract.image_to_data(inverted, config=config, output_type=pytesseract.Output.DICT)
            
            # Calcular confianza promedio
            if 'conf' in data and len(data['conf']) > 0:
                # Filtrar valores -1 (que indican error)
                confidences = [c for c in data['conf'] if c > 0]
                confidence = sum(confidences) / max(1, len(confidences)) / 100.0
            else:
                confidence = 0.5  # Valor por defecto
            
            # Limpiar y formatear texto
            text = self._format_plate_text(text)
            
            return text, confidence
            
        except Exception as e:
            logger.error(f"Error en OCR fallback: {e}")
            return None, 0.0
    
    def _format_plate_text(self, text):
        """
        Formatear y limpiar texto de placa
        
        Args:
            text (str): Texto reconocido
            
        Returns:
            str: Texto formateado
        """
        if not text:
            return None
        
        # Eliminar espacios y caracteres no deseados
        text = ''.join(c for c in text if c.isalnum() or c == '-')
        
        # Convertir a mayúsculas
        text = text.upper()
        
        # Si está vacío después de limpieza
        if not text:
            return None
        
        # Formato típico de placas colombianas: AAA-123 o AAA123
        # Verificar longitud típica (6-7 caracteres)
        if 6 <= len(text) <= 7:
            # Si tiene exactamente 6 caracteres y no tiene guión, agregar guión
            if len(text) == 6 and '-' not in text:
                # Insertar guión después de los primeros 3 caracteres
                text = text[:3] + '-' + text[3:]
        
        return text


# Para ejecutar pruebas directamente
if __name__ == "__main__":
    import argparse
    import glob
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Prueba de detector de placas")
    parser.add_argument("--detector", help="Ruta al modelo detector de placas (TensorRT)")
    parser.add_argument("--ocr", help="Ruta al modelo OCR (TensorRT)")
    parser.add_argument("--image", help="Ruta a imagen para procesar")
    parser.add_argument("--dir", help="Directorio con imágenes para procesar")
    
    args = parser.parse_args()
    
    # Rutas por defecto si no se especifican
    if not args.detector:
        args.detector = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'modelos', 'ai_models', 'plate_detector.trt'
        )
    
    if not args.ocr:
        args.ocr = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'modelos', 'ai_models', 'plate_ocr.trt'
        )
    
    # Inicializar detector
    detector = DetectorPlacas(
        modelo_detector=args.detector,
        modelo_ocr=args.ocr
    )
    
    # Procesar imágenes
    if args.image:
        # Procesar una sola imagen
        print(f"Procesando imagen: {args.image}")
        img = cv2.imread(args.image)
        
        if img is None:
            print(f"No se pudo cargar la imagen {args.image}")
            exit(1)
        
        texto, confianza, img_placa = detector.detectar_placa(img)
        
        print(f"Texto detectado: {texto}")
        print(f"Confianza: {confianza:.2f}")
        
        # Mostrar imagen original y recorte
        cv2.imshow("Imagen Original", img)
        
        if img_placa is not None:
            cv2.imshow("Placa Detectada", img_placa)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif args.dir:
        # Procesar un directorio de imágenes
        img_files = []
        for ext in ['jpg', 'jpeg', 'png']:
            img_files.extend(glob.glob(os.path.join(args.dir, f"*.{ext}")))
        
        if not img_files:
            print(f"No se encontraron imágenes en {args.dir}")
            exit(1)
        
        print(f"Encontradas {len(img_files)} imágenes")
        
        for img_file in img_files:
            print(f"\nProcesando imagen: {img_file}")
            img = cv2.imread(img_file)
            
            if img is None:
                print(f"No se pudo cargar la imagen {img_file}")
                continue
            
            texto, confianza, img_placa = detector.detectar_placa(img)
            
            print(f"Texto detectado: {texto}")
            print(f"Confianza: {confianza:.2f}")
            
            # Mostrar imagen original y recorte
            cv2.imshow("Imagen Original", img)
            
            if img_placa is not None:
                cv2.imshow("Placa Detectada", img_placa)
            
            key = cv2.waitKey(0)
            if key == 27:  # ESC para salir
                break
        
        cv2.destroyAllWindows()
    
    else:
        # Si no se especificó imagen ni directorio, usar webcam
        print("No se especificó imagen ni directorio. Usando webcam...")
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("No se pudo abrir la webcam")
            exit(1)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error al capturar frame")
                break
            
            # Procesar frame
            texto, confianza, img_placa = detector.detectar_placa(frame)
            
            # Mostrar información en frame
            if texto:
                cv2.putText(
                    frame, f"{texto} ({confianza:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                )
            
            # Mostrar frame
            cv2.imshow("Detector de Placas", frame)
            
            if img_placa is not None:
                cv2.imshow("Placa Detectada", img_placa)
            
            # Salir con ESC
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()

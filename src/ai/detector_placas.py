#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo avanzado para detección automática de placas vehiculares
Implementa múltiples técnicas de procesamiento de imágenes para optimizar
la detección sin depender de TensorRT ni clasificadores específicos
"""
import os
import cv2
import time
import logging
import numpy as np
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

logger = logging.getLogger('parqueadero.detector_placas')

class DetectorPlacas:
    """
    Clase para detección automática de placas vehiculares
    Implementa múltiples algoritmos avanzados de visión por computadora
    """
    
    def __init__(self, modelo_detector=None, modelo_ocr=None):
        """
        Inicializa el detector avanzado
        
        Args:
            modelo_detector: No utilizado, para compatibilidad
            modelo_ocr: No utilizado, para compatibilidad
        """
        # Comprobar disponibilidad de Tesseract
        self.tesseract_available = TESSERACT_AVAILABLE
        if not self.tesseract_available:
            logger.warning("Tesseract OCR no está disponible. La capacidad de OCR será limitada.")
        
        # Cargar clasificadores de OpenCV
        self._cargar_clasificadores()
        
        # Configurar parámetros optimizados
        self._configurar_parametros()
        
        logger.info("Detector de placas avanzado inicializado.")
    
    def _cargar_clasificadores(self):
        """Cargar clasificadores disponibles en OpenCV"""
        # Intentar cargar clasificador para placas rusas (funciona bien para muchas placas)
        try:
            # Primero buscar en ruta relativa del proyecto
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
        except Exception as e:
            logger.warning(f"Error al cargar clasificador para placas: {e}")
            self.plate_cascade = None
        
        # Intentar cargar clasificador para vehículos
        try:
            vehicle_cascade_paths = [
                cv2.data.haarcascades + 'haarcascade_car.xml',
                '/usr/share/opencv4/haarcascades/haarcascade_car.xml'
            ]
            
            self.vehicle_cascade = None
            for path in vehicle_cascade_paths:
                if os.path.exists(path):
                    self.vehicle_cascade = cv2.CascadeClassifier(path)
                    logger.info(f"Clasificador de vehículos cargado desde: {path}")
                    break
                    
            if self.vehicle_cascade is None:
                logger.warning("No se encontró clasificador para vehículos")
        except Exception as e:
            logger.warning(f"Error al cargar clasificador para vehículos: {e}")
            self.vehicle_cascade = None
    
    def _configurar_parametros(self):
        """Configurar parámetros optimizados para detección"""
        # Parámetros para detección de bordes
        self.canny_threshold1 = 30
        self.canny_threshold2 = 200
        
        # Parámetros para filtrado de contornos
        self.min_aspect_ratio = 2.0
        self.max_aspect_ratio = 6.0
        self.min_area_ratio = 0.001
        self.max_area_ratio = 0.1
        
        # Parámetros para Tesseract
        self.tesseract_config = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
        
        # Parámetros para validación de placas
        self.min_plate_chars = 5
        self.min_confidence = 0.4
    
    def detectar_placa(self, imagen):
        """
        Detecta una placa en la imagen utilizando múltiples técnicas
        
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
        
        # Guardar copia original para visualización
        imagen_original = imagen.copy()
        
        # Crear múltiples versiones preprocesadas de la imagen
        imagenes_procesadas = self._generar_versiones_preprocesadas(imagen)
        
        # Lista para almacenar todos los resultados de detección
        todos_resultados = []
        
        # Método 1: Detección basada en contornos y formas
        resultados_contornos = self._detectar_por_contornos(imagenes_procesadas)
        todos_resultados.extend(resultados_contornos)
        
        # Método 2: Detección con cascade classifier si está disponible
        if self.plate_cascade is not None:
            resultados_cascade = self._detectar_con_cascade(imagenes_procesadas)
            todos_resultados.extend(resultados_cascade)
        
        # Método 3: Detección primero de vehículos y luego buscar placas
        if self.vehicle_cascade is not None:
            resultados_vehiculos = self._detectar_vehiculos_y_placas(imagenes_procesadas)
            todos_resultados.extend(resultados_vehiculos)
        
        # Ordenar resultados por confianza
        todos_resultados.sort(key=lambda x: x[1], reverse=True)
        
        # Evaluar resultados y elegir el mejor
        mejor_placa = None
        mejor_confianza = 0.0
        mejor_imagen_placa = None
        
        for placa, confianza, imagen_placa in todos_resultados:
            if confianza > mejor_confianza:
                mejor_placa = placa
                mejor_confianza = confianza
                mejor_imagen_placa = imagen_placa
        
        # Verificar si encontramos un resultado con suficiente confianza
        if mejor_placa and mejor_confianza >= self.min_confidence:
            elapsed_time = time.time() - start_time
            logger.info(f"Placa detectada: {mejor_placa} (confianza: {mejor_confianza:.2f}, tiempo: {elapsed_time:.3f}s)")
            return mejor_placa, mejor_confianza, mejor_imagen_placa
        else:
            elapsed_time = time.time() - start_time
            logger.warning(f"No se detectó placa con suficiente confianza (tiempo: {elapsed_time:.3f}s)")
            return None, 0.0, None
    
    def _generar_versiones_preprocesadas(self, imagen):
        """
        Genera múltiples versiones preprocesadas de la imagen
        para mejorar la detección
        
        Args:
            imagen (numpy.ndarray): Imagen original
            
        Returns:
            list: Lista de imágenes preprocesadas
        """
        resultado = []
        
        # Imagen original en escala de grises
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        resultado.append(('gris', gris))
        
        # Filtro bilateral para reducir ruido preservando bordes
        bilateral = cv2.bilateralFilter(gris, 11, 17, 17)
        resultado.append(('bilateral', bilateral))
        
        # Ecualización de histograma para mejorar contraste
        ecualizado = cv2.equalizeHist(gris)
        resultado.append(('ecualizado', ecualizado))
        
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gris)
        resultado.append(('clahe', clahe_img))
        
        # Sobel para detección de bordes horizontales (placas suelen tener bordes horizontales fuertes)
        sobelx = cv2.Sobel(bilateral, cv2.CV_8U, 1, 0, ksize=3)
        resultado.append(('sobelx', sobelx))
        
        # Umbralización adaptativa
        threshold1 = cv2.adaptiveThreshold(
            bilateral, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        resultado.append(('threshold1', threshold1))
        
        threshold2 = cv2.adaptiveThreshold(
            clahe_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        resultado.append(('threshold2', threshold2))
        
        # Combinación de transformaciones
        canny = cv2.Canny(bilateral, self.canny_threshold1, self.canny_threshold2)
        resultado.append(('canny', canny))
        
        return resultado
    
    def _detectar_por_contornos(self, imagenes_procesadas):
        """
        Detecta placas basándose en contornos y formas
        
        Args:
            imagenes_procesadas (list): Lista de imágenes preprocesadas
            
        Returns:
            list: Lista de tuplas (placa, confianza, imagen_placa)
        """
        resultados = []
        
        # Intentar diferentes versiones de imágenes
        for nombre, img in imagenes_procesadas:
            if nombre in ['canny', 'threshold1', 'threshold2', 'sobelx']:
                try:
                    # Encontrar contornos
                    contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Ordenar contornos por área, de mayor a menor
                    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:20]
                    
                    # Buscar contornos rectangulares que podrían ser placas
                    for contorno in contornos:
                        # Calcular el perímetro y aproximar el contorno
                        perimetro = cv2.arcLength(contorno, True)
                        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
                        
                        # Verificar si es aproximadamente rectangular (4 puntos)
                        # También aceptamos polígonos de 4-8 puntos ya que las placas
                        # a veces no tienen bordes perfectamente rectangulares
                        if 4 <= len(aprox) <= 8:
                            # Obtener rectángulo
                            rect = cv2.minAreaRect(contorno)
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)
                            
                            # Calcular la perspectiva y rectificar la imagen
                            width = int(rect[1][0])
                            height = int(rect[1][1])
                            
                            # Asegurar dimensiones mínimas y relación de aspecto adecuada
                            if width < 10 or height < 10:
                                continue
                                
                            # Calcular relación de aspecto
                            aspect_ratio = max(width, height) / min(width, height)
                            
                            # Placas típicamente tienen relación de aspecto entre 2:1a 5:1
                            if self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
                                # Calcular ROI
                                src_pts = box.astype("float32")
                                dst_pts = np.array([[0, height-1],
                                                   [0, 0],
                                                   [width-1, 0],
                                                   [width-1, height-1]], dtype="float32")
                                
                                # Si el ancho es menor que la altura, ajustar para que sea apaisado
                                if width < height:
                                    width, height = height, width
                                    dst_pts = np.array([[0, 0],
                                                       [width-1, 0],
                                                       [width-1, height-1],
                                                       [0, height-1]], dtype="float32")
                                
                                # Obtener matriz de transformación de perspectiva
                                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                                
                                # Aplicar transformación de perspectiva
                                img_original = imagenes_procesadas[0][1]  # Imagen en escala de grises
                                warped = cv2.warpPerspective(img_original, M, (width, height))
                                
                                # Procesar para OCR
                                placa_procesada = self._preprocesar_para_ocr(warped)
                                
                                # Reconocer texto si Tesseract está disponible
                                if self.tesseract_available:
                                    texto, confianza = self._reconocer_texto_tesseract(placa_procesada)
                                    
                                    # Validar texto como placa
                                    if texto and self._validar_formato_placa(texto) and confianza > 0.3:
                                        # Normalizar formato
                                        texto_normalizado = self._normalizar_placa(texto)
                                        
                                        # Factor de confianza adicional por la forma
                                        factor_forma = min(1.0, (1.0 / abs(aspect_ratio - 3.5)))
                                        confianza_ajustada = (confianza * 0.7) + (factor_forma * 0.3)
                                        
                                        # Almacenar resultado
                                        resultados.append((texto_normalizado, confianza_ajustada, warped))
                                    
                except Exception as e:
                    logger.warning(f"Error al procesar contornos en imagen {nombre}: {e}")
        
        return resultados
    
    def _detectar_con_cascade(self, imagenes_procesadas):
        """
        Detecta placas usando clasificador Haar cascade
        
        Args:
            imagenes_procesadas (list): Lista de imágenes preprocesadas
            
        Returns:
            list: Lista de tuplas (placa, confianza, imagen_placa)
        """
        resultados = []
        
        if self.plate_cascade is None:
            return resultados
        
        # Intentar diferentes versiones de imágenes
        for nombre, img in imagenes_procesadas:
            if nombre in ['gris', 'ecualizado', 'clahe']:
                try:
                    # Detectar placas con cascade classifier
                    placas = self.plate_cascade.detectMultiScale(
                        img, scaleFactor=1.1, minNeighbors=5,
                        minSize=(60, 20), maxSize=(300, 100)
                    )
                    
                    for (x, y, w, h) in placas:
                        # Extraer región
                        roi = img[y:y+h, x:x+w]
                        
                        # Procesar para OCR
                        placa_procesada = self._preprocesar_para_ocr(roi)
                        
                        # Reconocer texto si Tesseract está disponible
                        if self.tesseract_available:
                            texto, confianza = self._reconocer_texto_tesseract(placa_procesada)
                            
                            # Validar texto como placa
                            if texto and self._validar_formato_placa(texto) and confianza > 0.3:
                                # Normalizar formato
                                texto_normalizado = self._normalizar_placa(texto)
                                
                                # Confianza adicional por usar Haar cascade
                                confianza_ajustada = confianza * 1.1  # 10% extra de confianza
                                
                                # Almacenar resultado
                                resultados.append((texto_normalizado, confianza_ajustada, roi))
                        
                except Exception as e:
                    logger.warning(f"Error al procesar cascade en imagen {nombre}: {e}")
        
        return resultados
    
    def _detectar_vehiculos_y_placas(self, imagenes_procesadas):
        """
        Detecta vehículos primero y luego busca placas en ellos
        
        Args:
            imagenes_procesadas (list): Lista de imágenes preprocesadas
            
        Returns:
            list: Lista de tuplas (placa, confianza, imagen_placa)
        """
        resultados = []
        
        if self.vehicle_cascade is None:
            return resultados
        
        # Usar la imagen en escala de grises original
        for nombre, img in imagenes_procesadas:
            if nombre == 'gris':
                try:
                    # Detectar vehículos
                    vehiculos = self.vehicle_cascade.detectMultiScale(
                        img, scaleFactor=1.1, minNeighbors=4,
                        minSize=(100, 100)
                    )
                    
                    for (x, y, w, h) in vehiculos:
                        # La placa suele estar en la parte delantera/trasera
                        # Dividimos el vehículo en 3 secciones horizontales
                        # y nos enfocamos en la sección inferior y superior
                        
                        # Sección inferior (parte trasera típicamente)
                        y_bottom = y + int(h * 0.66)
                        height_bottom = h - int(h * 0.66)
                        roi_bottom = img[y_bottom:y+h, x:x+w]
                        
                        # Sección superior (parte delantera en algunas imágenes)
                        y_top = y
                        height_top = int(h * 0.33)
                        roi_top = img[y_top:y_top+height_top, x:x+w]
                        
                        # Procesar ambas secciones
                        for roi, seccion in [(roi_bottom, "inferior"), (roi_top, "superior")]:
                            # Verificar tamaño mínimo
                            if roi.shape[0] < 10 or roi.shape[1] < 20:
                                continue
                                
                            # Buscar rectángulos y contornos en esta región
                            try:
                                # Aplicar Canny para detectar bordes
                                roi_canny = cv2.Canny(roi, 50, 150)
                                
                                # Encontrar contornos
                                contornos, _ = cv2.findContours(roi_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                
                                # Ordenar contornos por área, de mayor a menor
                                contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:5]
                                
                                for contorno in contornos:
                                    # Obtener rectángulo
                                    x1, y1, w1, h1 = cv2.boundingRect(contorno)
                                    
                                    # Verificar proporciones
                                    if w1 < 30 or h1 < 10:
                                        continue
                                        
                                    aspect_ratio = w1 / float(h1)
                                    if 2.0 <= aspect_ratio <= 6.0:
                                        # Extraer región candidata
                                        if seccion == "inferior":
                                            roi_plate = roi[y1:y1+h1, x1:x1+w1]
                                        else:
                                            roi_plate = roi[y1:y1+h1, x1:x1+w1]
                                        
                                        # Procesar para OCR
                                        placa_procesada = self._preprocesar_para_ocr(roi_plate)
                                        
                                        # Reconocer texto si Tesseract está disponible
                                        if self.tesseract_available:
                                            texto, confianza = self._reconocer_texto_tesseract(placa_procesada)
                                            
                                            # Validar texto como placa
                                            if texto and self._validar_formato_placa(texto) and confianza > 0.3:
                                                # Normalizar formato
                                                texto_normalizado = self._normalizar_placa(texto)
                                                
                                                # Confianza ajustada
                                                confianza_ajustada = confianza * 0.9  # Penalización por ser indirecto
                                                
                                                # Almacenar resultado
                                                resultados.append((texto_normalizado, confianza_ajustada, roi_plate))
                            except Exception as e:
                                logger.debug(f"Error al procesar sección {seccion} de vehículo: {e}")
                        
                except Exception as e:
                    logger.warning(f"Error al procesar vehículos: {e}")
        
        return resultados
    
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
    
    def _reconocer_texto_tesseract(self, img):
        """
        Reconocer texto usando Tesseract OCR con parámetros optimizados
        
        Args:
            img (numpy.ndarray): Imagen preprocesada
            
        Returns:
            tuple: (texto, confianza)
        """
        if not self.tesseract_available:
            return None, 0.0
        
        try:
            # Realizar OCR con múltiples configuraciones
            resultados = []
            
            # Configuración 1: Modo de página 7 (línea de texto)
            config1 = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            texto1 = pytesseract.image_to_string(img, config=config1).strip()
            
            if texto1:
                # Obtener datos de confianza
                data = pytesseract.image_to_data(img, config=config1, output_type=pytesseract.Output.DICT)
                confidences = [conf for conf in data['conf'] if conf > 0]
                
                if confidences:
                    avg_confidence1 = sum(confidences) / len(confidences) / 100.0
                    resultados.append((texto1, avg_confidence1))
            
            # Configuración 2: Modo de página 8 (palabra)
            config2 = r'--oem 1 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            texto2 = pytesseract.image_to_string(img, config=config2).strip()
            
            if texto2:
                data = pytesseract.image_to_data(img, config=config2, output_type=pytesseract.Output.DICT)
                confidences = [conf for conf in data['conf'] if conf > 0]
                
                if confidences:
                    avg_confidence2 = sum(confidences) / len(confidences) / 100.0
                    resultados.append((texto2, avg_confidence2))
            
            # Seleccionar el mejor resultado
            if resultados:
                # Ordenar por confianza
                resultados.sort(key=lambda x: x[1], reverse=True)
                return resultados[0]  # (texto, confianza)
            
            return None, 0.0
            
        except Exception as e:
            logger.warning(f"Error en OCR: {e}")
            return None, 0.0
    
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

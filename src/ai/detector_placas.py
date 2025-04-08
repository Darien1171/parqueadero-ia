#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo avanzado para detección automática de placas vehiculares
Implementa múltiples técnicas de procesamiento de imágenes para optimizar
la detección sin depender de TensorRT ni clasificadores específicos

Mejoras:
- Uso de type hints para mejor documentación
- Configuración más flexible mediante dataclass
- Soporte para múltiples formatos de placas
- Mejoras en el preprocesamiento de imágenes
- Mejor manejo de errores y logging
- Modo de depuración con guardado de imágenes
- Opciones de procesamiento por lotes
"""
import os
import cv2
import time
import logging
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from pathlib import Path
import re

# Intentar importar Tesseract OCR
try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

# Configurar logger
logger = logging.getLogger('parqueadero.detector_placas')

@dataclass
class ConfigDetectorPlacas:
    """Configuración para el detector de placas vehiculares"""
    # Parámetros para detección de bordes
    canny_threshold1: int = 30
    canny_threshold2: int = 200
    
    # Parámetros para filtrado de contornos
    min_aspect_ratio: float = 2.0
    max_aspect_ratio: float = 6.0
    min_area_ratio: float = 0.001
    max_area_ratio: float = 0.1
    
    # Parámetros para Tesseract
    tesseract_config: str = r'--oem 1 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
    
    # Parámetros para validación de placas
    min_plate_chars: int = 5
    min_confidence: float = 0.4
    
    # Parámetros para cascade classifier
    cascade_scale_factor: float = 1.1
    cascade_min_neighbors: int = 5
    min_plate_size: Tuple[int, int] = (60, 20)
    max_plate_size: Tuple[int, int] = (300, 100)
    
    # Parámetros para detección de vehículos
    vehicle_scale_factor: float = 1.1
    vehicle_min_neighbors: int = 4
    min_vehicle_size: Tuple[int, int] = (100, 100)
    
    # Patrones conocidos de placas por país
    plate_patterns: Dict[str, List[Dict[str, Any]]] = field(default_factory=lambda: {
        'colombia': [
            {'regex': r'^[A-Z]{3}-?\d{3}$', 'descripcion': 'Placa colombiana estándar'},
            {'regex': r'^[A-Z]{3}\d{3}$', 'descripcion': 'Placa colombiana sin guion'}
        ],
        'internacional': [
            {'regex': r'^[A-Z0-9]{5,8}$', 'descripcion': 'Placa internacional genérica'},
            {'regex': r'^[A-Z]{1,3}[0-9]{3,4}[A-Z]{0,2}$', 'descripcion': 'Patrón común: letras-números-letras'}
        ]
    })
    
    # Opciones de depuración
    guardar_imagenes_debug: bool = False
    directorio_debug: str = 'debug_placas'


class DetectorPlacas:
    """
    Clase avanzada para detección automática de placas vehiculares
    Implementa múltiples algoritmos de visión por computadora para maximizar
    la precisión en la detección de placas bajo diversas condiciones.
    """
    
    def __init__(
        self, 
        modelo_detector=None, 
        modelo_ocr=None,
        config: Optional[ConfigDetectorPlacas] = None,
        ruta_cascade_placas: Optional[str] = None, 
        ruta_cascade_vehiculos: Optional[str] = None
    ):
        """
        Inicializa el detector avanzado de placas
        
        Args:
            modelo_detector: No utilizado, para compatibilidad con código anterior
            modelo_ocr: No utilizado, para compatibilidad con código anterior
            config: Configuración personalizada para el detector
            ruta_cascade_placas: Ruta explícita al archivo cascade para placas
            ruta_cascade_vehiculos: Ruta explícita al archivo cascade para vehículos
        """
        # Inicializar configuración
        self.config = config or ConfigDetectorPlacas()
        
        # Mantener compatibilidad con API anterior
        if modelo_detector is not None or modelo_ocr is not None:
            logger.info("Los parámetros modelo_detector y modelo_ocr están obsoletos pero se mantienen para compatibilidad")
        
        # Comprobar disponibilidad de Tesseract
        self.tesseract_available = TESSERACT_AVAILABLE
        if not self.tesseract_available:
            logger.warning("Tesseract OCR no está disponible. La capacidad de OCR será limitada.")
        
        # Crear directorio de depuración si está habilitado
        if self.config.guardar_imagenes_debug:
            os.makedirs(self.config.directorio_debug, exist_ok=True)
            logger.info(f"Las imágenes de depuración se guardarán en {self.config.directorio_debug}")
        
        # Cargar clasificadores de OpenCV
        self._cargar_clasificadores(ruta_cascade_placas, ruta_cascade_vehiculos)
        
        logger.info("Detector de placas avanzado inicializado correctamente.")
    
    def _cargar_clasificadores(self, ruta_cascade_placas: Optional[str] = None, ruta_cascade_vehiculos: Optional[str] = None) -> None:
        """
        Cargar clasificadores disponibles en OpenCV
        
        Args:
            ruta_cascade_placas: Ruta explícita al archivo cascade para placas
            ruta_cascade_vehiculos: Ruta explícita al archivo cascade para vehículos
        """
        # Cargar clasificador para placas
        try:
            if ruta_cascade_placas and os.path.exists(ruta_cascade_placas):
                self.plate_cascade = cv2.CascadeClassifier(ruta_cascade_placas)
                logger.info(f"Clasificador de placas cargado desde: {ruta_cascade_placas}")
            else:
                # Intentar múltiples ubicaciones potenciales
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
        
        # Cargar clasificador para vehículos
        try:
            if ruta_cascade_vehiculos and os.path.exists(ruta_cascade_vehiculos):
                self.vehicle_cascade = cv2.CascadeClassifier(ruta_cascade_vehiculos)
                logger.info(f"Clasificador de vehículos cargado desde: {ruta_cascade_vehiculos}")
            else:
                # Intentar múltiples ubicaciones potenciales
                vehicle_cascade_paths = [
                    cv2.data.haarcascades + 'haarcascade_car.xml',
                    '/usr/share/opencv4/haarcascades/haarcascade_car.xml',
                    os.path.join(os.path.dirname(__file__), 'cascades', 'haarcascade_car.xml')
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
    
    def detectar_placa(self, imagen: np.ndarray) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """
        Detecta una placa en la imagen utilizando múltiples técnicas
        
        Args:
            imagen: Imagen BGR de OpenCV
            
        Returns:
            Tupla con (texto_placa, confianza, imagen_placa)
            texto_placa: Texto de la placa detectada o None si no se encontró
            confianza: Nivel de confianza de la detección (0.0-1.0)
            imagen_placa: Imagen de la placa detectada o None si no se encontró
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
        resultados_contornos = self._detectar_por_contornos(imagenes_procesadas, imagen_original)
        todos_resultados.extend(resultados_contornos)
        
        # Método 2: Detección con cascade classifier si está disponible
        if self.plate_cascade is not None:
            resultados_cascade = self._detectar_con_cascade(imagenes_procesadas, imagen_original)
            todos_resultados.extend(resultados_cascade)
        
        # Método 3: Detección primero de vehículos y luego buscar placas
        if self.vehicle_cascade is not None:
            resultados_vehiculos = self._detectar_vehiculos_y_placas(imagenes_procesadas, imagen_original)
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
        if mejor_placa and mejor_confianza >= self.config.min_confidence:
            elapsed_time = time.time() - start_time
            logger.info(f"Placa detectada: {mejor_placa} (confianza: {mejor_confianza:.2f}, tiempo: {elapsed_time:.3f}s)")
            return mejor_placa, mejor_confianza, mejor_imagen_placa
        else:
            elapsed_time = time.time() - start_time
            logger.warning(f"No se detectó placa con suficiente confianza (tiempo: {elapsed_time:.3f}s)")
            return None, 0.0, None
    
    def _generar_versiones_preprocesadas(self, imagen: np.ndarray) -> List[Tuple[str, np.ndarray]]:
        """
        Genera múltiples versiones preprocesadas de la imagen para mejorar la detección
        
        Args:
            imagen: Imagen original
            
        Returns:
            Lista de tuplas con (nombre, imagen_procesada)
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
        canny = cv2.Canny(bilateral, self.config.canny_threshold1, self.config.canny_threshold2)
        resultado.append(('canny', canny))
        
        # Agregar detección de bordes Laplacian (nuevo)
        laplacian = cv2.Laplacian(bilateral, cv2.CV_8U)
        resultado.append(('laplacian', laplacian))
        
        # Agregar Sobel combinado (nuevo)
        sobely = cv2.Sobel(bilateral, cv2.CV_8U, 0, 1, ksize=3)
        sobel_combined = cv2.addWeighted(sobelx, 0.5, sobely, 0.5, 0)
        resultado.append(('sobel_combined', sobel_combined))
        
        # Guardar imágenes de depuración si está habilitado
        if self.config.guardar_imagenes_debug:
            for nombre, img in resultado:
                ruta_salida = os.path.join(self.config.directorio_debug, f"preprocesado_{nombre}.jpg")
                cv2.imwrite(ruta_salida, img)
        
        return resultado
    
    def _detectar_por_contornos(
        self, 
        imagenes_procesadas: List[Tuple[str, np.ndarray]], 
        imagen_original: np.ndarray
    ) -> List[Tuple[str, float, np.ndarray]]:
        """
        Detecta placas basándose en contornos y formas
        
        Args:
            imagenes_procesadas: Lista de imágenes preprocesadas
            imagen_original: Imagen original de entrada
            
        Returns:
            Lista de tuplas con (texto_placa, confianza, imagen_placa)
        """
        resultados = []
        
        # Intentar diferentes versiones de imágenes
        for nombre, img in imagenes_procesadas:
            if nombre in ['canny', 'threshold1', 'threshold2', 'sobelx', 'laplacian', 'sobel_combined']:
                try:
                    # Encontrar contornos
                    contornos, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    # Ordenar contornos por área, de mayor a menor
                    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)[:20]
                    
                    # Obtener imagen en escala de grises original para OCR
                    gris_original = imagenes_procesadas[0][1]  # Primera imagen procesada es escala de grises
                    
                    # Buscar contornos rectangulares que podrían ser placas
                    for i, contorno in enumerate(contornos):
                        # Calcular el perímetro y aproximar el contorno
                        perimetro = cv2.arcLength(contorno, True)
                        aprox = cv2.approxPolyDP(contorno, 0.02 * perimetro, True)
                        
                        # Verificar si es aproximadamente rectangular (4-8 puntos)
                        # También aceptamos polígonos de 4-8 puntos ya que las placas
                        # a veces no tienen bordes perfectamente rectangulares
                        if 4 <= len(aprox) <= 8:
                            # Obtener rectángulo
                            rect = cv2.minAreaRect(contorno)
                            box = cv2.boxPoints(rect)
                            box = np.int0(box)
                            
                            # Calcular dimensiones
                            width = int(rect[1][0])
                            height = int(rect[1][1])
                            
                            # Asegurar dimensiones mínimas
                            if width < 10 or height < 10:
                                continue
                                
                            # Calcular relación de aspecto
                            aspect_ratio = max(width, height) / min(width, height)
                            
                            # Placas típicamente tienen relación de aspecto entre 2:1 y 6:1
                            if self.config.min_aspect_ratio <= aspect_ratio <= self.config.max_aspect_ratio:
                                # Calcular región de interés
                                src_pts = box.astype("float32")
                                
                                # Asegurar que el ancho sea mayor que la altura (orientación apaisada)
                                if width < height:
                                    width, height = height, width
                                    dst_pts = np.array([[0, 0],
                                                      [width-1, 0],
                                                      [width-1, height-1],
                                                      [0, height-1]], dtype="float32")
                                else:
                                    dst_pts = np.array([[0, height-1],
                                                      [0, 0],
                                                      [width-1, 0],
                                                      [width-1, height-1]], dtype="float32")
                                
                                # Obtener matriz de transformación de perspectiva
                                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                                
                                # Aplicar transformación de perspectiva
                                warped = cv2.warpPerspective(gris_original, M, (width, height))
                                
                                # Guardar imagen de depuración si está habilitado
                                if self.config.guardar_imagenes_debug:
                                    debug_img = imagen_original.copy()
                                    cv2.drawContours(debug_img, [box], 0, (0, 255, 0), 2)
                                    debug_path = os.path.join(self.config.directorio_debug, 
                                                          f"contorno_{nombre}_{i}.jpg")
                                    cv2.imwrite(debug_path, debug_img)
                                    
                                    # Guardar candidato de placa transformado
                                    warped_path = os.path.join(self.config.directorio_debug, 
                                                           f"candidato_placa_{nombre}_{i}.jpg")
                                    cv2.imwrite(warped_path, warped)
                                
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
                                        
                                        # Guardar detección exitosa si depuración está habilitada
                                        if self.config.guardar_imagenes_debug:
                                            success_path = os.path.join(self.config.directorio_debug, 
                                                                     f"exito_{texto_normalizado}_{confianza_ajustada:.2f}.jpg")
                                            cv2.imwrite(success_path, placa_procesada)
                        
                except Exception as e:
                    logger.warning(f"Error al procesar contornos en imagen {nombre}: {e}")
        
        return resultados
    
    def _detectar_con_cascade(
        self, 
        imagenes_procesadas: List[Tuple[str, np.ndarray]], 
        imagen_original: np.ndarray
    ) -> List[Tuple[str, float, np.ndarray]]:
        """
        Detecta placas usando clasificador Haar cascade
        
        Args:
            imagenes_procesadas: Lista de imágenes preprocesadas
            imagen_original: Imagen original de entrada
            
        Returns:
            Lista de tuplas con (texto_placa, confianza, imagen_placa)
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
                        img, 
                        scaleFactor=self.config.cascade_scale_factor, 
                        minNeighbors=self.config.cascade_min_neighbors,
                        minSize=self.config.min_plate_size, 
                        maxSize=self.config.max_plate_size
                    )
                    
                    # Procesar cada placa detectada
                    for i, (x, y, w, h) in enumerate(placas):
                        # Extraer región
                        roi = img[y:y+h, x:x+w]
                        
                        # Guardar imagen de depuración si está habilitado
                        if self.config.guardar_imagenes_debug:
                            debug_img = imagen_original.copy()
                            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            debug_path = os.path.join(self.config.directorio_debug, 
                                                  f"cascade_{nombre}_{i}.jpg")
                            cv2.imwrite(debug_path, debug_img)
                        
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
                                
                                # Guardar detección exitosa si depuración está habilitada
                                if self.config.guardar_imagenes_debug:
                                    success_path = os.path.join(self.config.directorio_debug, 
                                                             f"cascade_exito_{texto_normalizado}_{confianza_ajustada:.2f}.jpg")
                                    cv2.imwrite(success_path, placa_procesada)
                        
                except Exception as e:
                    logger.warning(f"Error al procesar cascade en imagen {nombre}: {e}")
        
        return resultados
    
    def _detectar_vehiculos_y_placas(
        self, 
        imagenes_procesadas: List[Tuple[str, np.ndarray]], 
        imagen_original: np.ndarray
    ) -> List[Tuple[str, float, np.ndarray]]:
        """
        Detecta vehículos primero y luego busca placas en ellos
        
        Args:
            imagenes_procesadas: Lista de imágenes preprocesadas
            imagen_original: Imagen original de entrada
            
        Returns:
            Lista de tuplas con (texto_placa, confianza, imagen_placa)
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
                        img, 
                        scaleFactor=self.config.vehicle_scale_factor, 
                        minNeighbors=self.config.vehicle_min_neighbors,
                        minSize=self.config.min_vehicle_size
                    )
                    
                    # Procesar cada vehículo detectado
                    for i, (x, y, w, h) in enumerate(vehiculos):
                        # Guardar imagen de depuración si está habilitado
                        if self.config.guardar_imagenes_debug:
                            debug_img = imagen_original.copy()
                            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                            debug_path = os.path.join(self.config.directorio_debug, 
                                                  f"vehiculo_{i}.jpg")
                            cv2.imwrite(debug_path, debug_img)
                        
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
                                
                                for j, contorno in enumerate(contornos):
                                    # Obtener rectángulo
                                    x1, y1, w1, h1 = cv2.boundingRect(contorno)
                                    
                                    # Verificar proporciones
                                    if w1 < 30 or h1 < 10:
                                        continue
                                        
                                    aspect_ratio = w1 / float(h1)
                                    if 2.0 <= aspect_ratio <= 6.0:
                                        # Extraer región candidata
                                        roi_plate = roi[y1:y1+h1, x1:x1+w1]
                                        
                                        # Guardar imagen de depuración si está habilitado
                                        if self.config.guardar_imagenes_debug:
                                            vehicle_roi_path = os.path.join(self.config.directorio_debug, 
                                                                         f"vehiculo_{i}_{seccion}_{j}.jpg")
                                            if roi_plate.size > 0:
                                                cv2.imwrite(vehicle_roi_path, roi_plate)
                                        
                                        # Procesar para OCR
                                        placa_procesada = self._preprocesar_para_ocr(roi_plate)
                                        
                                        # Reconocer texto si Tesseract está disponible
                                        if self.tesseract_available:
                                            texto, confianza = self._reconocer_texto_tesseract(placa_procesada)
                                            
                                            # Validar texto como placa
                                            if texto and self._validar_formato_placa(texto) and confianza > 0.3:
                                                # Normalizar formato
                                                texto_normalizado = self._normalizar_placa(texto)
                                                
                                                # Confianza ajustada (penalización por detección indirecta)
                                                confianza_ajustada = confianza * 0.9
                                                
                                                # Almacenar resultado
                                                resultados.append((texto_normalizado, confianza_ajustada, roi_plate))
                                                
                                                # Guardar detección exitosa si depuración está habilitada
                                                if self.config.guardar_imagenes_debug:
                                                    success_path = os.path.join(self.config.directorio_debug, 
                                                                           f"vehiculo_exito_{texto_normalizado}_{confianza_ajustada:.2f}.jpg")
                                                    cv2.imwrite(success_path, placa_procesada)
                            except Exception as e:
                                logger.debug(f"Error al procesar sección {seccion} de vehículo: {e}")
                        
                except Exception as e:
                    logger.warning(f"Error al procesar vehículos: {e}")
        
        return resultados
    
    def _preprocesar_para_ocr(self, imagen: np.ndarray) -> np.ndarray:
        """
        Preprocesa una imagen para mejorar OCR
        
        Args:
            imagen: Imagen en escala de grises
            
        Returns:
            Imagen preprocesada
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
            
            # Aplicar CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            equalized = clahe.apply(resized)
            
            # Aplicar filtro gaussiano
            blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
            
            # Aplicar umbralización Otsu
            _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
            
            # Aplicar operaciones morfológicas para limpiar ruido
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
            morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
            
            # Inversión para texto negro en fondo blanco (mejor para OCR)
            inverted = cv2.bitwise_not(morph)
            
            # Guardar imagen de depuración si está habilitado
            if self.config.guardar_imagenes_debug:
                debug_path = os.path.join(self.config.directorio_debug, 
                                      f"ocr_prep_{hash(str(imagen.data.tobytes()))}.jpg")
                cv2.imwrite(debug_path, inverted)
            
            return inverted
            
        except Exception as e:
            logger.warning(f"Error en preprocesamiento: {e}")
            return imagen
    
    def _reconocer_texto_tesseract(self, img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Reconocer texto usando Tesseract OCR con parámetros optimizados
        
        Args:
            img: Imagen preprocesada
            
        Returns:
            Tupla con (texto, confianza)
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
            
            # Configuración 3: Modo de página 6 (bloque de texto)
            config3 = r'--oem 1 --psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-'
            texto3 = pytesseract.image_to_string(img, config=config3).strip()
            
            if texto3:
                data = pytesseract.image_to_data(img, config=config3, output_type=pytesseract.Output.DICT)
                confidences = [conf for conf in data['conf'] if conf > 0]
                
                if confidences:
                    avg_confidence3 = sum(confidences) / len(confidences) / 100.0
                    resultados.append((texto3, avg_confidence3))
            
            # Seleccionar el mejor resultado
            if resultados:
                # Ordenar por confianza
                resultados.sort(key=lambda x: x[1], reverse=True)
                return resultados[0]  # (texto, confianza)
            
            return None, 0.0
            
        except Exception as e:
            logger.warning(f"Error en OCR: {e}")
            return None, 0.0
    
    def _validar_formato_placa(self, texto: str) -> bool:
        """
        Validar si un texto tiene formato de placa vehicular
        
        Args:
            texto: Texto a validar
            
        Returns:
            True si parece placa, False si no
        """
        if not texto:
            return False
        
        # Eliminar espacios y convertir a mayúsculas
        texto = texto.strip().upper()
        
        # Eliminar caracteres no alfanuméricos excepto guion
        texto_limpio = ''.join(c for c in texto if c.isalnum() or c == '-')
        
        # Verificar longitud mínima
        if len(texto_limpio) < self.config.min_plate_chars:
            return False
        
        # Verificar que tenga al menos una letra y un número
        tiene_letra = any(c.isalpha() for c in texto_limpio)
        tiene_numero = any(c.isdigit() for c in texto_limpio)
        
        if not (tiene_letra and tiene_numero):
            return False
        
        # Verificar contra patrones conocidos
        for pais, patrones in self.config.plate_patterns.items():
            for patron in patrones:
                if re.match(patron['regex'], texto_limpio):
                    logger.debug(f"Coincide con patrón de placa: {patron['descripcion']} ({pais})")
                    return True
        
        # Verificar patrones comunes
        # Patrón 1: 3 letras seguidas de 3 números (AAA-123, AAA123)
        patron1 = all(c.isalpha() for c in texto_limpio[:3]) and all(c.isdigit() for c in texto_limpio[-3:])
        
        # Patrón 2: Letra(s) al principio, números en medio, letra(s) al final
        patron2 = (texto_limpio[0].isalpha() and 
                  any(c.isdigit() for c in texto_limpio[1:-1]) and 
                  texto_limpio[-1].isalpha())
        
        # Patrón 3: Números al principio, letras al final
        patron3 = all(c.isdigit() for c in texto_limpio[:3]) and all(c.isalpha() for c in texto_limpio[-2:])
        
        return patron1 or patron2 or patron3
    
    def _normalizar_placa(self, texto: str) -> Optional[str]:
        """
        Normalizar formato de placa
        
        Args:
            texto: Texto de placa
            
        Returns:
            Placa normalizada
        """
        if not texto:
            return None
        
        # Eliminar espacios y convertir a mayúsculas
        texto = texto.strip().upper()
        
        # Eliminar caracteres no alfanuméricos excepto guion
        texto = ''.join(c for c in texto if c.isalnum() or c == '-')
        
        # Formato estándar para placas colombianas: 3 letras, guion, 3 números
        if len(texto) == 6:
            # Si tiene 6 caracteres sin guion, insertar guion después de los primeros 3
            if texto[:3].isalpha() and texto[3:].isdigit():
                texto = texto[:3] + '-' + texto[3:]
        
        return texto
    
    def analizar_imagen(self, ruta_imagen: str) -> Tuple[Optional[str], float, Optional[np.ndarray]]:
        """
        Método de conveniencia para analizar un archivo de imagen
        
        Args:
            ruta_imagen: Ruta al archivo de imagen
            
        Returns:
            Tupla con (texto_placa, confianza, imagen_placa)
        """
        try:
            # Leer imagen
            imagen = cv2.imread(ruta_imagen)
            if imagen is None:
                logger.error(f"No se pudo leer la imagen: {ruta_imagen}")
                return None, 0.0, None
            
            # Detectar placa
            return self.detectar_placa(imagen)
            
        except Exception as e:
            logger.error(f"Error al analizar imagen {ruta_imagen}: {e}")
            return None, 0.0, None
    
    def procesar_lote(self, rutas_imagenes: List[str]) -> List[Tuple[str, Optional[str], float]]:
        """
        Procesar múltiples imágenes en lote
        
        Args:
            rutas_imagenes: Lista de rutas a archivos de imagen
            
        Returns:
            Lista de tuplas con (ruta_imagen, texto_placa, confianza)
        """
        resultados = []
        total = len(rutas_imagenes)
        
        for i, ruta in enumerate(rutas_imagenes):
            logger.info(f"Procesando imagen {i+1}/{total}: {ruta}")
            
            try:
                texto_placa, confianza, _ = self.analizar_imagen(ruta)
                resultados.append((ruta, texto_placa, confianza))
                
                logger.info(f"Resultado: {texto_placa if texto_placa else 'No se detectó placa'} "
                           f"(confianza: {confianza:.2f})")
                
            except Exception as e:
                logger.error(f"Error al procesar imagen {ruta}: {e}")
                resultados.append((ruta, None, 0.0))
        
        return resultados


# Ejemplo de uso cuando se ejecuta el módulo directamente
if __name__ == "__main__":
    import sys
    import argparse
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Analizar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Detector de Placas Vehiculares')
    parser.add_argument('-i', '--imagen', help='Ruta al archivo de imagen')
    parser.add_argument('-d', '--directorio', help='Ruta al directorio que contiene imágenes')
    parser.add_argument('--debug', action='store_true', help='Habilitar modo de depuración con guardado de imágenes')
    args = parser.parse_args()
    
    # Crear configuración
    config = ConfigDetectorPlacas()
    if args.debug:
        config.guardar_imagenes_debug = True
    
    # Crear detector
    detector = DetectorPlacas(config)
    
    if args.imagen:
        # Procesar una sola imagen
        texto_placa, confianza, imagen_placa = detector.analizar_imagen(args.imagen)
        
        if texto_placa:
            print(f"Placa detectada: {texto_placa} (confianza: {confianza:.2f})")
        else:
            print("No se detectó placa")
            
    elif args.directorio:
        # Procesar directorio de imágenes
        import glob
        rutas_imagenes = glob.glob(os.path.join(args.directorio, '*.jpg')) + \
                        glob.glob(os.path.join(args.directorio, '*.jpeg')) + \
                        glob.glob(os.path.join(args.directorio, '*.png'))
        
        resultados = detector.procesar_lote(rutas_imagenes)
        
        # Imprimir resumen
        print("\nResumen:")
        for ruta, texto_placa, confianza in resultados:
            resultado = f"{texto_placa} (confianza: {confianza:.2f})" if texto_placa else "No se detectó placa"
            print(f"{os.path.basename(ruta)}: {resultado}")
    else:
        parser.print_help()

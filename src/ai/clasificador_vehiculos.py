#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para clasificación de vehículos por tipo, marca y color
utilizando modelos de deep learning optimizados con TensorRT
"""
import os
import cv2
import time
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger('parqueadero.clasificador_vehiculos')

class ClasificadorVehiculos:
    """
    Clase para clasificar vehículos por tipo, marca y color
    utilizando modelos de deep learning optimizados con TensorRT
    """
    
    def __init__(self, modelo):
        """
        Inicializa el clasificador de vehículos
        
        Args:
            modelo (str): Ruta al modelo TensorRT para clasificación de vehículos
        """
        self.modelo_path = modelo
        
        # Clases para tipos de vehículo
        self.tipos_vehiculo = [
            'carro', 'moto', 'camión', 'bus', 'camioneta', 'van'
        ]
        
        # Clases para marcas comunes
        self.marcas_vehiculo = [
            'Chevrolet', 'Renault', 'Mazda', 'Nissan', 'Toyota', 'Kia', 
            'Hyundai', 'Ford', 'Volkswagen', 'Mercedes-Benz', 'BMW', 'Honda',
            'Suzuki', 'Yamaha', 'Bajaj', 'AKT', 'Otros'
        ]
        
        # Clases para colores comunes
        self.colores_vehiculo = [
            'blanco', 'negro', 'gris', 'rojo', 'azul', 'verde', 
            'amarillo', 'naranja', 'marrón', 'plata', 'otros'
        ]
        
        # Inicializar modelo
        self.modelo_loaded = False
        self.modelo = None
        
        # Cargar modelo
        self._load_model()
        
        logger.info("Clasificador de vehículos inicializado")
    
    def _load_model(self):
        """Cargar modelo de clasificación"""
        try:
            # Verificar disponibilidad de TensorRT
            self._check_tensorrt()
            
            # Cargar modelo TensorRT
            if os.path.exists(self.modelo_path):
                self.modelo = self._load_tensorrt_model(self.modelo_path)
                self.modelo_loaded = True
                logger.info(f"Modelo clasificador cargado desde {self.modelo_path}")
            else:
                logger.warning(f"Modelo clasificador no encontrado en {self.modelo_path}")
                self.modelo_loaded = False
            
        except Exception as e:
            logger.error(f"Error al cargar modelo: {e}")
            # Configurar modo fallback si el modelo no está disponible
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
            
            # Determinar cuántas salidas tiene el modelo
            num_outputs = engine.num_bindings - 1
            output_shapes = [engine.get_binding_shape(i) for i in range(1, engine.num_bindings)]
            
            # Crear modelo con información necesaria
            model = {
                'engine': engine,
                'context': context,
                'input_shape': input_shape,
                'output_shapes': output_shapes,
                'num_outputs': num_outputs,
                'bindings': [],
                'input_alloc': None,
                'output_allocs': []
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
                    model['output_allocs'].append({'host': host_mem, 'device': device_mem})
            
            return model
            
        except Exception as e:
            logger.error(f"Error al cargar modelo TensorRT: {e}")
            raise
    
    def _setup_fallback_mode(self):
        """Configurar modo fallback usando OpenCV para clasificación básica"""
        logger.info("Configurando modo fallback con OpenCV")
        
        # Cargar modelos pre-entrenados de OpenCV si están disponibles
        try:
            # Modelo para detección de vehículos
            self.vehicle_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_car.xml'
            )
            logger.info("Clasificador Haar para vehículos cargado")
        except Exception as e:
            logger.error(f"Error al cargar clasificador Haar: {e}")
            self.vehicle_cascade = None
    
    def clasificar(self, imagen):
        """
        Clasificar tipo, marca y color de vehículo en una imagen
        
        Args:
            imagen (numpy.ndarray): Imagen BGR de OpenCV
            
        Returns:
            tuple: (tipo, marca, color, confianzas)
        """
        start_time = time.time()
        
        # Verificar si la imagen es válida
        if imagen is None or imagen.size == 0:
            logger.warning("Imagen inválida recibida")
            return None, None, None, (0.0, 0.0, 0.0)
        
        # Clasificar vehículo
        if self.modelo_loaded:
            # Usar modelo TensorRT
            tipo, marca, color, confianzas = self._clasificar_tensorrt(imagen)
        else:
            # Usar modo fallback
            tipo, marca, color, confianzas = self._clasificar_fallback(imagen)
        
        elapsed_time = time.time() - start_time
        logger.debug(f"Tiempo de clasificación: {elapsed_time:.3f}s, Tipo: {tipo}, Marca: {marca}, Color: {color}")
        
        return tipo, marca, color, confianzas
    
    def _clasificar_tensorrt(self, imagen):
        """
        Clasificar vehículo usando modelo TensorRT
        
        Args:
            imagen (numpy.ndarray): Imagen BGR de OpenCV
            
        Returns:
            tuple: (tipo, marca, color, confianzas)
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            
            # Redimensionar imagen para entrada del modelo
            input_shape = self.modelo['input_shape']
            
            # Preprocesar imagen para el modelo
            input_height, input_width = input_shape[2], input_shape[3]
            preprocessed = cv2.resize(imagen, (input_width, input_height))
            preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            preprocessed = preprocessed.transpose((2, 0, 1))  # HWC -> CHW
            preprocessed = preprocessed.astype(np.float32) / 255.0
            preprocessed = np.expand_dims(preprocessed, axis=0)  # Add batch dimension
            
            # Copiar a memoria page-locked
            np.copyto(self.modelo['input_alloc']['host'], preprocessed.ravel())
            
            # Transferir a GPU
            cuda.memcpy_htod(self.modelo['input_alloc']['device'], 
                            self.modelo['input_alloc']['host'])
            
            # Ejecutar inferencia
            self.modelo['context'].execute_v2(self.modelo['bindings'])
            
            # Transferir resultados de GPU a CPU
            outputs = []
            for i in range(self.modelo['num_outputs']):
                cuda.memcpy_dtoh(self.modelo['output_allocs'][i]['host'], 
                                self.modelo['output_allocs'][i]['device'])
                
                # Reshape output according to its shape
                output = self.modelo['output_allocs'][i]['host'].reshape(
                    self.modelo['output_shapes'][i]
                )
                outputs.append(output)
            
            # Extraer predicciones
            # Asumiendo que el modelo tiene 3 salidas: tipo, marca, color
            if len(outputs) >= 3:
                # Para modelos multi-salida
                tipo_probs = outputs[0][0]  # Primera salida, primer batch
                marca_probs = outputs[1][0]  # Segunda salida, primer batch
                color_probs = outputs[2][0]  # Tercera salida, primer batch
                
                # Obtener índices de mayor probabilidad
                tipo_idx = np.argmax(tipo_probs)
                marca_idx = np.argmax(marca_probs)
                color_idx = np.argmax(color_probs)
                
                # Obtener confianzas
                conf_tipo = float(tipo_probs[tipo_idx])
                conf_marca = float(marca_probs[marca_idx])
                conf_color = float(color_probs[color_idx])
                
                # Convertir índices a etiquetas
                tipo = self.tipos_vehiculo[tipo_idx] if tipo_idx < len(self.tipos_vehiculo) else "desconocido"
                marca = self.marcas_vehiculo[marca_idx] if marca_idx < len(self.marcas_vehiculo) else "desconocido"
                color = self.colores_vehiculo[color_idx] if color_idx < len(self.colores_vehiculo) else "desconocido"
                
                return tipo, marca, color, (conf_tipo, conf_marca, conf_color)
            else:
                # Para modelos de salida única, asumimos que es una salida concatenada
                predictions = outputs[0][0]  # Primera salida, primer batch
                
                # Dividir predicciones según número de clases
                n_tipos = len(self.tipos_vehiculo)
                n_marcas = len(self.marcas_vehiculo)
                n_colores = len(self.colores_vehiculo)
                
                tipo_probs = predictions[:n_tipos]
                marca_probs = predictions[n_tipos:n_tipos+n_marcas]
                color_probs = predictions[n_tipos+n_marcas:n_tipos+n_marcas+n_colores]
                
                # Obtener índices de mayor probabilidad
                tipo_idx = np.argmax(tipo_probs)
                marca_idx = np.argmax(marca_probs)
                color_idx = np.argmax(color_probs)
                
                # Obtener confianzas
                conf_tipo = float(tipo_probs[tipo_idx])
                conf_marca = float(marca_probs[marca_idx])
                conf_color = float(color_probs[color_idx])
                
                # Convertir índices a etiquetas
                tipo = self.tipos_vehiculo[tipo_idx] if tipo_idx < len(self.tipos_vehiculo) else "desconocido"
                marca = self.marcas_vehiculo[marca_idx] if marca_idx < len(self.marcas_vehiculo) else "desconocido"
                color = self.colores_vehiculo[color_idx] if color_idx < len(self.colores_vehiculo) else "desconocido"
                
                return tipo, marca, color, (conf_tipo, conf_marca, conf_color)
            
        except Exception as e:
            logger.error(f"Error en clasificación con TensorRT: {e}")
            return self._clasificar_fallback(imagen)
    
    def _clasificar_fallback(self, imagen):
        """
        Clasificar vehículo usando OpenCV como fallback
        
        Args:
            imagen (numpy.ndarray): Imagen BGR de OpenCV
            
        Returns:
            tuple: (tipo, marca, color, confianzas)
        """
        try:
            # Determinar tipo de vehículo basado en forma y tamaño
            tipo = self._determinar_tipo_fallback(imagen)
            conf_tipo = 0.6  # Confianza moderada en el modo fallback
            
            # Determinar color
            color = self._determinar_color_fallback(imagen)
            conf_color = 0.7  # Confianza moderada en determinación de color
            
            # Para marca, indicamos "desconocido" ya que es difícil determinar sin ML
            marca = "desconocido"
            conf_marca = 0.0
            
            return tipo, marca, color, (conf_tipo, conf_marca, conf_color)
            
        except Exception as e:
            logger.error(f"Error en clasificación fallback: {e}")
            return "desconocido", "desconocido", "desconocido", (0.0, 0.0, 0.0)
    
    def _determinar_tipo_fallback(self, imagen):
        """
        Determinar tipo de vehículo usando características básicas
        
        Args:
            imagen (numpy.ndarray): Imagen BGR de OpenCV
            
        Returns:
            str: Tipo de vehículo estimado
        """
        # Si tenemos detector Haar disponible
        if hasattr(self, 'vehicle_cascade') and self.vehicle_cascade is not None:
            # Convertir a escala de grises
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            
            # Detectar vehículos
            vehicles = self.vehicle_cascade.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            if len(vehicles) > 0:
                # Si detecta un vehículo, asumimos que es un carro
                return "carro"
        
        # Análisis basado en relación de aspecto
        height, width = imagen.shape[:2]
        aspect_ratio = width / float(height)
        
        # Motos suelen ser más altas que anchas
        if aspect_ratio < 1.0:
            return "moto"
        # Vehículos grandes como buses y camiones suelen ser más anchos
        elif aspect_ratio > 2.2:
            return "bus" if height > width * 0.5 else "camión"
        # Carros normales tienen relación de aspecto entre 1.5-2.0 aprox.
        else:
            return "carro"
    
    def _determinar_color_fallback(self, imagen):
        """
        Determinar color predominante del vehículo
        
        Args:
            imagen (numpy.ndarray): Imagen BGR de OpenCV
            
        Returns:
            str: Color predominante
        """
        # Convertir a HSV que es mejor para análisis de color
        hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)
        
        # Crear máscara para eliminar fondo
        lower = np.array([0, 30, 30])
        upper = np.array([180, 255, 255])
        mask = cv2.inRange(hsv, lower, upper)
        
        # Aplicar máscara
        masked_img = cv2.bitwise_and(hsv, hsv, mask=mask)
        
        # Obtener histograma de Hue
        hist = cv2.calcHist([masked_img], [0], mask, [18], [0, 180])
        
        # Normalizar histograma
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        
        # Obtener índice de mayor frecuencia
        max_idx = np.argmax(hist)
        
        # Mapear índice a color
        # Dividimos el rango Hue (0-180) en 18 bins
        colors = {
            0: "rojo",      # 0-10
            1: "naranja",   # 10-20
            2: "amarillo",  # 20-30
            3: "verde",     # 30-40
            4: "verde",     # 40-50
            5: "verde",     # 50-60
            6: "verde",     # 60-70
            7: "verde",     # 70-80
            8: "azul",      # 80-90
            9: "azul",      # 90-100
            10: "azul",     # 100-110
            11: "azul",     # 110-120
            12: "azul",     # 120-130
            13: "morado",   # 130-140
            14: "morado",   # 140-150
            15: "morado",   # 150-160
            16: "rojo",     # 160-170
            17: "rojo"      # 170-180
        }
        
        # Verificar si es un color acromático (blanco, negro, gris)
        # Obtener histograma de Value y Saturation
        hist_s = cv2.calcHist([masked_img], [1], mask, [256], [0, 256])
        hist_v = cv2.calcHist([masked_img], [2], mask, [256], [0, 256])
        
        # Calcular mediana de S y V
        s_values = np.where(hist_s > 0)[0]
        v_values = np.where(hist_v > 0)[0]
        
        if len(s_values) > 0 and len(v_values) > 0:
            median_s = np.median(s_values)
            median_v = np.median(v_values)
            
            # Si saturación baja, es un color acromático
            if median_s < 50:
                if median_v < 80:
                    return "negro"
                elif median_v > 200:
                    return "blanco"
                else:
                    return "gris"
            
            # Si saturación media-baja y valor alto, podría ser plata
            if median_s < 80 and median_v > 180:
                return "plata"
        
        # Devolver color cromático según histograma de Hue
        return colors.get(max_idx, "desconocido")


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
    parser = argparse.ArgumentParser(description="Prueba de clasificador de vehículos")
    parser.add_argument("--model", help="Ruta al modelo clasificador (TensorRT)")
    parser.add_argument("--image", help="Ruta a imagen para procesar")
    parser.add_argument("--dir", help="Directorio con imágenes para procesar")
    
    args = parser.parse_args()
    
    # Ruta por defecto si no se especifica
    if not args.model:
        args.model = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'modelos', 'ai_models', 'vehicle_classifier.trt'
        )
    
    # Inicializar clasificador
    clasificador = ClasificadorVehiculos(modelo=args.model)
    
    # Procesar imágenes
    if args.image:
        # Procesar una sola imagen
        print(f"Procesando imagen: {args.image}")
        img = cv2.imread(args.image)
        
        if img is None:
            print(f"No se pudo cargar la imagen {args.image}")
            exit(1)
        
        tipo, marca, color, confianzas = clasificador.clasificar(img)
        
        print(f"Tipo de vehículo: {tipo} (confianza: {confianzas[0]:.2f})")
        print(f"Marca: {marca} (confianza: {confianzas[1]:.2f})")
        print(f"Color: {color} (confianza: {confianzas[2]:.2f})")
        
        # Mostrar imagen con resultados
        result_img = img.copy()
        cv2.putText(result_img, f"Tipo: {tipo}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result_img, f"Marca: {marca}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result_img, f"Color: {color}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        cv2.imshow("Clasificación de Vehículo", result_img)
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
            
            tipo, marca, color, confianzas = clasificador.clasificar(img)
            
            print(f"Tipo de vehículo: {tipo} (confianza: {confianzas[0]:.2f})")
            print(f"Marca: {marca} (confianza: {confianzas[1]:.2f})")
            print(f"Color: {color} (confianza: {confianzas[2]:.2f})")
            
            # Mostrar imagen con resultados
            result_img = img.copy()
            cv2.putText(result_img, f"Tipo: {tipo}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(result_img, f"Marca: {marca}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(result_img, f"Color: {color}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow("Clasificación de Vehículo", result_img)
            
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
            tipo, marca, color, confianzas = clasificador.clasificar(frame)
            
            # Mostrar información en frame
            cv2.putText(frame, f"Tipo: {tipo} ({confianzas[0]:.2f})", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Marca: {marca} ({confianzas[1]:.2f})", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Color: {color} ({confianzas[2]:.2f})", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Mostrar frame
            cv2.imshow("Clasificador de Vehículos", frame)
            
            # Salir con ESC
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break
        
        cap.release()
        cv2.destroyAllWindows()

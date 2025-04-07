#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para probar el detector de placas con DeepStream
"""
import os
import sys
import argparse
import cv2
import time
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger('test_detector')

def main():
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description="Prueba del detector de placas con DeepStream")
    parser.add_argument("--imagen", help="Ruta a imagen para prueba")
    parser.add_argument("--dir", help="Directorio con imágenes para prueba")
    parser.add_argument("--video", help="Ruta a video para prueba")
    parser.add_argument("--camara", type=int, default=-1, help="Índice de cámara (predeterminado: -1 = no usar)")
    parser.add_argument("--config", help="Ruta al directorio de configuración")
    
    args = parser.parse_args()
    
    # Determinar ruta base del proyecto
    if args.config:
        BASE_DIR = args.config
    else:
        # Intentar detectar automáticamente
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if os.path.exists(os.path.join(script_dir, "config", "deepstream")):
            BASE_DIR = script_dir
        else:
            BASE_DIR = os.path.dirname(script_dir)
    
    # Rutas a archivos de configuración
    CONFIG_DIR = os.path.join(BASE_DIR, "config", "deepstream")
    MODEL_DIR = os.path.join(BASE_DIR, "models", "ai_models")
    
    # Verificar que existan los directorios
    if not os.path.exists(CONFIG_DIR):
        logger.error(f"No se encontró el directorio de configuración: {CONFIG_DIR}")
        sys.exit(1)
    
    if not os.path.exists(MODEL_DIR):
        logger.error(f"No se encontró el directorio de modelos: {MODEL_DIR}")
        sys.exit(1)
    
    # Rutas a archivos de configuración
    detector_config = os.path.join(CONFIG_DIR, "plate_detector_config.txt")
    
    # Verificar que existan los archivos de configuración
    if not os.path.exists(detector_config):
        logger.error(f"No se encontró el archivo de configuración: {detector_config}")
        sys.exit(1)
    
    # Añadir directorio src al path para importar módulos
    sys.path.append(os.path.join(BASE_DIR, "src"))
    
    # Importar detector de placas (después de añadir al path)
    try:
        from ai.detector_placas_deepstream import DetectorPlacas
    except ImportError:
        logger.error("No se pudo importar el módulo detector_placas_deepstream")
        logger.error("Asegúrese de que el archivo detector_placas_deepstream.py está en src/ai/")
        sys.exit(1)
    
    # Inicializar detector
    try:
        detector = DetectorPlacas(
            modelo_detector=detector_config,
            modelo_ocr=None  # No se usa un modelo OCR específico en esta implementación
        )
        logger.info("Detector de placas inicializado correctamente")
    except Exception as e:
        logger.error(f"Error al inicializar detector: {e}")
        sys.exit(1)
    
    # Procesar según el modo seleccionado
    if args.imagen:
        procesar_imagen(args.imagen, detector)
    elif args.dir:
        procesar_directorio(args.dir, detector)
    elif args.video:
        procesar_video(args.video, detector)
    elif args.camara >= 0:
        procesar_camara(args.camara, detector)
    else:
        logger.error("Debe especificar una imagen, directorio, video o cámara para procesar")
        sys.exit(1)


def procesar_imagen(ruta_imagen, detector):
    """Procesar una sola imagen"""
    logger.info(f"Procesando imagen: {ruta_imagen}")
    
    # Verificar que la imagen existe
    if not os.path.exists(ruta_imagen):
        logger.error(f"No se encontró la imagen: {ruta_imagen}")
        return
    
    # Cargar imagen
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        logger.error(f"No se pudo leer la imagen: {ruta_imagen}")
        return
    
    # Medir tiempo de procesamiento
    start_time = time.time()
    
    # Detectar placa
    placa, confianza, imagen_placa = detector.detectar_placa(imagen)
    
    # Calcular tiempo de procesamiento
    elapsed_time = time.time() - start_time
    
    # Mostrar resultados
    if placa:
        logger.info(f"Placa detectada: {placa} (confianza: {confianza:.2f})")
        logger.info(f"Tiempo de procesamiento: {elapsed_time:.3f} segundos")
        
        # Mostrar imagen con resultados
        result_img = imagen.copy()
        cv2.putText(result_img, f"Placa: {placa}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result_img, f"Confianza: {confianza:.2f}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(result_img, f"Tiempo: {elapsed_time:.3f}s", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Mostrar imágenes
        cv2.imshow("Original", result_img)
        
        if imagen_placa is not None:
            cv2.imshow("Placa detectada", imagen_placa)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        logger.warning(f"No se detectó placa en la imagen (tiempo: {elapsed_time:.3f}s)")
        
        # Mostrar imagen sin resultados
        cv2.imshow("Original", imagen)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def procesar_directorio(ruta_dir, detector):
    """Procesar todas las imágenes en un directorio"""
    logger.info(f"Procesando imágenes en directorio: {ruta_dir}")
    
    # Verificar que el directorio existe
    if not os.path.exists(ruta_dir):
        logger.error(f"No se encontró el directorio: {ruta_dir}")
        return
    
    # Listar archivos de imagen
    extensiones = ['.jpg', '.jpeg', '.png', '.bmp']
    imagenes = []
    
    for root, _, files in os.walk(ruta_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in extensiones):
                imagenes.append(os.path.join(root, file))
    
    if not imagenes:
        logger.error(f"No se encontraron imágenes en el directorio: {ruta_dir}")
        return
    
    logger.info(f"Encontradas {len(imagenes)} imágenes")
    
    # Procesar cada imagen
    for ruta_imagen in imagenes:
        # Mostrar progreso
        logger.info(f"Procesando: {os.path.basename(ruta_imagen)}")
        
        # Cargar imagen
        imagen = cv2.imread(ruta_imagen)
        if imagen is None:
            logger.error(f"No se pudo leer la imagen: {ruta_imagen}")
            continue
        
        # Detectar placa
        placa, confianza, imagen_placa = detector.detectar_placa(imagen)
        
        # Mostrar resultados
        if placa:
            logger.info(f"Placa detectada: {placa} (confianza: {confianza:.2f})")
            
            # Mostrar imagen con resultados
            result_img = imagen.copy()
            cv2.putText(result_img, f"Placa: {placa}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(result_img, f"Confianza: {confianza:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Mostrar imágenes
            cv2.imshow("Original", result_img)
            
            if imagen_placa is not None:
                cv2.imshow("Placa detectada", imagen_placa)
            
            # Esperar tecla (Esc para salir, cualquier otra para continuar)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if key == 27:  # Esc
                break
        else:
            logger.warning(f"No se detectó placa en la imagen")
            
            # Mostrar imagen sin resultados
            cv2.imshow("Original", imagen)
            
            # Esperar tecla (Esc para salir, cualquier otra para continuar)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            if key == 27:  # Esc
                break


def procesar_video(ruta_video, detector):
    """Procesar un video"""
    logger.info(f"Procesando video: {ruta_video}")
    
    # Verificar que el video existe
    if not os.path.exists(ruta_video):
        logger.error(f"No se encontró el video: {ruta_video}")
        return
    
    # Abrir video
    cap = cv2.VideoCapture(ruta_video)
    if not cap.isOpened():
        logger.error(f"No se pudo abrir el video: {ruta_video}")
        return
    
    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Video: {width}x{height} @ {fps} FPS")
    
    # Variables para estadísticas
    frame_count = 0
    detection_count = 0
    total_time = 0
    
    # Ventana de visualización
    cv2.namedWindow("Detector de Placas", cv2.WINDOW_NORMAL)
    
    while True:
        # Leer frame
        ret, frame = cap.read()
        if not ret:
            logger.info("Fin del video o error al leer frame")
            break
        
        # Incrementar contador de frames
        frame_count += 1
        
        # Procesar cada 5 frames para mayor fluidez
        if frame_count % 5 != 0:
            # Mostrar frame sin procesar
            cv2.imshow("Detector de Placas", frame)
            
            # Esperar tecla (27 = Esc para salir)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
            continue
        
        # Medir tiempo de procesamiento
        start_time = time.time()
        
        # Detectar placa
        placa, confianza, imagen_placa = detector.detectar_placa(frame)
        
        # Calcular tiempo de procesamiento
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        
        # Mostrar frame con resultados
        result_img = frame.copy()
        
        if placa:
            # Incrementar contador de detecciones
            detection_count += 1
            
            # Mostrar información
            cv2.putText(result_img, f"Placa: {placa}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(result_img, f"Confianza: {confianza:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(result_img, "No se detectó placa", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        # Mostrar estadísticas
        avg_time = total_time / (frame_count / 5) if frame_count > 0 else 0
        cv2.putText(result_img, f"Frame: {frame_count}", (10, height - 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, f"Tiempo: {elapsed_time:.3f}s", (10, height - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, f"Promedio: {avg_time:.3f}s", (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar frame
        cv2.imshow("Detector de Placas", result_img)
        
        # Mostrar placa si se detectó
        if placa and imagen_placa is not None:
            cv2.imshow("Placa Detectada", imagen_placa)
        
        # Esperar tecla (27 = Esc para salir)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    
    # Mostrar estadísticas finales
    logger.info(f"Procesados {frame_count} frames")
    logger.info(f"Detectadas {detection_count} placas")
    logger.info(f"Tiempo promedio de procesamiento: {total_time / (frame_count / 5):.3f}s")


def procesar_camara(indice_camara, detector):
    """Procesar video en vivo desde cámara"""
    logger.info(f"Procesando video desde cámara {indice_camara}")
    
    # Abrir cámara
    cap = cv2.VideoCapture(indice_camara)
    if not cap.isOpened():
        logger.error(f"No se pudo abrir la cámara {indice_camara}")
        return
    
    # Obtener propiedades
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    logger.info(f"Cámara: {width}x{height}")
    
    # Variables para estadísticas
    frame_count = 0
    detection_count = 0
    total_time = 0
    
    # Variables para control de FPS
    process_every = 5  # Procesar cada N frames
    last_detection = None
    last_placa_img = None
    
    # Ventana de visualización
    cv2.namedWindow("Detector de Placas", cv2.WINDOW_NORMAL)
    
    while True:
        # Leer frame
        ret, frame = cap.read()
        if not ret:
            logger.error("Error al leer frame de la cámara")
            break
        
        # Incrementar contador de frames
        frame_count += 1
        
        # Inicializar frame de resultado
        result_img = frame.copy()
        
        # Procesar cada N frames para mayor fluidez
        if frame_count % process_every == 0:
            # Medir tiempo de procesamiento
            start_time = time.time()
            
            # Detectar placa
            placa, confianza, imagen_placa = detector.detectar_placa(frame)
            
            # Calcular tiempo de procesamiento
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            if placa:
                # Guardar última detección
                last_detection = (placa, confianza)
                last_placa_img = imagen_placa
                
                # Incrementar contador de detecciones
                detection_count += 1
                
                # Mostrar información
                cv2.putText(result_img, f"Placa: {placa}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(result_img, f"Confianza: {confianza:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                # Mantener última detección por unos frames
                if last_detection and frame_count - last_update < 30:
                    placa, confianza = last_detection
                    
                    # Mostrar información (en amarillo para indicar que es la última detección)
                    cv2.putText(result_img, f"Placa: {placa}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(result_img, f"Confianza: {confianza:.2f}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    cv2.putText(result_img, "No se detectó placa", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Guardar timestamp de última actualización
            last_update = frame_count
        else:
            # Usar la última detección conocida
            if last_detection and frame_count - last_update < 30:
                placa, confianza = last_detection
                
                # Mostrar información (en amarillo para indicar que es la última detección)
                cv2.putText(result_img, f"Placa: {placa}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(result_img, f"Confianza: {confianza:.2f}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        # Mostrar estadísticas
        processed_frames = frame_count // process_every
        avg_time = total_time / processed_frames if processed_frames > 0 else 0
        fps = 1.0 / avg_time if avg_time > 0 else 0
        
        cv2.putText(result_img, f"FPS: {fps:.1f}", (10, height - 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, f"Detecciones: {detection_count}", (10, height - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar frame
        cv2.imshow("Detector de Placas", result_img)
        
        # Mostrar placa si se tiene una imagen
        if last_placa_img is not None:
            cv2.imshow("Placa Detectada", last_placa_img)
        
        # Esperar tecla (27 = Esc para salir)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    
    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()
    
    # Mostrar estadísticas finales
    logger.info(f"Procesados {frame_count} frames")
    logger.info(f"Detectadas {detection_count} placas")
    
    processed_frames = frame_count // process_every
    if processed_frames > 0:
        logger.info(f"Tiempo promedio de procesamiento: {total_time / processed_frames:.3f}s")
        logger.info(f"FPS promedio: {processed_frames / total_time:.1f}")


if __name__ == "__main__":
    main()

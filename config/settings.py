#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuraciones globales para el Sistema de Parqueadero con IA
"""

import os
import yaml
import logging
from pathlib import Path

# Directorio base del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Cargar archivo de configuración externo si existe
config_file = os.path.join(BASE_DIR, 'config', 'config.yaml')
custom_settings = {}

if os.path.exists(config_file):
    try:
        with open(config_file, 'r') as f:
            custom_settings = yaml.safe_load(f)
        logging.info(f"Configuración personalizada cargada desde {config_file}")
    except Exception as e:
        logging.error(f"Error al cargar archivo de configuración {config_file}: {e}")
        custom_settings = {}

# Configuración predeterminada
DEFAULT_SETTINGS = {
    # Configuración de base de datos
    'database': {
        'host': 'localhost',
        'port': 3306,
        'user': 'usuario_parqueadero',
        'password': 'tu_contraseña',
        'database': 'parqueadero',
        'charset': 'utf8mb4'
    },
    
    # Configuración de cámara PTZ
    'camera': {
        'enabled': True,
        'ip': '192.168.1.100',
        'port': 80,
        'username': 'admin',
        'password': 'admin',
        'rtsp_url': 'rtsp://{username}:{password}@{ip}:554/Streaming/Channels/101',  # Hikvision
        'preset_entrada': 1,
        'preset_salida': 2,
        'preset_general': 3,
        'fps': 15,
        'resolution': '1280x720'
    },
    
    # Configuración de GPIO para fotoceldas
    'gpio': {
        'enabled': True,
        'pin_entrada': 35,  # Pin en modo BOARD
        'pin_salida': 37,   # Pin en modo BOARD
        'debounce_ms': 300  # Tiempo de debounce en milisegundos
    },
    
    # Configuración de modelos de IA
    'ai': {
        'modelo_detector_placas': os.path.join(BASE_DIR, 'modelos', 'ai_models', 'plate_detector.trt'),
        'modelo_ocr_placas': os.path.join(BASE_DIR, 'modelos', 'ai_models', 'plate_ocr.trt'),
        'modelo_clasificador_vehiculos': os.path.join(BASE_DIR, 'modelos', 'ai_models', 'vehicle_classifier.trt'),
        'usar_clasificador_vehiculos': False,
        'umbral_confianza': 0.7,
        'precision': 'fp16'  # Opciones: fp32, fp16
    },
    
    # Configuración de interfaz de usuario
    'ui': {
        'show_ptz_controls': True,
        'theme': 'default'  # Opciones: default, dark
    },
    
    # Configuración de logging
    'logging': {
        'level': 'INFO',
        'max_size_mb': 10,
        'backup_count': 5
    },
    
    # Configuración de almacenamiento de imágenes
    'storage': {
        'entrada_dir': os.path.join(BASE_DIR, 'imagenes', 'entrada'),
        'salida_dir': os.path.join(BASE_DIR, 'imagenes', 'salida'),
        'procesado_dir': os.path.join(BASE_DIR, 'imagenes', 'procesado'),
        'max_days': 30  # Días máximos para conservar imágenes
    },
    
    # Configuración de rendimiento
    'performance': {
        'max_workers': 2,  # Número de hilos de trabajo para procesamiento
        'limit_fps': True,  # Limitar FPS para reducir carga de CPU/GPU
        'target_fps': 15   # FPS objetivo
    }
}

# Combinar configuración predeterminada con personalizada
def deep_update(source, overrides):
    """Actualización profunda de diccionarios anidados"""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source[key], dict):
            deep_update(source[key], value)
        else:
            source[key] = value
    return source

# Aplicar configuración personalizada si existe
SETTINGS = deep_update(DEFAULT_SETTINGS.copy(), custom_settings)

# Crear directorios si no existen
for directory in [
    SETTINGS['storage']['entrada_dir'],
    SETTINGS['storage']['salida_dir'],
    SETTINGS['storage']['procesado_dir'],
    os.path.join(BASE_DIR, 'logs')
]:
    os.makedirs(directory, exist_ok=True)

# Configuración adicional de logging
logging.basicConfig(
    level=getattr(logging, SETTINGS['logging']['level']),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(BASE_DIR, 'logs', f'parqueadero.log')),
        logging.StreamHandler()
    ]
)

# Exportar configuración
if __name__ == "__main__":
    import json
    print(json.dumps(SETTINGS, indent=4))

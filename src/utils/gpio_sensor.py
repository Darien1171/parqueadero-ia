#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para gestión de fotoceldas conectadas a GPIO en la Jetson Orin Nano
"""
import RPi.GPIO as GPIO
import time
import threading
import logging

logger = logging.getLogger('parqueadero.gpio')

class GPIOSensor:
    """Clase para gestionar sensores conectados a GPIO"""
    
    def __init__(self, pin=35, callback=None, debounce_ms=300):
        """
        Inicializa el sensor GPIO
        
        Args:
            pin (int): Número de pin GPIO en modo BOARD
            callback (function): Función a llamar cuando se detecta un vehículo
            debounce_ms (int): Tiempo de debounce en milisegundos
        """
        self.pin = pin
        self.callback = callback
        self.debounce_ms = debounce_ms
        self.is_running = False
        self.last_trigger_time = 0
        
        # Inicializar GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setup(self.pin, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        logger.info(f"Sensor GPIO inicializado en pin {self.pin}")
    
    def _on_sensor_event(self, channel):
        """Manejador de eventos de la fotocelda con debounce"""
        current_time = time.time() * 1000  # Convertir a milisegundos
        
        # Aplicar debounce para evitar activaciones múltiples
        if (current_time - self.last_trigger_time) > self.debounce_ms:
            self.last_trigger_time = current_time
            
            # Detectar estado del sensor
            if GPIO.input(self.pin):
                logger.info(f"Vehículo detectado en sensor pin {self.pin}")
                
                # Ejecutar callback si existe
                if self.callback:
                    self.callback(True)
            else:
                logger.info(f"Vehículo retirado en sensor pin {self.pin}")
                
                # Ejecutar callback con estado inactivo
                if self.callback:
                    self.callback(False)
    
    def start(self):
        """Inicia el monitoreo del sensor"""
        if self.is_running:
            logger.warning(f"El sensor en pin {self.pin} ya está activo")
            return
            
        try:
            # Configurar detección de eventos
            GPIO.add_event_detect(
                self.pin, 
                GPIO.BOTH, 
                callback=self._on_sensor_event,
                bouncetime=self.debounce_ms
            )
            
            self.is_running = True
            logger.info(f"Monitoreo de sensor en pin {self.pin} iniciado")
            
        except Exception as e:
            logger.error(f"Error al iniciar sensor GPIO: {e}")
            raise Exception(f"Error al iniciar sensor GPIO: {e}")
    
    def stop(self):
        """Detiene el monitoreo del sensor"""
        if not self.is_running:
            return
            
        try:
            GPIO.remove_event_detect(self.pin)
            self.is_running = False
            logger.info(f"Monitoreo de sensor en pin {self.pin} detenido")
            
        except Exception as e:
            logger.error(f"Error al detener sensor GPIO: {e}")
    
    def read_state(self):
        """
        Lee el estado actual del sensor
        
        Returns:
            bool: True si se detecta vehículo, False si no
        """
        try:
            return GPIO.input(self.pin)
        except Exception as e:
            logger.error(f"Error al leer estado de sensor: {e}")
            return False
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        self.stop()


class GPIOSensorSimulator(GPIOSensor):
    """
    Clase para simular un sensor GPIO cuando no se está ejecutando en Jetson
    Útil para pruebas en equipos de desarrollo sin pines GPIO
    """
    
    def __init__(self, pin=35, callback=None, debounce_ms=300):
        """
        Inicializa el simulador de sensor GPIO
        
        Args:
            pin (int): Número de pin GPIO simulado
            callback (function): Función a llamar cuando se detecta un vehículo
            debounce_ms (int): Tiempo de debounce en milisegundos
        """
        self.pin = pin
        self.callback = callback
        self.debounce_ms = debounce_ms
        self.is_running = False
        self.last_trigger_time = 0
        self.state = False  # Estado inicial: no hay vehículo
        
        logger.info(f"Simulador de sensor GPIO inicializado en pin simulado {self.pin}")
    
    def start(self):
        """Inicia el monitoreo simulado del sensor"""
        if self.is_running:
            logger.warning(f"El simulador de sensor en pin {self.pin} ya está activo")
            return
        
        self.is_running = True
        logger.info(f"Monitoreo simulado de sensor en pin {self.pin} iniciado")
    
    def stop(self):
        """Detiene el monitoreo simulado del sensor"""
        if not self.is_running:
            return
        
        self.is_running = False
        logger.info(f"Monitoreo simulado de sensor en pin {self.pin} detenido")
    
    def read_state(self):
        """
        Lee el estado actual del sensor simulado
        
        Returns:
            bool: Estado actual del sensor
        """
        return self.state
    
    def simulate_vehicle_detected(self):
        """Simula detección de vehículo"""
        if not self.is_running:
            logger.warning("El simulador no está activo")
            return
        
        current_time = time.time() * 1000
        
        if (current_time - self.last_trigger_time) > self.debounce_ms:
            self.last_trigger_time = current_time
            self.state = True
            
            logger.info(f"Simulando vehículo detectado en pin {self.pin}")
            
            if self.callback:
                self.callback(True)
    
    def simulate_vehicle_removed(self):
        """Simula que el vehículo ya no está presente"""
        if not self.is_running:
            logger.warning("El simulador no está activo")
            return
        
        current_time = time.time() * 1000
        
        if (current_time - self.last_trigger_time) > self.debounce_ms:
            self.last_trigger_time = current_time
            self.state = False
            
            logger.info(f"Simulando vehículo retirado en pin {self.pin}")
            
            if self.callback:
                self.callback(False)


# Función para detectar si estamos en una Jetson real
def is_jetson():
    """
    Detecta si estamos ejecutando en una Jetson
    
    Returns:
        bool: True si estamos en una Jetson, False si no
    """
    try:
        with open('/proc/device-tree/model', 'r') as f:
            model = f.read()
            return 'NVIDIA Jetson' in model
    except:
        return False


# Crear la clase adecuada según la plataforma
def create_sensor(pin=35, callback=None, debounce_ms=300, force_simulation=False):
    """
    Crea un sensor GPIO real o simulado según la plataforma
    
    Args:
        pin (int): Número de pin GPIO
        callback (function): Función a llamar cuando se detecta un vehículo
        debounce_ms (int): Tiempo de debounce en milisegundos
        force_simulation (bool): Forzar el uso de simulación aunque estemos en Jetson
        
    Returns:
        GPIOSensor or GPIOSensorSimulator: Instancia del sensor
    """
    if is_jetson() and not force_simulation:
        return GPIOSensor(pin, callback, debounce_ms)
    else:
        return GPIOSensorSimulator(pin, callback, debounce_ms)


# Para ejecutar pruebas directamente
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Callback de prueba
    def on_vehicle_detected(is_present):
        if is_present:
            print("Vehículo detectado!")
        else:
            print("Vehículo retirado!")
    
    # Determinar si usar sensor real o simulado
    if is_jetson():
        print("Ejecutando en Jetson. Usando sensor GPIO real.")
        sensor = GPIOSensor(pin=35, callback=on_vehicle_detected)
    else:
        print("No estamos en Jetson. Usando simulador GPIO.")
        sensor = GPIOSensorSimulator(pin=35, callback=on_vehicle_detected)
    
    # Iniciar monitoreo
    sensor.start()
    
    try:
        print("Monitoreo de sensor iniciado. Presione Ctrl+C para salir.")
        
        # Si estamos en simulación, simular eventos periódicamente
        if not is_jetson():
            print("Presione 'd' para simular detección de vehículo")
            print("Presione 'r' para simular retiro de vehículo")
            
            import sys
            import select
            import termios
            import tty
            
            # Configurar terminal para lectura de teclas sin esperar Enter
            old_settings = termios.tcgetattr(sys.stdin)
            try:
                tty.setcbreak(sys.stdin.fileno())
                
                while True:
                    # Verificar si hay entrada desde teclado
                    if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                        key = sys.stdin.read(1)
                        
                        if key == 'd':
                            print("Simulando detección de vehículo...")
                            sensor.simulate_vehicle_detected()
                        elif key == 'r':
                            print("Simulando retiro de vehículo...")
                            sensor.simulate_vehicle_removed()
                        elif key == 'q':
                            break
                    
                    time.sleep(0.1)
            finally:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        else:
            # En Jetson real, simplemente esperar
            while True:
                time.sleep(1)
                
    except KeyboardInterrupt:
        print("\nPrograma terminado por el usuario")
    finally:
        # Detener monitoreo
        sensor.stop()
        
        # Limpiar GPIO si estamos en Jetson
        if is_jetson():
            GPIO.cleanup()
        
        print("Sensor detenido y recursos liberados")

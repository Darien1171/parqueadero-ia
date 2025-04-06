#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo para gestión de estado (entrada/salida) en el Sistema de Parqueadero con IA
"""
import logging
from datetime import datetime, timedelta
from config.database import Database
from modelos.vehiculo import Vehiculo

logger = logging.getLogger('parqueadero.estado')

class Estado:
    """
    Clase para gestionar entradas y salidas de vehículos en el parqueadero
    Proporciona métodos para registrar movimientos y consultar estado
    """
    
    def __init__(self, id=None):
        """
        Inicializa un objeto Estado
        
        Args:
            id (int, optional): ID del registro para cargar datos
        """
        # Conexión a base de datos
        self.db = Database()
        
        # Propiedades del registro
        self.id = id
        self.id_vehiculo = None
        self.placa = ""
        self.fecha_entrada = None
        self.fecha_salida = None
        self.imagen_entrada = ""
        self.imagen_salida = ""
        self.observaciones_entrada = ""
        self.observaciones_salida = ""
        
        # Si se proporciona ID, cargar datos
        if id:
            self.cargar(id)
    
    def cargar(self, id):
        """
        Cargar datos de registro desde la base de datos
        
        Args:
            id (int): ID del registro
            
        Returns:
            bool: True si se encontró y cargó el registro, False si no
        """
        try:
            query = """
                SELECT id, id_vehiculo, placa, fecha_entrada, fecha_salida,
                imagen_entrada, imagen_salida, observaciones_entrada, observaciones_salida
                FROM estado
                WHERE id = %s
            """
            
            result = self.db.execute_one(query, (id,))
            
            if result:
                self.id = result['id']
                self.id_vehiculo = result['id_vehiculo']
                self.placa = result['placa']
                self.fecha_entrada = result['fecha_entrada']
                self.fecha_salida = result['fecha_salida']
                self.imagen_entrada = result['imagen_entrada']
                self.imagen_salida = result['imagen_salida']
                self.observaciones_entrada = result['observaciones_entrada']
                self.observaciones_salida = result['observaciones_salida']
                
                return True
            else:
                logger.warning(f"Registro con ID {id} no encontrado")
                return False
                
        except Exception as e:
            logger.error(f"Error al cargar registro {id}: {e}")
            return False
    
    def registrar_entrada(self, placa, imagen_entrada="", observaciones=""):
        """
        Registrar entrada de vehículo al parqueadero
        
        Args:
            placa (str): Placa del vehículo
            imagen_entrada (str): Ruta a la imagen de entrada
            observaciones (str): Observaciones sobre la entrada
            
        Returns:
            int: ID del registro creado o 0 si hubo error
        """
        try:
            # Normalizar placa
            placa = Vehiculo.normalizar_placa(placa)
            
            if not placa:
                logger.error("No se puede registrar entrada sin placa")
                return 0
            
            # Verificar si el vehículo existe
            vehiculo = Vehiculo().buscar_por_placa(placa)
            id_vehiculo = vehiculo.id if vehiculo else None
            
            # Verificar si ya hay una entrada activa para esta placa
            entrada_activa = self.buscar_entrada_activa(placa)
            if entrada_activa:
                logger.warning(f"Ya existe una entrada activa para la placa {placa}")
                return 0
            
            # Registrar entrada
            query = """
                INSERT INTO estado
                (id_vehiculo, placa, fecha_entrada, imagen_entrada, observaciones_entrada)
                VALUES (%s, %s, %s, %s, %s)
            """
            
            # Fecha actual para la entrada
            fecha_entrada = datetime.now()
            
            # Insertar registro
            self.id = self.db.execute_update(query, (
                id_vehiculo, placa, fecha_entrada, imagen_entrada, observaciones
            ))
            
            # Actualizar propiedades del objeto
            self.id_vehiculo = id_vehiculo
            self.placa = placa
            self.fecha_entrada = fecha_entrada
            self.imagen_entrada = imagen_entrada
            self.observaciones_entrada = observaciones
            
            logger.info(f"Entrada registrada para vehículo {placa}, ID: {self.id}")
            return self.id
            
        except Exception as e:
            logger.error(f"Error al registrar entrada para {placa}: {e}")
            return 0
    
    def registrar_salida(self, placa, imagen_salida="", observaciones=""):
        """
        Registrar salida de vehículo del parqueadero
        
        Args:
            placa (str): Placa del vehículo
            imagen_salida (str): Ruta a la imagen de salida
            observaciones (str): Observaciones sobre la salida
            
        Returns:
            bool: True si se registró correctamente, False si no
        """
        try:
            # Normalizar placa
            placa = Vehiculo.normalizar_placa(placa)
            
            if not placa:
                logger.error("No se puede registrar salida sin placa")
                return False
            
            # Buscar entrada activa para esta placa
            entrada = self.buscar_entrada_activa(placa)
            if not entrada:
                logger.warning(f"No hay entrada activa para la placa {placa}")
                return False
            
            # Registrar salida
            query = """
                UPDATE estado
                SET fecha_salida = %s, imagen_salida = %s, observaciones_salida = %s
                WHERE id = %s
            """
            
            # Fecha actual para la salida
            fecha_salida = datetime.now()
            
            # Actualizar registro
            self.db.execute_update(query, (
                fecha_salida, imagen_salida, observaciones, entrada.id
            ))
            
            logger.info(f"Salida registrada para vehículo {placa}, ID: {entrada.id}")
            return True
            
        except Exception as e:
            logger.error(f"Error al registrar salida para {placa}: {e}")
            return False
    
    def buscar_entrada_activa(self, placa):
        """
        Buscar si hay una entrada activa (sin salida) para una placa
        
        Args:
            placa (str): Placa del vehículo
            
        Returns:
            Estado: Instancia con datos de la entrada activa, o None si no hay
        """
        try:
            # Normalizar placa
            placa = Vehiculo.normalizar_placa(placa)
            
            query = """
                SELECT id FROM estado
                WHERE placa = %s AND fecha_salida IS NULL
                ORDER BY fecha_entrada DESC
                LIMIT 1
            """
            
            result = self.db.execute_one(query, (placa,))
            
            if result:
                # Crear y devolver instancia de Estado
                entrada = Estado(result['id'])
                return entrada
            else:
                logger.debug(f"No hay entrada activa para la placa {placa}")
                return None
                
        except Exception as e:
            logger.error(f"Error al buscar entrada activa para {placa}: {e}")
            return None
    
    def obtener_vehiculos_en_parqueadero(self):
        """
        Obtener lista de vehículos actualmente en el parqueadero
        
        Returns:
            list: Lista de registros de vehículos en parqueadero
        """
        try:
            query = """
                SELECT e.id, e.placa, e.fecha_entrada, 
                       v.tipo, v.marca, v.color,
                       u.nombre as propietario
                FROM estado e
                LEFT JOIN vehiculo v ON e.id_vehiculo = v.id
                LEFT JOIN usuario u ON v.id_propietario = u.id
                WHERE e.fecha_salida IS NULL
                ORDER BY e.fecha_entrada DESC
            """
            
            results = self.db.execute_query(query)
            
            # Calcular tiempo de estancia para cada vehículo
            for result in results:
                # Tiempo transcurrido desde la entrada
                if result['fecha_entrada']:
                    tiempo_estancia = datetime.now() - result['fecha_entrada']
                    result['tiempo_estancia'] = tiempo_estancia
                    
                    # Formatear en horas y minutos
                    horas = int(tiempo_estancia.total_seconds() // 3600)
                    minutos = int((tiempo_estancia.total_seconds() % 3600) // 60)
                    result['tiempo_estancia_str'] = f"{horas}h {minutos}m"
                else:
                    result['tiempo_estancia'] = None
                    result['tiempo_estancia_str'] = "Desconocido"
            
            return results
            
        except Exception as e:
            logger.error(f"Error al obtener vehículos en parqueadero: {e}")
            return []
    
    def obtener_historial_vehiculo(self, placa, limit=10):
        """
        Obtener historial de entradas y salidas de un vehículo
        
        Args:
            placa (str): Placa del vehículo
            limit (int): Límite de registros a devolver
            
        Returns:
            list: Lista de registros históricos del vehículo
        """
        try:
            # Normalizar placa
            placa = Vehiculo.normalizar_placa(placa)
            
            query = """
                SELECT id, fecha_entrada, fecha_salida, 
                       imagen_entrada, imagen_salida,
                       observaciones_entrada, observaciones_salida
                FROM estado
                WHERE placa = %s
                ORDER BY fecha_entrada DESC
                LIMIT %s
            """
            
            results = self.db.execute_query(query, (placa, limit))
            
            # Calcular tiempo de estancia para cada registro
            for result in results:
                if result['fecha_entrada'] and result['fecha_salida']:
                    tiempo_estancia = result['fecha_salida'] - result['fecha_entrada']
                    result['tiempo_estancia'] = tiempo_estancia
                    
                    # Formatear en horas y minutos
                    horas = int(tiempo_estancia.total_seconds() // 3600)
                    minutos = int((tiempo_estancia.total_seconds() % 3600) // 60)
                    result['tiempo_estancia_str'] = f"{horas}h {minutos}m"
                elif result['fecha_entrada']:
                    # Para entradas sin salida, calcular tiempo hasta ahora
                    tiempo_estancia = datetime.now() - result['fecha_entrada']
                    result['tiempo_estancia'] = tiempo_estancia
                    
                    # Formatear en horas y minutos
                    horas = int(tiempo_estancia.total_seconds() // 3600)
                    minutos = int((tiempo_estancia.total_seconds() % 3600) // 60)
                    result['tiempo_estancia_str'] = f"{horas}h {minutos}m (en curso)"
                else:
                    result['tiempo_estancia'] = None
                    result['tiempo_estancia_str'] = "Desconocido"
            
            return results
            
        except Exception as e:
            logger.error(f"Error al obtener historial para {placa}: {e}")
            return []
    
    def obtener_estadisticas(self, periodo=30):
        """
        Obtener estadísticas de uso del parqueadero
        
        Args:
            periodo (int): Período en días para las estadísticas
            
        Returns:
            dict: Diccionario con estadísticas
        """
        try:
            # Fecha límite para el período
            fecha_limite = datetime.now() - timedelta(days=periodo)
            
            # Estadísticas generales
            query_general = """
                SELECT 
                    COUNT(*) as total_registros,
                    COUNT(DISTINCT placa) as total_vehiculos,
                    SUM(CASE WHEN fecha_salida IS NULL THEN 1 ELSE 0 END) as vehiculos_activos,
                    AVG(TIMESTAMPDIFF(MINUTE, fecha_entrada, IFNULL(fecha_salida, NOW()))) as tiempo_promedio_minutos
                FROM estado
                WHERE fecha_entrada >= %s
            """
            
            result_general = self.db.execute_one(query_general, (fecha_limite,))
            
            # Estadísticas por tipo de vehículo
            query_tipos = """
                SELECT 
                    v.tipo, 
                    COUNT(*) as cantidad
                FROM estado e
                LEFT JOIN vehiculo v ON e.id_vehiculo = v.id
                WHERE e.fecha_entrada >= %s
                GROUP BY v.tipo
                ORDER BY cantidad DESC
            """
            
            result_tipos = self.db.execute_query(query_tipos, (fecha_limite,))
            
            # Estadísticas por hora del día
            query_horas = """
                SELECT 
                    HOUR(fecha_entrada) as hora, 
                    COUNT(*) as entradas
                FROM estado
                WHERE fecha_entrada >= %s
                GROUP BY hora
                ORDER BY hora
            """
            
            result_horas = self.db.execute_query(query_horas, (fecha_limite,))
            
            # Estadísticas por día de la semana
            query_dias = """
                SELECT 
                    DAYOFWEEK(fecha_entrada) as dia, 
                    COUNT(*) as entradas
                FROM estado
                WHERE fecha_entrada >= %s
                GROUP BY dia
                ORDER BY dia
            """
            
            result_dias = self.db.execute_query(query_dias, (fecha_limite,))
            
            # Construir resultado
            estadisticas = {
                'general': result_general,
                'por_tipo': result_tipos,
                'por_hora': result_horas,
                'por_dia': result_dias,
                'periodo_dias': periodo
            }
            
            # Convertir tiempo promedio a formato legible
            if result_general and result_general['tiempo_promedio_minutos']:
                minutos = int(result_general['tiempo_promedio_minutos'])
                horas = minutos // 60
                minutos_restantes = minutos % 60
                estadisticas['general']['tiempo_promedio_str'] = f"{horas}h {minutos_restantes}m"
            
            # Convertir días de la semana a nombres
            dias_semana = ['Domingo', 'Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado']
            for dia in estadisticas['por_dia']:
                dia['nombre_dia'] = dias_semana[dia['dia'] - 1]
            
            return estadisticas
            
        except Exception as e:
            logger.error(f"Error al obtener estadísticas: {e}")
            return {}
    
    def as_dict(self):
        """
        Convertir objeto a diccionario
        
        Returns:
            dict: Representación del registro como diccionario
        """
        # Calcular tiempo de estancia
        tiempo_estancia = None
        tiempo_estancia_str = ""
        
        if self.fecha_entrada:
            if self.fecha_salida:
                tiempo_estancia = self.fecha_salida - self.fecha_entrada
            else:
                tiempo_estancia = datetime.now() - self.fecha_entrada
            
            # Formatear en horas y minutos
            if tiempo_estancia:
                horas = int(tiempo_estancia.total_seconds() // 3600)
                minutos = int((tiempo_estancia.total_seconds() % 3600) // 60)
                tiempo_estancia_str = f"{horas}h {minutos}m"
                if not self.fecha_salida:
                    tiempo_estancia_str += " (en curso)"
        
        return {
            'id': self.id,
            'id_vehiculo': self.id_vehiculo,
            'placa': self.placa,
            'fecha_entrada': self.fecha_entrada,
            'fecha_salida': self.fecha_salida,
            'imagen_entrada': self.imagen_entrada,
            'imagen_salida': self.imagen_salida,
            'observaciones_entrada': self.observaciones_entrada,
            'observaciones_salida': self.observaciones_salida,
            'tiempo_estancia': tiempo_estancia,
            'tiempo_estancia_str': tiempo_estancia_str
        }


# Para ejecutar pruebas directamente
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejemplo de uso
    print("Prueba de modelo Estado")
    
    # Primero, verificar que existe un vehículo para probar
    placa_prueba = input("Ingrese una placa para prueba (o presione Enter para usar 'ABC-123'): ")
    if not placa_prueba:
        placa_prueba = "ABC-123"
        
        # Verificar si el vehículo existe o crearlo
        vehiculo = Vehiculo().buscar_por_placa(placa_prueba)
        if not vehiculo:
            print(f"Creando vehículo de prueba con placa {placa_prueba}")
            vehiculo = Vehiculo()
            vehiculo.placa = placa_prueba
            vehiculo.tipo = "carro"
            vehiculo.marca = "Toyota"
            vehiculo.guardar()
    
    # Registrar entrada
    estado = Estado()
    id_registro = estado.registrar_entrada(
        placa_prueba, 
        "test_entrada.jpg", 
        "Registro de prueba"
    )
    
    if id_registro:
        print(f"Entrada registrada con ID: {id_registro}")
        
        # Mostrar vehículos en parqueadero
        en_parqueadero = estado.obtener_vehiculos_en_parqueadero()
        print(f"Vehículos actualmente en parqueadero: {len(en_parqueadero)}")
        for veh in en_parqueadero:
            print(f"  - {veh['placa']} ({veh['tiempo_estancia_str']})")
        
        # Simular tiempo de estancia
        input("Presione Enter para registrar la salida...")
        
        # Registrar salida
        if estado.registrar_salida(
            placa_prueba, 
            "test_salida.jpg", 
            "Salida de prueba"
        ):
            print(f"Salida registrada para {placa_prueba}")
            
            # Mostrar historial
            historial = estado.obtener_historial_vehiculo(placa_prueba)
            print(f"Historial para {placa_prueba}:")
            for reg in historial:
                print(f"  - Entrada: {reg['fecha_entrada']}, Salida: {reg['fecha_salida']}")
                print(f"    Tiempo: {reg['tiempo_estancia_str']}")
        else:
            print(f"Error al registrar salida para {placa_prueba}")
    else:
        print("Error al registrar entrada")
    
    # Mostrar algunas estadísticas
    print("\nEstadísticas del parqueadero (últimos 30 días):")
    estadisticas = estado.obtener_estadisticas()
    
    if 'general' in estadisticas and estadisticas['general']:
        print(f"Total registros: {estadisticas['general']['total_registros']}")
        print(f"Total vehículos: {estadisticas['general']['total_vehiculos']}")
        print(f"Vehículos activos: {estadisticas['general']['vehiculos_activos']}")
        if 'tiempo_promedio_str' in estadisticas['general']:
            print(f"Tiempo promedio: {estadisticas['general']['tiempo_promedio_str']}")

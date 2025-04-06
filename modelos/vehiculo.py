#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo para gestión de vehículos en el Sistema de Parqueadero con IA
"""
import logging
from datetime import datetime
from config.database import Database

logger = logging.getLogger('parqueadero.vehiculo')

class Vehiculo:
    """
    Clase para gestionar vehículos en el sistema de parqueadero
    Proporciona operaciones CRUD y métodos específicos para vehículos
    """
    
    def __init__(self, id=None):
        """
        Inicializa un objeto Vehículo
        
        Args:
            id (int, optional): ID del vehículo para cargar datos
        """
        # Conexión a base de datos
        self.db = Database()
        
        # Propiedades del vehículo
        self.id = id
        self.placa = ""
        self.tipo = ""  # carro, moto, camión, bus, etc.
        self.marca = ""
        self.modelo = ""
        self.color = ""
        self.id_propietario = None
        self.fecha_registro = None
        self.activo = True
        
        # Si se proporciona ID, cargar datos
        if id:
            self.cargar(id)
    
    def cargar(self, id):
        """
        Cargar datos de vehículo desde la base de datos
        
        Args:
            id (int): ID del vehículo
            
        Returns:
            bool: True si se encontró y cargó el vehículo, False si no
        """
        try:
            query = """
                SELECT id, placa, tipo, marca, modelo, color, id_propietario, 
                fecha_registro, activo
                FROM vehiculo
                WHERE id = %s
            """
            
            result = self.db.execute_one(query, (id,))
            
            if result:
                self.id = result['id']
                self.placa = result['placa']
                self.tipo = result['tipo']
                self.marca = result['marca']
                self.modelo = result['modelo']
                self.color = result['color']
                self.id_propietario = result['id_propietario']
                self.fecha_registro = result['fecha_registro']
                self.activo = bool(result['activo'])
                
                return True
            else:
                logger.warning(f"Vehículo con ID {id} no encontrado")
                return False
                
        except Exception as e:
            logger.error(f"Error al cargar vehículo {id}: {e}")
            return False
    
    def guardar(self):
        """
        Guardar vehículo en la base de datos (crear o actualizar)
        
        Returns:
            int: ID del vehículo guardado o 0 si hubo error
        """
        try:
            # Si no hay placa, no podemos guardar
            if not self.placa:
                logger.error("No se puede guardar vehículo sin placa")
                return 0
                
            # Verificar si el vehículo ya existe por placa
            existing = self.buscar_por_placa(self.placa)
            
            if existing and not self.id:
                # Si existe y estamos intentando crear uno nuevo
                logger.warning(f"Ya existe un vehículo con placa {self.placa}")
                return existing.id
            
            # Normalizar placa
            self.placa = self.normalizar_placa(self.placa)
            
            if self.id:
                # Actualizar vehículo existente
                query = """
                    UPDATE vehiculo SET
                    placa = %s, tipo = %s, marca = %s, modelo = %s, color = %s,
                    id_propietario = %s, activo = %s
                    WHERE id = %s
                """
                
                self.db.execute_update(query, (
                    self.placa, self.tipo, self.marca, self.modelo, self.color,
                    self.id_propietario, self.activo, self.id
                ))
                
                logger.info(f"Vehículo {self.id} ({self.placa}) actualizado")
                return self.id
            else:
                # Crear nuevo vehículo
                query = """
                    INSERT INTO vehiculo
                    (placa, tipo, marca, modelo, color, id_propietario, fecha_registro, activo)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                
                # Si no hay fecha de registro, usar ahora
                if not self.fecha_registro:
                    self.fecha_registro = datetime.now()
                
                # Insertar vehículo
                self.id = self.db.execute_update(query, (
                    self.placa, self.tipo, self.marca, self.modelo, self.color,
                    self.id_propietario, self.fecha_registro, self.activo
                ))
                
                logger.info(f"Nuevo vehículo creado: {self.id} ({self.placa})")
                return self.id
                
        except Exception as e:
            logger.error(f"Error al guardar vehículo: {e}")
            return 0
    
    def eliminar(self):
        """
        Eliminar vehículo de la base de datos
        En realidad hace una eliminación lógica (activo = 0)
        
        Returns:
            bool: True si se eliminó correctamente, False si no
        """
        try:
            if not self.id:
                logger.error("No se puede eliminar vehículo sin ID")
                return False
            
            # Desactivar vehículo en lugar de eliminarlo (eliminación lógica)
            query = "UPDATE vehiculo SET activo = 0 WHERE id = %s"
            
            self.db.execute_update(query, (self.id,))
            
            self.activo = False
            
            logger.info(f"Vehículo {self.id} ({self.placa}) desactivado")
            return True
            
        except Exception as e:
            logger.error(f"Error al eliminar vehículo {self.id}: {e}")
            return False
    
    def buscar_por_placa(self, placa):
        """
        Buscar vehículo por número de placa
        
        Args:
            placa (str): Número de placa a buscar
            
        Returns:
            Vehiculo: Instancia de vehículo si se encuentra, None si no
        """
        try:
            # Normalizar placa
            placa = self.normalizar_placa(placa)
            
            query = """
                SELECT id FROM vehiculo
                WHERE placa = %s
            """
            
            result = self.db.execute_one(query, (placa,))
            
            if result:
                # Crear y devolver instancia de vehículo
                vehiculo = Vehiculo(result['id'])
                return vehiculo
            else:
                logger.debug(f"Vehículo con placa {placa} no encontrado")
                return None
                
        except Exception as e:
            logger.error(f"Error al buscar vehículo por placa {placa}: {e}")
            return None
    
    def buscar_por_propietario(self, id_propietario):
        """
        Buscar todos los vehículos de un propietario
        
        Args:
            id_propietario (int): ID del propietario
            
        Returns:
            list: Lista de objetos Vehiculo del propietario
        """
        try:
            query = """
                SELECT id FROM vehiculo
                WHERE id_propietario = %s AND activo = 1
                ORDER BY fecha_registro DESC
            """
            
            results = self.db.execute_query(query, (id_propietario,))
            
            vehiculos = []
            for result in results:
                vehiculo = Vehiculo(result['id'])
                vehiculos.append(vehiculo)
            
            return vehiculos
            
        except Exception as e:
            logger.error(f"Error al buscar vehículos del propietario {id_propietario}: {e}")
            return []
    
    def obtener_propietario(self):
        """
        Obtener datos del propietario del vehículo
        
        Returns:
            dict: Datos del propietario o None si no existe
        """
        try:
            if not self.id_propietario:
                return None
            
            query = """
                SELECT id, nombre, documento, telefono, email
                FROM usuario
                WHERE id = %s
            """
            
            result = self.db.execute_one(query, (self.id_propietario,))
            
            return result
            
        except Exception as e:
            logger.error(f"Error al obtener propietario del vehículo {self.id}: {e}")
            return None
    
    def obtener_todos(self, activos=True, limit=100):
        """
        Obtener todos los vehículos registrados
        
        Args:
            activos (bool): True para obtener solo activos, False para todos
            limit (int): Límite de resultados a devolver
            
        Returns:
            list: Lista de objetos Vehiculo
        """
        try:
            query = """
                SELECT id FROM vehiculo
                WHERE activo = %s
                ORDER BY fecha_registro DESC
                LIMIT %s
            """
            
            results = self.db.execute_query(query, (1 if activos else 0, limit))
            
            vehiculos = []
            for result in results:
                vehiculo = Vehiculo(result['id'])
                vehiculos.append(vehiculo)
            
            return vehiculos
            
        except Exception as e:
            logger.error(f"Error al obtener todos los vehículos: {e}")
            return []
    
    def buscar_por_criterios(self, criterios, limit=100):
        """
        Buscar vehículos según criterios específicos
        
        Args:
            criterios (dict): Diccionario con criterios de búsqueda
            limit (int): Límite de resultados a devolver
            
        Returns:
            list: Lista de objetos Vehiculo que cumplen los criterios
        """
        try:
            where_clauses = ["activo = 1"]
            params = []
            
            # Construir cláusulas WHERE según criterios
            if 'placa' in criterios and criterios['placa']:
                where_clauses.append("placa LIKE %s")
                params.append(f"%{criterios['placa']}%")
            
            if 'tipo' in criterios and criterios['tipo']:
                where_clauses.append("tipo = %s")
                params.append(criterios['tipo'])
            
            if 'marca' in criterios and criterios['marca']:
                where_clauses.append("marca = %s")
                params.append(criterios['marca'])
            
            if 'color' in criterios and criterios['color']:
                where_clauses.append("color = %s")
                params.append(criterios['color'])
            
            if 'id_propietario' in criterios and criterios['id_propietario']:
                where_clauses.append("id_propietario = %s")
                params.append(criterios['id_propietario'])
            
            # Construir query completa
            query = f"""
                SELECT id FROM vehiculo
                WHERE {' AND '.join(where_clauses)}
                ORDER BY fecha_registro DESC
                LIMIT %s
            """
            
            params.append(limit)
            
            # Ejecutar consulta
            results = self.db.execute_query(query, tuple(params))
            
            vehiculos = []
            for result in results:
                vehiculo = Vehiculo(result['id'])
                vehiculos.append(vehiculo)
            
            return vehiculos
            
        except Exception as e:
            logger.error(f"Error al buscar vehículos por criterios: {e}")
            return []
    
    @staticmethod
    def normalizar_placa(placa):
        """
        Normalizar formato de placa (mayúsculas, sin espacios)
        
        Args:
            placa (str): Placa a normalizar
            
        Returns:
            str: Placa normalizada
        """
        if not placa:
            return ""
            
        # Eliminar espacios y convertir a mayúsculas
        placa = placa.strip().upper()
        
        # Eliminar caracteres no alfanuméricos excepto guion
        placa = ''.join(c for c in placa if c.isalnum() or c == '-')
        
        # Formato estándar: 3 letras, guion, 3 números (placas colombianas)
        if len(placa) == 6:
            # Si tiene 6 caracteres sin guion, insertar guion después de los primeros 3
            if placa[:3].isalpha() and placa[3:].isdigit():
                placa = placa[:3] + '-' + placa[3:]
        
        return placa
    
    def as_dict(self):
        """
        Convertir objeto a diccionario
        
        Returns:
            dict: Representación del vehículo como diccionario
        """
        return {
            'id': self.id,
            'placa': self.placa,
            'tipo': self.tipo,
            'marca': self.marca,
            'modelo': self.modelo,
            'color': self.color,
            'id_propietario': self.id_propietario,
            'fecha_registro': self.fecha_registro,
            'activo': self.activo
        }


# Para ejecutar pruebas directamente
if __name__ == "__main__":
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejemplo de uso
    print("Prueba de modelo Vehiculo")
    
    # Crear vehículo de prueba
    vehiculo = Vehiculo()
    vehiculo.placa = "ABC-123"
    vehiculo.tipo = "carro"
    vehiculo.marca = "Toyota"
    vehiculo.modelo = "Corolla"
    vehiculo.color = "blanco"
    
    # Guardar en BD
    vehiculo_id = vehiculo.guardar()
    
    print(f"Vehículo guardado con ID: {vehiculo_id}")
    
    # Buscar por placa
    vehiculo_encontrado = Vehiculo().buscar_por_placa("ABC-123")
    
    if vehiculo_encontrado:
        print(f"Vehículo encontrado: {vehiculo_encontrado.as_dict()}")
    
    # Actualizar vehículo
    if vehiculo_encontrado:
        vehiculo_encontrado.color = "negro"
        vehiculo_encontrado.guardar()
        print(f"Vehículo actualizado: {vehiculo_encontrado.as_dict()}")
    
    # Buscar por criterios
    vehiculos = Vehiculo().buscar_por_criterios({'marca': 'Toyota'})
    print(f"Vehículos Toyota encontrados: {len(vehiculos)}")
    
    # Eliminar vehículo de prueba (opcional)
    if vehiculo_encontrado:
        if input("¿Eliminar vehículo de prueba? (s/n): ").lower() == 's':
            vehiculo_encontrado.eliminar()
            print("Vehículo eliminado")

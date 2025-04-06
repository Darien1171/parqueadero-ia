#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Modelo para gestión de usuarios/propietarios en el Sistema de Parqueadero con IA
"""
import logging
from datetime import datetime
from config.database import Database

logger = logging.getLogger('parqueadero.usuario')

class Usuario:
    """
    Clase para gestionar usuarios/propietarios en el sistema de parqueadero
    Proporciona operaciones CRUD y métodos específicos para usuarios
    """
    
    def __init__(self, id=None):
        """
        Inicializa un objeto Usuario
        
        Args:
            id (int, optional): ID del usuario para cargar datos
        """
        # Conexión a base de datos
        self.db = Database()
        
        # Propiedades del usuario
        self.id = id
        self.nombre = ""
        self.documento = ""
        self.telefono = ""
        self.email = ""
        self.direccion = ""
        self.fecha_registro = None
        self.activo = True
        
        # Si se proporciona ID, cargar datos
        if id:
            self.cargar(id)
    
    def cargar(self, id):
        """
        Cargar datos de usuario desde la base de datos
        
        Args:
            id (int): ID del usuario
            
        Returns:
            bool: True si se encontró y cargó el usuario, False si no
        """
        try:
            query = """
                SELECT id, nombre, documento, telefono, email, direccion, 
                fecha_registro, activo
                FROM usuario
                WHERE id = %s
            """
            
            result = self.db.execute_one(query, (id,))
            
            if result:
                self.id = result['id']
                self.nombre = result['nombre']
                self.documento = result['documento']
                self.telefono = result['telefono']
                self.email = result['email'] if result['email'] else ""
                self.direccion = result['direccion'] if result['direccion'] else ""
                self.fecha_registro = result['fecha_registro']
                self.activo = bool(result['activo'])
                
                return True
            else:
                logger.warning(f"Usuario con ID {id} no encontrado")
                return False
                
        except Exception as e:
            logger.error(f"Error al cargar usuario {id}: {e}")
            return False
    
    def guardar(self):
        """
        Guardar usuario en la base de datos (crear o actualizar)
        
        Returns:
            int: ID del usuario guardado o 0 si hubo error
        """
        try:
            # Si no hay nombre, no podemos guardar
            if not self.nombre:
                logger.error("No se puede guardar usuario sin nombre")
                return 0
                
            # Verificar si el usuario ya existe por documento
            if self.documento:
                existing = self.buscar_por_documento(self.documento)
                
                if existing and not self.id:
                    # Si existe y estamos intentando crear uno nuevo
                    logger.warning(f"Ya existe un usuario con documento {self.documento}")
                    return existing.id
            
            if self.id:
                # Actualizar usuario existente
                query = """
                    UPDATE usuario SET
                    nombre = %s, documento = %s, telefono = %s, email = %s,
                    direccion = %s, activo = %s
                    WHERE id = %s
                """
                
                self.db.execute_update(query, (
                    self.nombre, self.documento, self.telefono, self.email,
                    self.direccion, self.activo, self.id
                ))
                
                logger.info(f"Usuario {self.id} ({self.nombre}) actualizado")
                return self.id
            else:
                # Crear nuevo usuario
                query = """
                    INSERT INTO usuario
                    (nombre, documento, telefono, email, direccion, fecha_registro, activo)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                """
                
                # Si no hay fecha de registro, usar ahora
                if not self.fecha_registro:
                    self.fecha_registro = datetime.now()
                
                # Insertar usuario
                self.id = self.db.execute_update(query, (
                    self.nombre, self.documento, self.telefono, self.email,
                    self.direccion, self.fecha_registro, self.activo
                ))
                
                logger.info(f"Nuevo usuario creado: {self.id} ({self.nombre})")
                return self.id
                
        except Exception as e:
            logger.error(f"Error al guardar usuario: {e}")
            return 0
    
    def eliminar(self):
        """
        Eliminar usuario de la base de datos
        En realidad hace una eliminación lógica (activo = 0)
        
        Returns:
            bool: True si se eliminó correctamente, False si no
        """
        try:
            if not self.id:
                logger.error("No se puede eliminar usuario sin ID")
                return False
            
            # Verificar si tiene vehículos asociados
            vehiculos = self.obtener_vehiculos()
            if vehiculos:
                logger.warning(f"Usuario {self.id} tiene {len(vehiculos)} vehículos asociados")
                
                # Preguntar si también desea desactivar los vehículos
                for vehiculo in vehiculos:
                    vehiculo.activo = False
                    vehiculo.guardar()
                
                logger.info(f"Vehículos asociados al usuario {self.id} desactivados")
            
            # Desactivar usuario en lugar de eliminarlo (eliminación lógica)
            query = "UPDATE usuario SET activo = 0 WHERE id = %s"
            
            self.db.execute_update(query, (self.id,))
            
            self.activo = False
            
            logger.info(f"Usuario {self.id} ({self.nombre}) desactivado")
            return True
            
        except Exception as e:
            logger.error(f"Error al eliminar usuario {self.id}: {e}")
            return False
    
    def buscar_por_documento(self, documento):
        """
        Buscar usuario por número de documento
        
        Args:
            documento (str): Número de documento a buscar
            
        Returns:
            Usuario: Instancia de usuario si se encuentra, None si no
        """
        try:
            query = """
                SELECT id FROM usuario
                WHERE documento = %s
            """
            
            result = self.db.execute_one(query, (documento,))
            
            if result:
                # Crear y devolver instancia de usuario
                usuario = Usuario(result['id'])
                return usuario
            else:
                logger.debug(f"Usuario con documento {documento} no encontrado")
                return None
                
        except Exception as e:
            logger.error(f"Error al buscar usuario por documento {documento}: {e}")
            return None
    
    def buscar_por_nombre(self, nombre, exact=False):
        """
        Buscar usuarios por nombre
        
        Args:
            nombre (str): Nombre o parte del nombre a buscar
            exact (bool): True para búsqueda exacta, False para parcial
            
        Returns:
            list: Lista de objetos Usuario que coinciden
        """
        try:
            if exact:
                query = """
                    SELECT id FROM usuario
                    WHERE nombre = %s AND activo = 1
                    ORDER BY nombre
                """
                params = (nombre,)
            else:
                query = """
                    SELECT id FROM usuario
                    WHERE nombre LIKE %s AND activo = 1
                    ORDER BY nombre
                """
                params = (f"%{nombre}%",)
            
            results = self.db.execute_query(query, params)
            
            usuarios = []
            for result in results:
                usuario = Usuario(result['id'])
                usuarios.append(usuario)
            
            return usuarios
            
        except Exception as e:
            logger.error(f"Error al buscar usuarios por nombre {nombre}: {e}")
            return []
    
    def obtener_vehiculos(self):
        """
        Obtener vehículos asociados al usuario
        
        Returns:
            list: Lista de objetos Vehiculo del usuario
        """
        try:
            if not self.id:
                return []
            
            # Importar aquí para evitar dependencia circular
            from modelos.vehiculo import Vehiculo
            
            # Buscar vehículos del propietario
            return Vehiculo().buscar_por_propietario(self.id)
            
        except Exception as e:
            logger.error(f"Error al obtener vehículos del usuario {self.id}: {e}")
            return []
    
    def obtener_todos(self, activos=True, limit=100):
        """
        Obtener todos los usuarios registrados
        
        Args:
            activos (bool): True para obtener solo activos, False para todos
            limit (int): Límite de resultados a devolver
            
        Returns:
            list: Lista de objetos Usuario
        """
        try:
            query = """
                SELECT id FROM usuario
                WHERE activo = %s
                ORDER BY nombre
                LIMIT %s
            """
            
            results = self.db.execute_query(query, (1 if activos else 0, limit))
            
            usuarios = []
            for result in results:
                usuario = Usuario(result['id'])
                usuarios.append(usuario)
            
            return usuarios
            
        except Exception as e:
            logger.error(f"Error al obtener todos los usuarios: {e}")
            return []
    
    def buscar_por_criterios(self, criterios, limit=100):
        """
        Buscar usuarios según criterios específicos
        
        Args:
            criterios (dict): Diccionario con criterios de búsqueda
            limit (int): Límite de resultados a devolver
            
        Returns:
            list: Lista de objetos Usuario que cumplen los criterios
        """
        try:
            where_clauses = ["activo = 1"]
            params = []
            
            # Construir cláusulas WHERE según criterios
            if 'nombre' in criterios and criterios['nombre']:
                where_clauses.append("nombre LIKE %s")
                params.append(f"%{criterios['nombre']}%")
            
            if 'documento' in criterios and criterios['documento']:
                where_clauses.append("documento LIKE %s")
                params.append(f"%{criterios['documento']}%")
            
            if 'telefono' in criterios and criterios['telefono']:
                where_clauses.append("telefono LIKE %s")
                params.append(f"%{criterios['telefono']}%")
            
            if 'email' in criterios and criterios['email']:
                where_clauses.append("email LIKE %s")
                params.append(f"%{criterios['email']}%")
            
            # Construir query completa
            query = f"""
                SELECT id FROM usuario
                WHERE {' AND '.join(where_clauses)}
                ORDER BY nombre
                LIMIT %s
            """
            
            params.append(limit)
            
            # Ejecutar consulta
            results = self.db.execute_query(query, tuple(params))
            
            usuarios = []
            for result in results:
                usuario = Usuario(result['id'])
                usuarios.append(usuario)
            
            return usuarios
            
        except Exception as e:
            logger.error(f"Error al buscar usuarios por criterios: {e}")
            return []
    
    def as_dict(self):
        """
        Convertir objeto a diccionario
        
        Returns:
            dict: Representación del usuario como diccionario
        """
        return {
            'id': self.id,
            'nombre': self.nombre,
            'documento': self.documento,
            'telefono': self.telefono,
            'email': self.email,
            'direccion': self.direccion,
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
    print("Prueba de modelo Usuario")
    
    # Crear usuario de prueba
    usuario = Usuario()
    usuario.nombre = "Juan Pérez"
    usuario.documento = "1234567890"
    usuario.telefono = "3001234567"
    usuario.email = "juan.perez@example.com"
    
    # Guardar en BD
    usuario_id = usuario.guardar()
    
    print(f"Usuario guardado con ID: {usuario_id}")
    
    # Buscar por documento
    usuario_encontrado = Usuario().buscar_por_documento("1234567890")
    
    if usuario_encontrado:
        print(f"Usuario encontrado: {usuario_encontrado.as_dict()}")
    
    # Actualizar usuario
    if usuario_encontrado:
        usuario_encontrado.telefono = "3109876543"
        usuario_encontrado.guardar()
        print(f"Usuario actualizado: {usuario_encontrado.as_dict()}")
    
    # Buscar por nombre
    usuarios = Usuario().buscar_por_nombre("Juan")
    print(f"Usuarios encontrados con nombre 'Juan': {len(usuarios)}")
    
    # Eliminar usuario de prueba (opcional)
    if usuario_encontrado:
        if input("¿Eliminar usuario de prueba? (s/n): ").lower() == 's':
            usuario_encontrado.eliminar()
            print("Usuario eliminado")

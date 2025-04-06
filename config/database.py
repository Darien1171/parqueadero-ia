#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para gestión de conexiones a base de datos MySQL
Implementa patrón Singleton para asegurar una única conexión
"""

import logging
import mysql.connector
from mysql.connector import Error
from config.settings import SETTINGS

logger = logging.getLogger('parqueadero.database')

class Database:
    """
    Clase para la gestión de conexiones a base de datos MySQL
    Implementa el patrón Singleton para mantener una única conexión
    """
    
    _instance = None
    _connection = None
    
    def __new__(cls):
        """Implementación del patrón Singleton"""
        if cls._instance is None:
            cls._instance = super(Database, cls).__new__(cls)
            cls._connection = None
        return cls._instance
    
    def __init__(self):
        """Inicializar la conexión a la base de datos si no existe"""
        if Database._connection is None:
            self._connect()
    
    def _connect(self):
        """Establecer conexión con la base de datos MySQL"""
        try:
            Database._connection = mysql.connector.connect(
                host=SETTINGS['database']['host'],
                port=SETTINGS['database']['port'],
                user=SETTINGS['database']['user'],
                password=SETTINGS['database']['password'],
                database=SETTINGS['database']['database'],
                charset=SETTINGS['database']['charset']
            )
            
            if Database._connection.is_connected():
                db_info = Database._connection.get_server_info()
                logger.info(f"Conectado a MySQL versión {db_info}")
                
                # Configurar para que los cursores retornen diccionarios
                Database._connection.cursor(dictionary=True)
                
                # Habilitar autocommit
                Database._connection.autocommit = True
                
        except Error as e:
            logger.error(f"Error al conectar a MySQL: {e}")
            raise Exception(f"Error al conectar a la base de datos: {e}")
    
    def _check_connection(self):
        """Verificar si la conexión está activa y reconectar si es necesario"""
        try:
            if Database._connection is None or not Database._connection.is_connected():
                logger.warning("Conexión a base de datos perdida, reconectando...")
                self._connect()
        except Error as e:
            logger.error(f"Error al verificar conexión: {e}")
            raise Exception(f"Error al verificar conexión a la base de datos: {e}")
    
    def execute_query(self, query, params=None):
        """
        Ejecutar consulta SELECT y retornar resultados
        
        Args:
            query (str): Consulta SQL a ejecutar
            params (tuple, dict, optional): Parámetros para la consulta
            
        Returns:
            list: Lista de resultados como diccionarios
        """
        self._check_connection()
        cursor = None
        
        try:
            cursor = Database._connection.cursor(dictionary=True)
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            result = cursor.fetchall()
            return result
            
        except Error as e:
            logger.error(f"Error al ejecutar consulta: {e}")
            logger.error(f"Query: {query}, Params: {params}")
            raise Exception(f"Error al ejecutar consulta: {e}")
            
        finally:
            if cursor:
                cursor.close()
    
    def execute_one(self, query, params=None):
        """
        Ejecutar consulta SELECT y retornar un solo resultado
        
        Args:
            query (str): Consulta SQL a ejecutar
            params (tuple, dict, optional): Parámetros para la consulta
            
        Returns:
            dict: Un solo resultado como diccionario o None si no hay resultados
        """
        results = self.execute_query(query, params)
        if results:
            return results[0]
        return None
    
    def execute_update(self, query, params=None):
        """
        Ejecutar consultas de actualización (INSERT, UPDATE, DELETE)
        
        Args:
            query (str): Consulta SQL a ejecutar
            params (tuple, dict, optional): Parámetros para la consulta
            
        Returns:
            int: ID del último registro insertado o número de filas afectadas
        """
        self._check_connection()
        cursor = None
        
        try:
            cursor = Database._connection.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            # Commit explícito (aunque esté en autocommit)
            Database._connection.commit()
            
            # Si es un INSERT, devolver el último ID insertado
            if query.strip().upper().startswith("INSERT"):
                last_id = cursor.lastrowid
                return last_id
            
            # En caso contrario, devolver el número de filas afectadas
            return cursor.rowcount
            
        except Error as e:
            logger.error(f"Error al ejecutar actualización: {e}")
            logger.error(f"Query: {query}, Params: {params}")
            
            # Rollback en caso de error
            Database._connection.rollback()
            raise Exception(f"Error al ejecutar actualización: {e}")
            
        finally:
            if cursor:
                cursor.close()
    
    def execute_many(self, query, params_list):
        """
        Ejecutar una consulta con múltiples conjuntos de parámetros
        
        Args:
            query (str): Consulta SQL con placeholders
            params_list (list): Lista de tuplas o diccionarios con parámetros
            
        Returns:
            int: Número de filas afectadas
        """
        self._check_connection()
        cursor = None
        
        try:
            cursor = Database._connection.cursor()
            cursor.executemany(query, params_list)
            
            Database._connection.commit()
            return cursor.rowcount
            
        except Error as e:
            logger.error(f"Error al ejecutar executemany: {e}")
            Database._connection.rollback()
            raise Exception(f"Error al ejecutar múltiples consultas: {e}")
            
        finally:
            if cursor:
                cursor.close()
    
    def execute_script(self, script):
        """
        Ejecutar un script SQL completo
        
        Args:
            script (str): Script SQL a ejecutar
            
        Returns:
            bool: True si la ejecución fue exitosa
        """
        self._check_connection()
        cursor = None
        
        try:
            cursor = Database._connection.cursor()
            
            # Dividir el script en comandos individuales
            # (esto es una simplificación, podría fallar con ciertos scripts complejos)
            for command in script.split(';'):
                command = command.strip()
                if command:
                    cursor.execute(command)
            
            Database._connection.commit()
            return True
            
        except Error as e:
            logger.error(f"Error al ejecutar script SQL: {e}")
            Database._connection.rollback()
            raise Exception(f"Error al ejecutar script SQL: {e}")
            
        finally:
            if cursor:
                cursor.close()
    
    def close(self):
        """Cerrar la conexión a la base de datos"""
        if Database._connection and Database._connection.is_connected():
            Database._connection.close()
            Database._connection = None
            logger.info("Conexión a base de datos cerrada")
    
    def __del__(self):
        """Destructor que cierra la conexión"""
        self.close()


# Funciones de conveniencia para uso sin instanciar la clase

def query(query_str, params=None):
    """
    Ejecutar consulta SELECT y retornar resultados
    
    Args:
        query_str (str): Consulta SQL a ejecutar
        params (tuple, dict, optional): Parámetros para la consulta
        
    Returns:
        list: Lista de resultados como diccionarios
    """
    db = Database()
    return db.execute_query(query_str, params)


def query_one(query_str, params=None):
    """
    Ejecutar consulta SELECT y retornar un solo resultado
    
    Args:
        query_str (str): Consulta SQL a ejecutar
        params (tuple, dict, optional): Parámetros para la consulta
        
    Returns:
        dict: Un solo resultado como diccionario o None si no hay resultados
    """
    db = Database()
    return db.execute_one(query_str, params)


def update(query_str, params=None):
    """
    Ejecutar consultas de actualización (INSERT, UPDATE, DELETE)
    
    Args:
        query_str (str): Consulta SQL a ejecutar
        params (tuple, dict, optional): Parámetros para la consulta
        
    Returns:
        int: ID del último registro insertado o número de filas afectadas
    """
    db = Database()
    return db.execute_update(query_str, params)


# Ejemplo de uso
if __name__ == "__main__":
    # Probar conexión
    try:
        db = Database()
        
        # Consulta simple para verificar funcionamiento
        result = db.execute_query("SELECT VERSION() as version")
        print(f"Versión de MySQL: {result[0]['version']}")
        
        # Cerrar conexión al finalizar
        db.close()
        
    except Exception as e:
        print(f"Error: {e}")

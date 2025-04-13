from model.conexion import abrir_conexion
from model.conexion import cerrar_conexion
from model.usuario import Usuario


def registrar_usuario(usuario):
    
    conexion = abrir_conexion()  
    try:
        
        
        cursor = conexion.cursor()

        # Consulta INSERT
        sql = "INSERT INTO usuario (nombre,documento,telefono,correo) VALUES (%s, %s, %s, %s)"
        valores = (usuario.nombre, usuario.documento, usuario.telefono, usuario.correo)

        # Ejecutar la consulta
        cursor.execute(sql, valores)

        # Confirmar los cambios
        conexion.commit()
        cerrar_conexion(cursor, conexion)  
        return True
        
    except Exception as e:
        print(f"Error al registrar el usuario: {e}")
        cerrar_conexion(cursor, conexion)  
        return False
          

def consultar_usuario(documento):
    

    try:
        conexion = abrir_conexion()   
        
        cursor = conexion.cursor()

        # Consulta SELECT
        sql = "SELECT * FROM usuario WHERE documento = %s"
        valores = (documento,)

        # Ejecutar la consulta
        cursor.execute(sql, valores)

        # Obtener el resultado
        resultado = cursor.fetchone()

        cerrar_conexion(cursor, conexion)
        
        return resultado
    
    except Exception as e:
        print(f"Error al consultar el usuario: {e}")
        cerrar_conexion(cursor, conexion)  
        return False
      
      
def consultar_usuarios():
    try:
        conexion = abrir_conexion()   
        
        cursor = conexion.cursor()

        # Consulta SELECT
        sql = "SELECT * FROM usuario"

        # Ejecutar la consulta
        cursor.execute(sql)

        # Obtener todos los resultados
        resultados = cursor.fetchall()
        
        #Convertir los resultados a una lista de objetos Usuario
        lista_usuarios = []
        for resultado in resultados:
            usuario = Usuario(resultado[1], resultado[2], resultado[3], resultado[4], resultado[0])
            lista_usuarios.append(usuario)

        cerrar_conexion(cursor, conexion)
        
        return lista_usuarios
    
    except Exception as e:
        print(f"Error al consultar los usuarios: {e}")
        cerrar_conexion(cursor, conexion)  
        return False
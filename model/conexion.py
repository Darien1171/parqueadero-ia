import mysql.connector



def abrir_conexion():
    conexion = mysql.connector.connect(
    host="localhost",
    port=3306,
    user="usuario_Parqueadero",       # ejemplo: "root"
    password="P@rqueadero_2024",
    database="parqueadero") 
    return conexion
    
    
def cerrar_conexion(cursor, conexion):
    cursor.close()
    conexion.close()

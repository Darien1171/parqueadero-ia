from model.conexion import abrir_conexion
from model.conexion import cerrar_conexion


def registrar_Vehiculo(vehiculo):
    
    conexion = abrir_conexion()  
    try:
        
        cursor = conexion.cursor()

        # Consulta INSERT
        sql = "INSERT INTO vehiculo (placa,marca,modelo,soat_numero,soat_vencimiento,tecno_numero,tecno_vencimiento,idUsuario) VALUES (%s, %s, %s, %s, %s, %s, %s,%s)"
        valores = (vehiculo.placa, vehiculo.marca, vehiculo.modelo, vehiculo.soat_numero, vehiculo.soat_vencimiento, vehiculo.tecno_numero, vehiculo.tecno_vencimiento, vehiculo.idUsuario)

        # Ejecutar la consulta
        cursor.execute(sql, valores)

        # Confirmar los cambios
        conexion.commit()
        cerrar_conexion(cursor, conexion)  
        return True
        
    except Exception as e:
        print(f"Error al registrar el vehiculo: {e}")
        cerrar_conexion(cursor, conexion)  
        return False
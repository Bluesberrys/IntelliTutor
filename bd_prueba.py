import mysql.connector

# Prueba de conexión
try:
    conn = mysql.connector.connect(
        host="localhost",
        user="root",  # Cambia por tu usuario real
        password="",  # Cambia por tu contraseña real
        database="generador_practicas"
    )
    print("Conexión exitosa a la BD")
    conn.close()
except Exception as e:
    print(f"Error de conexión: {str(e)}")
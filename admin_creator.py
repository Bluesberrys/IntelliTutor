import mysql.connector
from werkzeug.security import generate_password_hash
import random

def get_db_connection():
    """Establece conexión con la base de datos."""
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='generador_practicas'
    )
    return connection

def generar_numero_cuenta():
    """Genera un número de cuenta único de 9 dígitos."""
    while True:
        numero_cuenta = random.randint(100000000, 999999999)
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM usuarios WHERE numero_cuenta = %s", (numero_cuenta,))
        existe = cursor.fetchone()[0]
        connection.close()
        if existe == 0:  # Si no existe, es único
            return numero_cuenta

def crear_admin():
    """Crea una cuenta de administrador directamente en la base de datos."""
    # Datos del administrador
    nombre = input("Nombre del administrador: ")
    email = input("Email del administrador: ")
    password = input("Contraseña: ")
    
    # Generar hash de la contraseña
    password_hash = generate_password_hash(password)
    
    # Generar número de cuenta único
    numero_cuenta = generar_numero_cuenta()
    
    # Insertar el administrador en la base de datos
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Verificar si existe la tabla solicitudes_registro
        cursor.execute("SHOW TABLES LIKE 'solicitudes_registro'")
        tabla_existe = cursor.fetchone()
        
        if not tabla_existe:
            # Crear la tabla si no existe
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS solicitudes_registro (
                id INT AUTO_INCREMENT PRIMARY KEY,
                nombre VARCHAR(100) NOT NULL,
                email VARCHAR(100) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                rol_solicitado ENUM('administrador', 'profesor', 'estudiante') NOT NULL,
                estado ENUM('pendiente', 'aprobada', 'rechazada') DEFAULT 'pendiente',
                fecha_solicitud DATETIME DEFAULT CURRENT_TIMESTAMP,
                password VARCHAR(255)
            )
            """)
            connection.commit()
            print("Tabla solicitudes_registro creada.")
        
        # Insertar directamente en la tabla usuarios
        query = """
        INSERT INTO usuarios (nombre, email, password_hash, rol, numero_cuenta)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (nombre, email, password_hash, 'administrador', numero_cuenta))
        connection.commit()
        
        print("\n=== CUENTA DE ADMINISTRADOR CREADA EXITOSAMENTE ===")
        print(f"Nombre: {nombre}")
        print(f"Email: {email}")
        print(f"Número de cuenta: {numero_cuenta}")
        print(f"Contraseña: {password}")
        print("Guarda esta información en un lugar seguro.")
        
    except mysql.connector.Error as err:
        connection.rollback()
        print(f"Error al crear el administrador: {err}")
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    print("=== CREACIÓN DE CUENTA DE ADMINISTRADOR ===")
    print("Este script creará una cuenta de administrador directamente en la base de datos.")
    crear_admin()

# import mysql.connector
# from werkzeug.security import generate_password_hash
# import random

# def get_db_connection():
#     connection = mysql.connector.connect(
#         host='localhost',
#         user='root',
#         password='',
#         database='generador_practicas'
#     )
#     return connection

# def generar_numero_cuenta():
#     """Genera un número de cuenta único de 9 dígitos."""
#     while True:
#         numero_cuenta = random.randint(100000000, 999999999)
#         connection = get_db_connection()
#         cursor = connection.cursor()
#         cursor.execute("SELECT COUNT(*) FROM usuarios WHERE numero_cuenta = %s", (numero_cuenta,))
#         existe = cursor.fetchone()[0]
#         connection.close()
#         if existe == 0:
#             return numero_cuenta

# def registrar_usuario(nombre, email, password_hash, rol):
#     """Inserta un nuevo usuario directamente en la tabla usuarios."""
#     numero_cuenta = generar_numero_cuenta()
#     connection = get_db_connection()
#     cursor = connection.cursor()
#     query = """
#         INSERT INTO usuarios (nombre, email, password_hash, rol, numero_cuenta)
#         VALUES (%s, %s, %s, %s, %s)
#     """
#     cursor.execute(query, (nombre, email, password_hash, rol, numero_cuenta))
#     connection.commit()
#     cursor.close()
#     connection.close()

# def obtener_usuarios():
#     """Devuelve todos los usuarios registrados."""
#     connection = get_db_connection()
#     cursor = connection.cursor(dictionary=True)
#     cursor.execute("SELECT id, nombre, email, rol FROM usuarios")
#     usuarios = cursor.fetchall()
#     cursor.close()
#     connection.close()
#     return usuarios

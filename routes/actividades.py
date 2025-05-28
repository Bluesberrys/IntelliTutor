from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
import mysql.connector
from datetime import datetime

actividades_bp = Blueprint('actividades', __name__)

def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='generador_practicas'
    )
    return connection

@actividades_bp.route('/api/actividades/recientes', methods=['GET'])
@login_required
def obtener_actividades_recientes():
    """Obtiene las actividades recientes para mostrar en el dashboard."""
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Si es administrador, mostrar todas las actividades
        if current_user.rol == 'administrador':
            query = """
                SELECT a.id, a.descripcion, a.fecha, a.tipo, a.recurso_id, 
                       u.nombre as usuario_nombre, u.rol as usuario_rol
                FROM actividades a
                LEFT JOIN usuarios u ON a.usuario_id = u.id
                ORDER BY a.fecha DESC
                LIMIT 10
            """
            cursor.execute(query)
        else:
            # Si es profesor, mostrar actividades relacionadas con sus grupos
            if current_user.rol == 'profesor':
                query = """
                    SELECT a.id, a.descripcion, a.fecha, a.tipo, a.recurso_id, 
                           u.nombre as usuario_nombre, u.rol as usuario_rol
                    FROM actividades a
                    LEFT JOIN usuarios u ON a.usuario_id = u.id
                    LEFT JOIN practicas p ON a.recurso_id = p.id AND a.tipo = 'practica'
                    LEFT JOIN grupos g ON (a.recurso_id = g.id AND a.tipo = 'grupo') OR (p.grupo_id = g.id)
                    WHERE g.profesor_id = %s OR a.usuario_id = %s
                    ORDER BY a.fecha DESC
                    LIMIT 10
                """
                cursor.execute(query, (current_user.id, current_user.id))
            else:
                # Si es estudiante, mostrar actividades relacionadas con sus grupos y prácticas
                query = """
                    SELECT a.id, a.descripcion, a.fecha, a.tipo, a.recurso_id, 
                           u.nombre as usuario_nombre, u.rol as usuario_rol
                    FROM actividades a
                    LEFT JOIN usuarios u ON a.usuario_id = u.id
                    LEFT JOIN practicas p ON a.recurso_id = p.id AND a.tipo = 'practica'
                    LEFT JOIN evaluaciones e ON p.id = e.practica_id
                    LEFT JOIN grupos g ON p.grupo_id = g.id
                    LEFT JOIN grupo_estudiante ge ON g.id = ge.grupo_id
                    WHERE e.estudiante_id = %s OR ge.estudiante_id = %s OR a.usuario_id = %s
                    ORDER BY a.fecha DESC
                    LIMIT 10
                """
                cursor.execute(query, (current_user.id, current_user.id, current_user.id))
        
        actividades = cursor.fetchall()
        connection.close()
        
        # Formatear fechas para JSON
        for actividad in actividades:
            actividad['fecha'] = actividad['fecha'].strftime('%Y-%m-%d %H:%M:%S')
        
        return jsonify({'success': True, 'actividades': actividades})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@actividades_bp.route('/api/actividades/registrar', methods=['POST'])
@login_required
def registrar_actividad():
    """Registra una nueva actividad en el sistema."""
    try:
        data = request.json
        descripcion = data.get('descripcion')
        tipo = data.get('tipo')
        recurso_id = data.get('recurso_id')
        
        if not descripcion or not tipo:
            return jsonify({'success': False, 'error': 'Faltan datos requeridos'}), 400
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        query = """
            INSERT INTO actividades (descripcion, fecha, usuario_id, tipo, recurso_id)
            VALUES (%s, NOW(), %s, %s, %s)
        """
        cursor.execute(query, (descripcion, current_user.id, tipo, recurso_id))
        connection.commit()
        
        actividad_id = cursor.lastrowid
        connection.close()
        
        return jsonify({'success': True, 'actividad_id': actividad_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

def registrar_actividad_sistema(descripcion, tipo, recurso_id=None, usuario_id=None):
    """
    Función auxiliar para registrar actividades desde otras partes del código.
    
    Args:
        descripcion: Descripción de la actividad
        tipo: Tipo de actividad (practica, grupo, evaluacion, etc.)
        recurso_id: ID del recurso relacionado (opcional)
        usuario_id: ID del usuario que realizó la actividad (opcional, por defecto el usuario actual)
    
    Returns:
        bool: True si se registró correctamente, False en caso contrario
    """
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        if usuario_id is None and current_user.is_authenticated:
            usuario_id = current_user.id
        
        query = """
            INSERT INTO actividades (descripcion, fecha, usuario_id, tipo, recurso_id)
            VALUES (%s, NOW(), %s, %s, %s)
        """
        cursor.execute(query, (descripcion, usuario_id, tipo, recurso_id))
        connection.commit()
        connection.close()
        
        return True
    except Exception as e:
        print(f"Error al registrar actividad: {str(e)}")
        return False

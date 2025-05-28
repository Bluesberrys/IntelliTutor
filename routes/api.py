from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
import mysql.connector
from datetime import datetime

api_bp = Blueprint('api', __name__)

def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='generador_practicas'
    )
    return connection

@api_bp.route('/api/grupos/<int:grupo_id>/criterios', methods=['GET'])
@login_required
def obtener_criterios(grupo_id):
    """Obtiene los criterios de evaluación de un grupo."""
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Verificar que el grupo exista y pertenezca al profesor actual (si no es administrador)
        if current_user.rol == 'profesor':
            cursor.execute("""
                SELECT id FROM grupos WHERE id = %s AND profesor_id = %s
            """, (grupo_id, current_user.id))
            grupo = cursor.fetchone()
            if not grupo:
                connection.close()
                return jsonify({'success': False, 'error': 'No tienes permisos para ver este grupo.'}), 403
        
        # Obtener criterios de evaluación
        cursor.execute("""
            SELECT * FROM criterios_evaluacion WHERE grupo_id = %s
        """, (grupo_id,))
        criterios = cursor.fetchone()
        
        connection.close()
        
        if not criterios:
            # Si no hay criterios, devolver valores por defecto
            return jsonify({
                'success': True,
                'criterios': {
                    'practicas_porcentaje': 40,
                    'examenes_porcentaje': 30,
                    'proyectos_porcentaje': 20,
                    'asistencia_porcentaje': 10
                }
            })
        
        return jsonify({
            'success': True,
            'criterios': criterios
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/api/grupos/<int:grupo_id>/criterios', methods=['POST'])
@login_required
def actualizar_criterios(grupo_id):
    """Actualiza los criterios de evaluación de un grupo."""
    try:
        data = request.json
        practicas_porcentaje = float(data.get('practicas_porcentaje', 40))
        examenes_porcentaje = float(data.get('examenes_porcentaje', 30))
        proyectos_porcentaje = float(data.get('proyectos_porcentaje', 20))
        asistencia_porcentaje = float(data.get('asistencia_porcentaje', 10))
        
        # Verificar que los porcentajes sumen 100%
        total = practicas_porcentaje + examenes_porcentaje + proyectos_porcentaje + asistencia_porcentaje
        if abs(total - 100) > 0.01:
            return jsonify({'success': False, 'error': 'Los porcentajes deben sumar 100%.'}), 400
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Verificar que el grupo exista y pertenezca al profesor actual (si no es administrador)
        if current_user.rol == 'profesor':
            cursor.execute("""
                SELECT id FROM grupos WHERE id = %s AND profesor_id = %s
            """, (grupo_id, current_user.id))
            grupo = cursor.fetchone()
            if not grupo:
                connection.close()
                return jsonify({'success': False, 'error': 'No tienes permisos para modificar este grupo.'}), 403
        
        # Verificar si ya existen criterios para este grupo
        cursor.execute("""
            SELECT id FROM criterios_evaluacion WHERE grupo_id = %s
        """, (grupo_id,))
        criterios_existentes = cursor.fetchone()
        
        if criterios_existentes:
            # Actualizar criterios existentes
            query = """
                UPDATE criterios_evaluacion 
                SET practicas_porcentaje = %s, examenes_porcentaje = %s, 
                    proyectos_porcentaje = %s, asistencia_porcentaje = %s
                WHERE grupo_id = %s
            """
            cursor.execute(query, (
                practicas_porcentaje, examenes_porcentaje, 
                proyectos_porcentaje, asistencia_porcentaje, 
                grupo_id
            ))
        else:
            # Crear nuevos criterios
            query = """
                INSERT INTO criterios_evaluacion 
                (grupo_id, practicas_porcentaje, examenes_porcentaje, proyectos_porcentaje, asistencia_porcentaje)
                VALUES (%s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                grupo_id, practicas_porcentaje, examenes_porcentaje, 
                proyectos_porcentaje, asistencia_porcentaje
            ))
        
        connection.commit()
        connection.close()
        
        # Registrar actividad
        from routes.actividades import registrar_actividad_sistema
        registrar_actividad_sistema(
            f"Actualización de criterios de evaluación para el grupo {grupo_id}",
            'grupo',
            grupo_id
        )
        
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@api_bp.route('/api/practicas/<int:practica_id>/detalles', methods=['GET'])
@login_required
def obtener_detalles_practica(practica_id):
    """Obtiene los detalles de una práctica."""
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Obtener información de la práctica
        query = """
            SELECT p.*, m.nombre as nombre_materia, g.nombre as nombre_clase
            FROM practicas p
            JOIN materias m ON p.materia_id = m.id
            LEFT JOIN grupos g ON p.grupo_id = g.id
            WHERE p.id = %s
        """
        cursor.execute(query, (practica_id,))
        practica = cursor.fetchone()
        
        if not practica:
            connection.close()
            return jsonify({'success': False, 'error': 'Práctica no encontrada.'}), 404
        
        # Obtener contenido generado si existe
        cursor.execute("""
            SELECT contenido FROM contenido_generado WHERE practica_id = %s
        """, (practica_id,))
        contenido_generado = cursor.fetchone()
        
        connection.close()
        
        # Formatear fechas para JSON
        practica['fecha_entrega'] = practica['fecha_entrega'].strftime('%Y-%m-%d')
        practica['fecha_creacion'] = practica['fecha_creacion'].strftime('%Y-%m-%d')
        
        return jsonify({
            'success': True,
            'practica': practica,
            'contenido_generado': contenido_generado['contenido'] if contenido_generado else None
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

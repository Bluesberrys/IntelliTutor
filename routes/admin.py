from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
from flask_login import login_required, current_user
import mysql.connector
from flask import current_app
from .profesor import generar_reporte_calificaciones, enviar_reporte_por_correo
from datetime import datetime
import json
import os

admin_bp = Blueprint('admin', __name__)

def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='generador_practicas'
    )
    return connection

@admin_bp.route('/materias', methods=['GET', 'POST'])
@login_required
def materias():
    """Gestión de materias para administradores."""
    if current_user.rol != 'administrador':
        flash('No tienes permisos para acceder a esta página.', 'error')
        return redirect(url_for('inicio'))
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Procesar formulario para crear/editar materia
    if request.method == 'POST':
        materia_id = request.form.get('materia_id')
        if materia_id:
            # Editar materia
            nombre = request.form['nombre']
            descripcion = request.form['descripcion']
            creditos = int(request.form['creditos'])
    
            query = """
            UPDATE materias 
            SET nombre = %s, descripcion = %s, creditos = %s
            WHERE id = %s
            """
            cursor.execute(query, (nombre, descripcion, creditos, materia_id))
            connection.commit()
            flash('Materia actualizada exitosamente.', 'success')
        else:
            # Crear nueva materia
            nombre = request.form['nombre']
            descripcion = request.form['descripcion']
            creditos = int(request.form['creditos'])
    
            query = """
            INSERT INTO materias (nombre, descripcion, creditos)
            VALUES (%s, %s, %s)
            """
            cursor.execute(query, (nombre, descripcion, creditos))
            connection.commit()
            flash('Materia creada exitosamente.', 'success')
    
        return redirect(url_for('admin.materias'))

    
    # Obtener todas las materias
    cursor.execute("SELECT * FROM materias ORDER BY nombre")
    materias = cursor.fetchall()
    
    connection.close()
    
    # Obtener información del usuario desde el archivo JSON
    usuarios = cargar_json('usuarios.json')
    usuario_id = str(current_user.id)
    usuario = usuarios.get(usuario_id, {})
    
    return render_template(
        'admin/materias.html',
        materias=materias,
        usuario=usuario,
        año_actual=datetime.now().year
    )

@admin_bp.route('/eliminar_materia/<int:materia_id>', methods=['POST'])
@login_required
def eliminar_materia(materia_id):
    """Elimina una materia."""
    if current_user.rol != 'administrador':
        return jsonify({'success': False, 'error': 'No tienes permisos para realizar esta acción.'}), 403
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Verificar si hay grupos o prácticas que usan esta materia
        cursor.execute("SELECT COUNT(*) FROM grupos WHERE materia_id = %s", (materia_id,))
        grupos_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM practicas WHERE materia_id = %s", (materia_id,))
        practicas_count = cursor.fetchone()[0]
        
        if grupos_count > 0 or practicas_count > 0:
            connection.close()
            return jsonify({
                'success': False, 
                'error': f'No se puede eliminar la materia porque está siendo utilizada en {grupos_count} grupos y {practicas_count} prácticas.'
            }), 400
        
        # Eliminar conceptos asociados a la materia
        cursor.execute("DELETE FROM conceptos WHERE materia_id = %s", (materia_id,))
        
        # Eliminar la materia
        cursor.execute("DELETE FROM materias WHERE id = %s", (materia_id,))
        connection.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        connection.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        connection.close()

@admin_bp.route('/grupos', methods=['GET', 'POST'])
@login_required
def grupos():
    """Gestión de grupos para administradores."""
    if current_user.rol != 'administrador':
        flash('No tienes permisos para acceder a esta página.', 'error')
        return redirect(url_for('inicio'))
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Procesar formulario para crear/editar grupo
    if request.method == 'POST':
        grupo_id = request.form.get('grupo_id')
        nombre = request.form['nombre']
        descripcion = request.form['descripcion']
        materia_id = request.form['materia_id']
        profesor_id = request.form['profesor_id']
        semestre_id = request.form['semestre_id']
        turno = request.form['turno']
        fecha_inicio = request.form.get('fecha_inicio')
        fecha_fin = request.form.get('fecha_fin')
    
        if grupo_id:
            # Editar grupo existente
            query = """
            UPDATE grupos 
            SET nombre = %s, descripcion = %s, materia_id = %s, profesor_id = %s, 
                semestre_id = %s, turno = %s, fecha_inicio = %s, fecha_fin = %s
            WHERE id = %s
            """
            cursor.execute(query, (
                nombre, descripcion, materia_id, profesor_id, 
                semestre_id, turno, fecha_inicio, fecha_fin, grupo_id
            ))
            connection.commit()
            flash('Grupo actualizado exitosamente.', 'success')
    
        else:
            # Crear nuevo grupo
            query = """
            INSERT INTO grupos (nombre, descripcion, materia_id, profesor_id, semestre_id, turno, fecha_inicio, fecha_fin)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(query, (
                nombre, descripcion, materia_id, profesor_id, semestre_id, turno, fecha_inicio, fecha_fin
            ))
    
            nuevo_grupo_id = cursor.lastrowid
    
            # Crear criterios de evaluación por defecto
            query_criterios = """
            INSERT INTO criterios_evaluacion 
            (grupo_id, practicas_porcentaje, examenes_porcentaje, proyectos_porcentaje, asistencia_porcentaje)
            VALUES (%s, 40, 30, 20, 10)
            """
            cursor.execute(query_criterios, (nuevo_grupo_id,))
    
            connection.commit()
            flash('Grupo creado exitosamente.', 'success')
    
        return redirect(url_for('admin.grupos'))

    
    # Obtener todos los grupos
    cursor.execute("""
        SELECT g.*, m.nombre as materia_nombre, u.nombre as profesor_nombre, s.nombre as semestre_nombre
        FROM grupos g
        JOIN materias m ON g.materia_id = m.id
        JOIN usuarios u ON g.profesor_id = u.id
        JOIN semestres s ON g.semestre_id = s.id
        ORDER BY g.nombre
    """)
    grupos = cursor.fetchall()
    
    # Obtener materias, profesores y semestres para los formularios
    cursor.execute("SELECT id, nombre FROM materias ORDER BY nombre")
    materias = cursor.fetchall()
    
    cursor.execute("SELECT id, nombre FROM usuarios WHERE rol = 'profesor' ORDER BY nombre")
    profesores = cursor.fetchall()
    
    cursor.execute("SELECT id, nombre FROM semestres ORDER BY fecha_inicio DESC")
    semestres = cursor.fetchall()
    
    connection.close()
    
    # Obtener información del usuario desde el archivo JSON
    usuarios = cargar_json('usuarios.json')
    usuario_id = str(current_user.id)
    usuario = usuarios.get(usuario_id, {})
    
    return render_template(
        'admin/grupos.html',
        grupos=grupos,
        materias=materias,
        profesores=profesores,
        semestres=semestres,
        usuario=usuario,
        año_actual=datetime.now().year
    )

@admin_bp.route('/eliminar_grupo/<int:grupo_id>', methods=['POST'])
@login_required
def eliminar_grupo(grupo_id):
    """Elimina un grupo."""
    if current_user.rol != 'administrador':
        return jsonify({'success': False, 'error': 'No tienes permisos para realizar esta acción.'}), 403
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Verificar si hay prácticas que usan este grupo
        cursor.execute("SELECT COUNT(*) FROM practicas WHERE grupo_id = %s", (grupo_id,))
        practicas_count = cursor.fetchone()[0]
        
        if practicas_count > 0:
            connection.close()
            return jsonify({
                'success': False, 
                'error': f'No se puede eliminar el grupo porque está siendo utilizado en {practicas_count} prácticas.'
            }), 400
        
        # Eliminar miembros del grupo
        cursor.execute("DELETE FROM grupo_miembros WHERE grupo_id = %s", (grupo_id,))
        
        # Eliminar criterios de evaluación
        cursor.execute("DELETE FROM criterios_evaluacion WHERE grupo_id = %s", (grupo_id,))
        
        # Eliminar el grupo
        cursor.execute("DELETE FROM grupos WHERE id = %s", (grupo_id,))
        connection.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        connection.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        connection.close()

@admin_bp.route('/semestres', methods=['GET', 'POST'])
@login_required
def semestres():
    """Gestión de semestres para administradores."""
    if current_user.rol != 'administrador':
        flash('No tienes permisos para acceder a esta página.', 'error')
        return redirect(url_for('inicio'))
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Procesar formulario para crear/editar semestre
    if request.method == 'POST':
        semestre_id = request.form.get('semestre_id')
        nombre = request.form['nombre']
        fecha_inicio = request.form['fecha_inicio']
        fecha_fin = request.form['fecha_fin']
        activo = 'activo' in request.form

        if semestre_id:
            # Editar semestre existente
            if activo:
                cursor.execute("UPDATE semestres SET activo = FALSE WHERE id != %s", (semestre_id,))

            query = """
            UPDATE semestres 
            SET nombre = %s, fecha_inicio = %s, fecha_fin = %s, activo = %s
            WHERE id = %s
            """
            cursor.execute(query, (nombre, fecha_inicio, fecha_fin, activo, semestre_id))
            connection.commit()
            flash('Semestre actualizado exitosamente.', 'success')

        else:
            # Crear nuevo semestre
            if activo:
                cursor.execute("UPDATE semestres SET activo = FALSE")

            query = """
            INSERT INTO semestres (nombre, fecha_inicio, fecha_fin, activo)
            VALUES (%s, %s, %s, %s)
            """
            cursor.execute(query, (nombre, fecha_inicio, fecha_fin, activo))
            connection.commit()
            flash('Semestre creado exitosamente.', 'success')

        return redirect(url_for('admin.semestres'))

    
    # Obtener todos los semestres
    cursor.execute("SELECT * FROM semestres ORDER BY fecha_inicio DESC")
    semestres = cursor.fetchall()
    
    connection.close()
    
    # Obtener información del usuario desde el archivo JSON
    usuarios = cargar_json('usuarios.json')
    usuario_id = str(current_user.id)
    usuario = usuarios.get(usuario_id, {})
    
    return render_template(
        'admin/semestres.html',
        semestres=semestres,
        usuario=usuario,
        año_actual=datetime.now().year
    )

@admin_bp.route('/eliminar_semestre/<int:semestre_id>', methods=['POST'])
@login_required
def eliminar_semestre(semestre_id):
    """Elimina un semestre."""
    if current_user.rol != 'administrador':
        return jsonify({'success': False, 'error': 'No tienes permisos para realizar esta acción.'}), 403
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Verificar si hay grupos que usan este semestre
        cursor.execute("SELECT COUNT(*) FROM grupos WHERE semestre_id = %s", (semestre_id,))
        grupos_count = cursor.fetchone()[0]
        
        if grupos_count > 0:
            connection.close()
            return jsonify({
                'success': False, 
                'error': f'No se puede eliminar el semestre porque está siendo utilizado en {grupos_count} grupos.'
            }), 400
        
        # Eliminar el semestre
        cursor.execute("DELETE FROM semestres WHERE id = %s", (semestre_id,))
        connection.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        connection.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        connection.close()

@admin_bp.route('/reportes', methods=['GET', 'POST'])
@login_required
def reportes():
    """Reportes y envío por correo para administradores."""
    if current_user.rol != 'administrador':
        flash('No tienes permisos para acceder a esta página.', 'error')
        return redirect(url_for('inicio'))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    if request.method == 'POST':
        grupo_id = request.form.get('grupo_id')
        destinatario = request.form.get('correo_destino')

        if grupo_id and destinatario:
            reporte = generar_reporte_calificaciones(grupo_id)
            enviado = enviar_reporte_por_correo(
                destinatario=destinatario,
                asunto=f"Reporte de Calificaciones - {reporte['grupo_nombre']}",
                contenido=f"Adjunto encontrará el reporte de calificaciones del grupo {reporte['grupo_nombre']}.",
                reporte_html=reporte['html']
            )
            if enviado:
                flash('Reporte enviado exitosamente al correo indicado.', 'success')
            else:
                flash('Ocurrió un error al enviar el reporte.', 'error')

            return redirect(url_for('admin.reportes'))

    # Obtener datos estadísticos
    cursor.execute("SELECT COUNT(*) as total FROM usuarios WHERE rol = 'estudiante'")
    total_estudiantes = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as total FROM usuarios WHERE rol = 'profesor'")
    total_profesores = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as total FROM grupos")
    total_grupos = cursor.fetchone()['total']

    cursor.execute("SELECT COUNT(*) as total FROM practicas")
    total_practicas = cursor.fetchone()['total']

    cursor.execute("""
        SELECT m.nombre, AVG(e.calificacion) as promedio
        FROM evaluaciones e
        JOIN practicas p ON e.practica_id = p.id
        JOIN materias m ON p.materia_id = m.id
        WHERE e.calificacion IS NOT NULL
        GROUP BY m.id, m.nombre
        ORDER BY promedio DESC
    """)
    promedios_materias = cursor.fetchall()

    cursor.execute("""
        SELECT s.nombre, COUNT(DISTINCT ge.estudiante_id) as total
        FROM semestres s
        JOIN grupos g ON s.id = g.semestre_id
        JOIN grupo_estudiante ge ON g.id = ge.grupo_id
        GROUP BY s.id, s.nombre
        ORDER BY s.fecha_inicio DESC
    """)
    estudiantes_por_semestre = cursor.fetchall()

    cursor.execute("""
        SELECT a.descripcion, a.fecha, u.nombre as usuario_nombre
        FROM actividades a
        LEFT JOIN usuarios u ON a.usuario_id = u.id
        ORDER BY a.fecha DESC
        LIMIT 10
    """)
    actividades_recientes = cursor.fetchall()

    # Obtener grupos para mostrar en el formulario
    cursor.execute("""
        SELECT g.id, g.nombre, m.nombre AS materia_nombre
        FROM grupos g
        JOIN materias m ON g.materia_id = m.id
        ORDER BY g.nombre
    """)
    grupos = cursor.fetchall()

    connection.close()

    # Obtener información del usuario desde JSON
    usuarios = cargar_json('usuarios.json')
    usuario = usuarios.get(str(current_user.id), {})

    return render_template(
        'admin/reportes.html',
        total_estudiantes=total_estudiantes,
        total_profesores=total_profesores,
        total_grupos=total_grupos,
        total_practicas=total_practicas,
        promedios_materias=promedios_materias,
        estudiantes_por_semestre=estudiantes_por_semestre,
        actividades_recientes=actividades_recientes,
        grupos=grupos,
        usuario=usuario,
        año_actual=datetime.now().year
    )

@admin_bp.route('/iniciar_semestre/<int:semestre_id>', methods=['POST'])
@login_required
def iniciar_semestre(semestre_id):
    """Inicia un nuevo semestre."""
    if current_user.rol != 'administrador':
        return jsonify({'success': False, 'error': 'No tienes permisos para realizar esta acción.'}), 403
    
    connection = get_db_connection()
    cursor = connection.cursor()
    
    try:
        # Desactivar todos los semestres
        cursor.execute("UPDATE semestres SET activo = FALSE")
        
        # Activar el semestre seleccionado
        cursor.execute("UPDATE semestres SET activo = TRUE WHERE id = %s", (semestre_id,))
        
        # Registrar actividad
        cursor.execute("""
            INSERT INTO actividades (descripcion, fecha, usuario_id, tipo)
            VALUES (%s, NOW(), %s, 'semestre')
        """, (f"Inicio del semestre {semestre_id}", current_user.id))
        
        connection.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        connection.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        connection.close()

def cargar_json(path):
    """Carga un archivo JSON y maneja errores."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                print(f"Error: El archivo {path} no contiene un JSON válido.")
                return {}
    else:
        print(f"Error: El archivo {path} no existe.")
        return {}


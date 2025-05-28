from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session, current_app
from flask_login import login_required, current_user
import mysql.connector
from datetime import datetime
import json
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

profesor_bp = Blueprint('profesor', __name__)

def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='generador_practicas'
    )
    return connection

@profesor_bp.route('/enviar_reporte', methods=['GET', 'POST'])
@login_required
def enviar_reporte():
    """Generación de reportes de calificaciones para profesores."""

    if current_user.rol != 'profesor':
        flash('No tienes permisos para acceder a esta página.', 'error')
        return redirect(url_for('inicio'))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    if request.method == 'POST':
        grupo_id = request.form.get('grupo_id')
        departamento = request.form.get('departamento')
        correo_destino = request.form.get('correo_destino')

        if grupo_id and departamento and correo_destino:
            # Generar reporte de calificaciones
            reporte = generar_reporte_calificaciones(grupo_id)

            # Obtener administradores del departamento
            admins = obtener_administradores_por_departamento(departamento)

            # Enviar reporte al correo indicado y a administradores
            enviados = []

            # Enviar al correo directo especificado
            enviados.append(enviar_reporte_por_correo(
                destinatario=correo_destino,
                asunto=f"Reporte de Calificaciones - {reporte['grupo_nombre']}",
                contenido=f"Adjunto encontrará el reporte de calificaciones del grupo {reporte['grupo_nombre']}.",
                reporte_html=reporte['html']
            ))

            # Enviar también a los administradores del departamento
            for admin in admins:
                enviados.append(enviar_reporte_por_correo(
                    destinatario=admin['email'],
                    asunto=f"Reporte de Calificaciones - {reporte['grupo_nombre']}",
                    contenido=f"Adjunto encontrará el reporte de calificaciones del grupo {reporte['grupo_nombre']}.",
                    reporte_html=reporte['html']
                ))

            if any(enviados):
                flash("Reporte enviado correctamente.", "success")
            else:
                flash("No se pudo enviar el reporte. Revisa los correos o intenta más tarde.", "error")

            return redirect(url_for('profesor.enviar_reporte'))

    # Obtener grupos del profesor
    cursor.execute("""
        SELECT g.id, g.nombre, m.nombre as materia_nombre
        FROM grupos g
        JOIN materias m ON g.materia_id = m.id
        WHERE g.profesor_id = %s
    """, (current_user.id,))
    grupos = cursor.fetchall()

    # Obtener departamentos desde la tabla
    cursor.execute("SELECT id, nombre FROM departamentos ORDER BY nombre")
    departamentos = cursor.fetchall()

    connection.close()

    # Cargar usuario desde JSON
    usuarios = cargar_json('usuarios.json')
    usuario_id = str(current_user.id)
    usuario = usuarios.get(usuario_id, {})

    return render_template(
        'profesor/enviar_reporte.html',
        grupos=grupos,
        departamentos=departamentos,
        usuario=usuario,
        año_actual=datetime.now().year
    )



def generar_reporte_calificaciones(grupo_id):
    """Genera un reporte de calificaciones para un grupo."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Obtener información del grupo
    cursor.execute("""
        SELECT g.nombre as grupo_nombre, m.nombre as materia_nombre, 
               s.nombre as semestre_nombre, u.nombre as profesor_nombre
        FROM grupos g
        JOIN materias m ON g.materia_id = m.id
        JOIN semestres s ON g.semestre_id = s.id
        JOIN usuarios u ON g.profesor_id = u.id
        WHERE g.id = %s
    """, (grupo_id,))
    grupo_info = cursor.fetchone()
    
    # Obtener criterios de evaluación
    cursor.execute("""
        SELECT practicas_porcentaje, examenes_porcentaje, proyectos_porcentaje, asistencia_porcentaje
        FROM criterios_evaluacion
        WHERE grupo_id = %s
    """, (grupo_id,))
    criterios = cursor.fetchone()
    
    # Obtener calificaciones de los estudiantes
    cursor.execute("""
        SELECT u.nombre as estudiante_nombre, cf.calificacion_final,
               cf.practicas_promedio, cf.examenes_promedio, 
               cf.proyectos_promedio, cf.asistencia_porcentaje
        FROM calificaciones_finales cf
        JOIN usuarios u ON cf.estudiante_id = u.id
        WHERE cf.grupo_id = %s
        ORDER BY u.nombre
    """, (grupo_id,))
    calificaciones = cursor.fetchall()
    
    connection.close()
    
    # Generar HTML del reporte
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #003d79; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #003d79; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Calificaciones</h1>
        <p><strong>Grupo:</strong> {grupo_info['grupo_nombre']}</p>
        <p><strong>Materia:</strong> {grupo_info['materia_nombre']}</p>
        <p><strong>Semestre:</strong> {grupo_info['semestre_nombre']}</p>
        <p><strong>Profesor:</strong> {grupo_info['profesor_nombre']}</p>
        
        <h2>Criterios de Evaluación</h2>
        <ul>
            <li>Prácticas: {criterios['practicas_porcentaje']}%</li>
            <li>Exámenes: {criterios['examenes_porcentaje']}%</li>
            <li>Proyectos: {criterios['proyectos_porcentaje']}%</li>
            <li>Asistencia: {criterios['asistencia_porcentaje']}%</li>
        </ul>
        
        <h2>Calificaciones</h2>
        <table>
            <tr>
                <th>Estudiante</th>
                <th>Prácticas</th>
                <th>Exámenes</th>
                <th>Proyectos</th>
                <th>Asistencia</th>
                <th>Calificación Final</th>
            </tr>
    """
    
    for cal in calificaciones:
        html += f"""
            <tr>
                <td>{cal['estudiante_nombre']}</td>
                <td>{cal['practicas_promedio']:.2f}</td>
                <td>{cal['examenes_promedio']:.2f}</td>
                <td>{cal['proyectos_promedio']:.2f}</td>
                <td>{cal['asistencia_porcentaje']:.2f}%</td>
                <td>{cal['calificacion_final']:.2f}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <p>Reporte generado el {0} por el sistema IntelliTutor UNAM.</p>
    </body>
    </html>
    """.format(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    
    return {
        'grupo_nombre': grupo_info['grupo_nombre'],
        'html': html
    }

def obtener_departamento_por_carrera(carrera):
    """Obtiene el departamento asociado a una carrera."""
    # Mapeo de carreras a departamentos (esto podría estar en la base de datos)
    departamentos = {
        'Ingeniería en Computación': 'Ingeniería',
        'Ingeniería Eléctrica': 'Ingeniería',
        'Ingeniería Mecánica': 'Ingeniería',
        'Medicina': 'Ciencias de la Salud',
        'Enfermería': 'Ciencias de la Salud',
        'Biología': 'Ciencias',
        'Física': 'Ciencias',
        'Matemáticas': 'Ciencias',
        'Derecho': 'Ciencias Sociales',
        'Economía': 'Ciencias Sociales',
        'Psicología': 'Ciencias Sociales',
        'Literatura': 'Humanidades',
        'Filosofía': 'Humanidades',
        'Historia': 'Humanidades'
    }
    
    return departamentos.get(carrera, 'General')

def obtener_administradores_por_departamento(departamento):
    """Obtiene los administradores asociados a un departamento."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT u.id, u.nombre, u.email
        FROM usuarios u
        JOIN perfiles_administrador pa ON u.id = pa.admin_id
        WHERE pa.departamento = %s
    """, (departamento,))
    admins = cursor.fetchall()
    
    connection.close()
    
    return admins

def enviar_reporte_por_correo(destinatario, asunto, contenido, reporte_html):
    """Envía un reporte por correo electrónico."""
    try:
        # Configuración del servidor SMTP
        smtp_server = current_app.config.get('MAIL_SERVER', 'smtp.gmail.com')
        smtp_port = current_app.config.get('MAIL_PORT', 587)
        smtp_user = current_app.config.get('MAIL_USERNAME', 'sanguineape2189@gmail.com')
        smtp_password = current_app.config.get('MAIL_PASSWORD', 'fvwr iyqg wjlo cskc')
        
        # Crear mensaje
        msg = MIMEMultipart()
        msg['From'] = smtp_user
        msg['To'] = destinatario
        msg['Subject'] = asunto
        
        # Agregar cuerpo del mensaje
        msg.attach(MIMEText(contenido, 'plain'))
        
        # Agregar reporte como archivo adjunto HTML
        attachment = MIMEText(reporte_html, 'html')
        attachment.add_header('Content-Disposition', 'attachment', filename='reporte.html')
        msg.attach(attachment)
        
        # Conectar al servidor SMTP y enviar correo
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_password)
            server.send_message(msg)
        
        return True
    except Exception as e:
        print(f"Error al enviar correo: {str(e)}")
        return False

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

@profesor_bp.route('/historial_sesiones')
@login_required
def historial_sesiones():
    """Muestra los historiales de los estudiantes del profesor."""
    if current_user.rol != 'profesor':
        flash('No tienes permisos para acceder a esta página.', 'error')
        return redirect(url_for('inicio'))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # Obtener estudiantes de los grupos del profesor
    cursor.execute("""
        SELECT u.id, u.nombre
        FROM grupos g
        JOIN grupo_miembros gm ON g.id = gm.grupo_id
        JOIN usuarios u ON gm.usuario_id = u.id
        WHERE g.profesor_id = %s
        GROUP BY u.id
        ORDER BY u.nombre
    """, (current_user.id,))
    estudiantes = cursor.fetchall()

    connection.close()

    return render_template('profesor/historial_sesiones.html', estudiantes=estudiantes)

@profesor_bp.route('/historial_sesiones/<int:estudiante_id>')
@login_required
def ver_historial_estudiante(estudiante_id):
    """Muestra el historial de sesiones de un estudiante, desde sesiones.json."""
    if current_user.rol != 'profesor':
        flash('No tienes permisos para acceder a esta página.', 'error')
        return redirect(url_for('inicio'))

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute("SELECT nombre FROM usuarios WHERE id = %s", (estudiante_id,))
    estudiante = cursor.fetchone()
    connection.close()

    if not estudiante:
        flash("Estudiante no encontrado.", "error")
        return redirect(url_for('profesor.historial_sesiones'))

    try:
        with open('sesiones.json', 'r') as f:
            sesiones = json.load(f)
    except FileNotFoundError:
        sesiones = {}

    historial = sesiones.get(str(estudiante_id), [])

    return render_template('profesor/ver_historial_estudiante.html', estudiante=estudiante, historial=historial)

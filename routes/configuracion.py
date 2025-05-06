import os
import json
from flask import Blueprint, render_template, request, jsonify, session, current_app, flash, redirect, url_for
from flask_login import login_required, current_user
from datetime import datetime
from db_utils import get_db_connection

# Crear Blueprint para la configuración
config_bp = Blueprint('configuracion', __name__, url_prefix='/configuracion')

@config_bp.route('/guardar', methods=['POST'])
@login_required
def guardar_configuracion():
    data = request.get_json()
    print("Formulario recibido:", data)

    usuario_id = str(current_user.id)
    tema = data.get('tema', 'light')
    color_acento = data.get('color_acento', '#003d79')
    tamano_fuente = data.get('tamano_fuente', '14')
    editor_tema = data.get('editor_tema', 'dracula')
    editor_tamano_fuente = data.get('editor_tamano_fuente', '14')
    editor_lenguaje_default = data.get('editor_lenguaje_default', 'javascript')
    editor_autocompletado = data.get('editor_autocompletado', False)
    editor_line_numbers = data.get('editor_line_numbers', False)
    editor_wrap_lines = data.get('editor_wrap_lines', False)
    editor_highlight_active_line = data.get('editor_highlight_active_line', False)
    notificaciones_email = data.get('notificaciones_email', False)
    notificaciones_sistema = data.get('notificaciones_sistema', False)
    perfil_publico = data.get('perfil_publico', False)
    mostrar_calificaciones = data.get('mostrar_calificaciones', False)

    configuraciones = cargar_configuraciones()
    configuraciones[usuario_id] = {
        'tema': tema,
        'color_acento': color_acento,
        'tamano_fuente': tamano_fuente,
        'editor_tema': editor_tema,
        'editor_tamano_fuente': editor_tamano_fuente,
        'editor_lenguaje_default': editor_lenguaje_default,
        'editor_autocompletado': editor_autocompletado,
        'editor_line_numbers': editor_line_numbers,
        'editor_wrap_lines': editor_wrap_lines,
        'editor_highlight_active_line': editor_highlight_active_line,
        'notificaciones_email': notificaciones_email,
        'notificaciones_sistema': notificaciones_sistema,
        'perfil_publico': perfil_publico,
        'mostrar_calificaciones': mostrar_calificaciones
    }
    guardar_configuraciones(configuraciones)
    return jsonify({'success': True, 'message': 'Configuración guardada exitosamente.'})


def cargar_configuraciones():
    """Carga las configuraciones desde el archivo JSON."""
    configuraciones_path = 'configuraciones.json'
    if os.path.exists(configuraciones_path):
        with open(configuraciones_path, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def guardar_configuraciones(configuraciones):
    """Guarda las configuraciones en el archivo JSON."""
    configuraciones_path = 'configuraciones.json'
    with open(configuraciones_path, 'w', encoding='utf-8') as f:
        json.dump(configuraciones, f, indent=4, ensure_ascii=False)

def cargar_configuracion(usuario_id):
    """Carga la configuración de un usuario desde el archivo JSON."""
    configuraciones = cargar_configuraciones()
    return configuraciones.get(str(usuario_id), obtener_configuracion_predeterminada())

def limpiar_configuraciones_huerfanas():
    """Elimina configuraciones de usuarios que ya no existen en la base de datos."""
    configuraciones = cargar_configuraciones()
    ids_a_eliminar = []

    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT id FROM usuarios")
        usuarios_existentes = {str(row[0]) for row in cursor.fetchall()}
        connection.close()
    except Exception as e:
        print(f"Error al obtener usuarios desde la BD: {e}")
        return

    for usuario_id in list(configuraciones.keys()):
        if usuario_id not in usuarios_existentes:
            ids_a_eliminar.append(usuario_id)

    for usuario_id in ids_a_eliminar:
        print(f"Eliminando configuración huérfana de usuario {usuario_id}")
        del configuraciones[usuario_id]

    guardar_configuraciones(configuraciones)
    print(f"Se eliminaron {len(ids_a_eliminar)} configuraciones huérfanas.")


def aplicar_configuracion(app):
    """Registra un procesador de contexto para aplicar configuraciones en todas las plantillas."""
    @app.context_processor
    def inject_configuracion():
        if current_user.is_authenticated:
            usuario_id = str(current_user.id)
            config = cargar_configuracion(usuario_id)
            return {'config_usuario': config}
        return {'config_usuario': obtener_configuracion_predeterminada()}

# Ruta para mostrar la página de configuración
@config_bp.route('/', methods=['GET'])
@login_required
def configuracion():
    """Muestra la página de configuración."""
    usuario_id = str(current_user.id)

    # 👉 Solo para admins: limpiar configuraciones huérfanas
    if getattr(current_user, 'rol', None) == 'admin':
        limpiar_configuraciones_huerfanas()

    configuracion = cargar_configuracion(usuario_id)
    return render_template('configuracion.html', configuracion=configuracion)


# Ruta para exportar datos del usuario
@config_bp.route('/exportar-datos', methods=['GET'])
@login_required
def exportar_datos():
    try:
        # Obtener datos del usuario
        datos = obtener_datos_usuario(current_user.id)
        
        # Convertir a JSON
        datos_json = json.dumps(datos, indent=4, ensure_ascii=False)
        
        # Devolver como archivo descargable
        response = current_app.response_class(
            response=datos_json,
            status=200,
            mimetype='application/json'
        )
        response.headers.set('Content-Disposition', 'attachment', filename='mis_datos.json')
        
        return response
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Función para obtener la configuración predeterminada
def obtener_configuracion_predeterminada():
    return {
        'tema': 'light',
        'color_acento': '#003d79',
        'tamano_fuente': '14',
        'editor_tema': 'dracula',
        'editor_tamano_fuente': '14',
        'editor_lenguaje_default': 'javascript',
        'editor_autocompletado': False,
        'editor_line_numbers': False,
        'editor_wrap_lines': False,
        'editor_highlight_active_line': False,
        'notificaciones_email': False,
        'notificaciones_sistema': False,
        'perfil_publico': False,
        'mostrar_calificaciones': False
    }

# Función para obtener todos los datos del usuario
def obtener_datos_usuario(usuario_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Obtener datos del usuario
        query_usuario = """
            SELECT id, nombre, email, rol, fecha_creacion
            FROM usuarios
            WHERE id = %s
        """
        cursor.execute(query_usuario, (usuario_id,))
        usuario = cursor.fetchone()
        
        if not usuario:
            connection.close()
            return {}
        
        # Obtener configuración
        configuracion = cargar_configuracion(usuario_id)
        
        # Obtener prácticas (si es estudiante)
        practicas = []
        if usuario['rol'] == 'estudiante':
            query_practicas = """
                SELECT p.id, p.titulo, p.objetivo, p.fecha_entrega, e.estado, e.calificacion
                FROM practicas p
                JOIN evaluaciones e ON p.id = e.practica_id
                WHERE e.estudiante_id = %s
                ORDER BY p.fecha_entrega DESC
            """
            cursor.execute(query_practicas, (usuario_id,))
            practicas = cursor.fetchall()
        
        # Obtener entregas (si es estudiante)
        entregas = []
        if usuario['rol'] == 'estudiante':
            query_entregas = """
                SELECT id, practica_id, fecha_entrega, estado, calificacion
                FROM entregas
                WHERE estudiante_id = %s
                ORDER BY fecha_entrega DESC
            """
            cursor.execute(query_entregas, (usuario_id,))
            entregas = cursor.fetchall()
        
        # Obtener grupos (si es estudiante)
        grupos = []
        if usuario['rol'] == 'estudiante':
            query_grupos = """
                SELECT g.id, g.nombre, g.descripcion, m.nombre as materia_nombre
                FROM grupos g
                JOIN grupo_miembros gm ON g.id = gm.grupo_id
                JOIN materias m ON g.materia_id = m.id
                WHERE gm.usuario_id = %s
                ORDER BY g.nombre
            """
            cursor.execute(query_grupos, (usuario_id,))
            grupos = cursor.fetchall()
        
        connection.close()
        
        # Construir el objeto de datos
        datos = {
            'usuario': usuario,
            'configuracion': configuracion,
            'practicas': practicas,
            'entregas': entregas,
            'grupos': grupos,
            'fecha_exportacion': datetime.now().isoformat()
        }
        
        return datos
    
    except Exception as e:
        print(f"Error al obtener datos del usuario: {str(e)}")
        return {}
    
# Función para obtener la configuración actual del usuario
def obtener_configuracion(usuario_id):
    # Intentar obtener de la base de datos primero
    configuracion_json = obtener_configuracion_json(usuario_id)
    if configuracion_json:
        return configuracion_json
    else:
        # Si no hay configuración, devolver valores predeterminados
        return obtener_configuracion_predeterminada()

# Función para obtener la configuración del usuario desde el archivo JSON
def obtener_configuracion_json(usuario_id):
    configuraciones_path = 'configuraciones.json'
    
    if os.path.exists(configuraciones_path):
        try:
            with open(configuraciones_path, 'r', encoding='utf-8') as f:
                configuraciones = json.load(f)
                return configuraciones.get(str(usuario_id), {})
        except json.JSONDecodeError:
            return {}
    
    return {}

# Función para crear configuración por defecto para un nuevo usuario
def crear_configuracion_por_defecto(usuario_id):
    """Crea una configuración por defecto para un nuevo usuario si no existe."""
    configuraciones = cargar_configuraciones()
    if str(usuario_id) not in configuraciones:
        configuraciones[str(usuario_id)] = obtener_configuracion_predeterminada()
        guardar_configuraciones(configuraciones)
        print(f"Configuración por defecto creada para usuario {usuario_id}")
    return configuraciones[str(usuario_id)]
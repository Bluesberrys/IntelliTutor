import os
import subprocess
import tempfile
import time
import json
from flask import Blueprint, render_template, request, jsonify, session, current_app
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from datetime import datetime
import mysql.connector
from db_utils import get_db_connection

# Crear Blueprint para el editor de código
code_editor_bp = Blueprint('code_editor', __name__)

# Ruta para mostrar el editor de código
@code_editor_bp.route('/editor', methods=['GET'])
@login_required
def editor():
    # Obtener prácticas disponibles para el usuario
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    if current_user.rol == 'estudiante':
        # Para estudiantes, mostrar prácticas asignadas
        query = """
        SELECT p.id, p.titulo
        FROM practicas p
        JOIN evaluaciones e ON p.id = e.practica_id
        WHERE e.estudiante_id = %s AND e.estado = 'pendiente'
        ORDER BY p.fecha_entrega ASC
        """
        cursor.execute(query, (current_user.id,))
    else:
        # Para profesores y administradores, mostrar todas las prácticas
        query = """
        SELECT id, titulo
        FROM practicas
        ORDER BY fecha_entrega DESC
        """
        cursor.execute(query)
    
    practicas = cursor.fetchall()
    connection.close()
    
    return render_template('code-editor.html', practicas=practicas)

# Ruta para ejecutar código
@code_editor_bp.route('/ejecutar-codigo', methods=['POST'])
@login_required
def ejecutar_codigo():
    data = request.get_json()
    codigo = data.get('codigo', '')
    lenguaje = data.get('lenguaje', 'python')
    
    if not codigo:
        return jsonify({'success': False, 'error': 'No se proporcionó código para ejecutar'})
    
    # Crear un archivo temporal para el código
    with tempfile.NamedTemporaryFile(suffix=get_file_extension(lenguaje), delete=False) as temp_file:
        temp_file_path = temp_file.name
        temp_file.write(codigo.encode('utf-8'))
    
    try:
        # Ejecutar el código según el lenguaje
        result = ejecutar_codigo_real(temp_file_path, lenguaje)
        
        # Registrar la ejecución en la base de datos
        registrar_ejecucion(codigo, lenguaje, result['success'])
        
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Error al ejecutar el código: {str(e)}'})
    finally:
        # Eliminar el archivo temporal
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

# Ruta para guardar código
@code_editor_bp.route('/guardar-codigo', methods=['POST'])
@login_required
def guardar_codigo():
    data = request.get_json()
    nombre_archivo = data.get('nombre_archivo', '')
    descripcion = data.get('descripcion', '')
    practica_id = data.get('practica_id', None)
    codigo = data.get('codigo', '')
    lenguaje = data.get('lenguaje', 'python')
    
    if not nombre_archivo or not codigo:
        return jsonify({'success': False, 'error': 'Nombre de archivo y código son requeridos'})
    
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Insertar en la base de datos
        query = """
        INSERT INTO archivos_codigo 
        (nombre, descripcion, codigo, lenguaje, usuario_id, practica_id, fecha_creacion)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            nombre_archivo,
            descripcion,
            codigo,
            lenguaje,
            current_user.id,
            practica_id if practica_id else None,
            datetime.now()
        ))
        
        connection.commit()
        archivo_id = cursor.lastrowid
        
        # Guardar también como archivo físico
        extension = get_file_extension(lenguaje)
        directorio = os.path.join(current_app.config['UPLOAD_FOLDER'], 'codigo', str(current_user.id))
        
        # Crear directorio si no existe
        os.makedirs(directorio, exist_ok=True)
        
        # Nombre de archivo seguro
        nombre_seguro = secure_filename(f"{nombre_archivo}_{archivo_id}{extension}")
        ruta_archivo = os.path.join(directorio, nombre_seguro)
        
        with open(ruta_archivo, 'w', encoding='utf-8') as f:
            f.write(codigo)
        
        # Actualizar ruta en la base de datos
        query_update = """
        UPDATE archivos_codigo
        SET ruta_archivo = %s
        WHERE id = %s
        """
        cursor.execute(query_update, (ruta_archivo, archivo_id))
        connection.commit()
        
        connection.close()
        
        return jsonify({'success': True, 'archivo_id': archivo_id})
    except Exception as e:
        if connection:
            connection.rollback()
            connection.close()
        return jsonify({'success': False, 'error': str(e)})

# Ruta para entregar código
@code_editor_bp.route('/entregar_codigo', methods=['POST'])
@login_required
def entregar_codigo():
    if current_user.rol != 'estudiante':
        return jsonify({'success': False, 'error': 'Solo los estudiantes pueden entregar código'})
    
    codigo = request.form.get('codigo', '')
    lenguaje = request.form.get('lenguaje', 'python')
    practica_id = request.form.get('practica_id')
    comentarios = request.form.get('comentarios', '')
    
    if not codigo or not practica_id:
        return jsonify({'success': False, 'error': 'Código y práctica son requeridos'})
    
    try:
        # Verificar que la práctica existe y está asignada al estudiante
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query_check = """
        SELECT 1 FROM evaluaciones
        WHERE practica_id = %s AND estudiante_id = %s AND estado = 'pendiente'
        """
        cursor.execute(query_check, (practica_id, current_user.id))
        
        if not cursor.fetchone():
            connection.close()
            return jsonify({'success': False, 'error': 'La práctica no existe o no está asignada a este estudiante'})
        
        # Guardar el código en un archivo
        extension = get_file_extension(lenguaje)
        directorio = os.path.join(current_app.config['UPLOAD_FOLDER'], 'entregas', str(current_user.id))
        
        # Crear directorio si no existe
        os.makedirs(directorio, exist_ok=True)
        
        # Nombre de archivo seguro
        timestamp = int(time.time())
        nombre_archivo = secure_filename(f"entrega_{practica_id}_{timestamp}{extension}")
        ruta_archivo = os.path.join(directorio, nombre_archivo)
        
        with open(ruta_archivo, 'w', encoding='utf-8') as f:
            f.write(codigo)
        
        # Registrar la entrega en la base de datos
        query_entrega = """
        INSERT INTO entregas 
        (practica_id, estudiante_id, fecha_entrega, estado, archivos_url, contenido)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(query_entrega, (
            practica_id,
            current_user.id,
            datetime.now(),
            'entregado',
            nombre_archivo,
            codigo
        ))
        
        entrega_id = cursor.lastrowid
        
        # Actualizar el estado de la evaluación
        query_evaluacion = """
        UPDATE evaluaciones
        SET estado = 'entregado'
        WHERE practica_id = %s AND estudiante_id = %s
        """
        
        cursor.execute(query_evaluacion, (practica_id, current_user.id))
        
        # Obtener el ID de la evaluación
        query_get_eval = """
        SELECT id FROM evaluaciones
        WHERE practica_id = %s AND estudiante_id = %s
        """
        
        cursor.execute(query_get_eval, (practica_id, current_user.id))
        eval_result = cursor.fetchone()
        evaluacion_id = eval_result['id'] if eval_result else None
        
        # Actualizar la entrega con el ID de evaluación
        if evaluacion_id:
            query_update_entrega = """
            UPDATE entregas
            SET evaluacion_id = %s
            WHERE id = %s
            """
            cursor.execute(query_update_entrega, (evaluacion_id, entrega_id))
        
        connection.commit()
        connection.close()
        
        return jsonify({'success': True, 'entrega_id': entrega_id})
    except Exception as e:
        if connection:
            connection.rollback()
            connection.close()
        return jsonify({'success': False, 'error': str(e)})

# Función para ejecutar código real
def ejecutar_codigo_real(archivo, lenguaje):
    timeout = 10  # Tiempo máximo de ejecución en segundos

    try:
        if lenguaje == 'python':
            proceso = subprocess.Popen(
                ['python', archivo],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        elif lenguaje == 'javascript':
            proceso = subprocess.Popen(
                ['node', archivo],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        elif lenguaje == 'cpp':
            # Compilar primero
            nombre_base = os.path.splitext(archivo)[0]  # Obtener nombre sin extensión
            nombre_ejecutable = nombre_base + ('.exe' if os.name == 'nt' else '.out')
            
            # Asegurarse de que el directorio existe
            os.makedirs(os.path.dirname(nombre_ejecutable), exist_ok=True)
            
            # Comando de compilación
            compilacion = subprocess.run(
                ['g++', archivo, '-o', nombre_ejecutable],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False  # No lanzar excepción si falla
            )
            
            if compilacion.returncode != 0:
                return {
                    'success': False,
                    'error': f'Error de compilación:\n{compilacion.stderr}'
                }
            
            # Ejecutar el programa compilado
            proceso = subprocess.Popen(
                [nombre_ejecutable],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        elif lenguaje == 'java':
            # Extraer el nombre de la clase principal
            with open(archivo, 'r', encoding='utf-8') as f:
                contenido = f.read()
            
            import re
            match = re.search(r'public\s+class\s+(\w+)', contenido)
            if not match:
                return {
                    'success': False,
                    'error': 'No se encontró una clase pública en el código Java'
                }
            
            nombre_clase = match.group(1)
            directorio = os.path.dirname(archivo)
            
            # Compilar primero - asegurarse de que el directorio existe
            os.makedirs(directorio, exist_ok=True)
            
            # Comando de compilación
            compilacion = subprocess.run(
                ['javac', archivo],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False  # No lanzar excepción si falla
            )
            
            if compilacion.returncode != 0:
                return {
                    'success': False,
                    'error': f'Error de compilación:\n{compilacion.stderr}'
                }
            
            # Ejecutar el programa compilado
            proceso = subprocess.Popen(
                ['java', '-cp', directorio, nombre_clase],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
        else:
            return {
                'success': False,
                'error': f'Lenguaje no soportado: {lenguaje}'
            }
        
        try:
            stdout, stderr = proceso.communicate(timeout=timeout)
            
            if proceso.returncode != 0:
                return {
                    'success': False,
                    'error': stderr
                }
            
            return {
                'success': True,
                'output': stdout
            }
        except subprocess.TimeoutExpired:
            proceso.kill()
            return {
                'success': False,
                'error': f'La ejecución excedió el tiempo límite de {timeout} segundos'
            }
    except Exception as e:
        return {
            'success': False,
            'error': f'Error al ejecutar el código: {str(e)}'
        }

# Función para obtener la extensión de archivo según el lenguaje
def get_file_extension(lenguaje):
    extensiones = {
        'python': '.py',
        'javascript': '.js',
        'cpp': '.cpp',
        'java': '.java'
    }
    return extensiones.get(lenguaje, '.txt')

# Función para registrar la ejecución en la base de datos
def registrar_ejecucion(codigo, lenguaje, exito):
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        query = """
        INSERT INTO ejecuciones_codigo 
        (usuario_id, codigo, lenguaje, exito, fecha_ejecucion)
        VALUES (%s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            current_user.id,
            codigo,
            lenguaje,
            exito,
            datetime.now()
        ))
        
        connection.commit()
        connection.close()
    except Exception as e:
        print(f"Error al registrar ejecución: {str(e)}")

print("code_editor cargado correctamente")
print("Variables definidas:", dir())

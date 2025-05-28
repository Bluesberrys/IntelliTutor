from flask import Flask, render_template, request, g, redirect, url_for, flash, jsonify, session
from collections import defaultdict
from generador_practicas import GeneradorPracticasExtendido, Practica, Usuario
from modelo_selector import GeneradorPracticasML
from generar_retroalimentacion import generar_retroalimentacion
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask_mail import Mail, Message
from config import DB_CONFIG, APP_CONFIG
from datetime import datetime, timedelta
import time
from contextlib import closing
import json
import os
import re
import admin_creator
import mysql.connector
import random
from routes.code_editor import code_editor_bp
# Importar el blueprint de configuración
from routes.configuracion import config_bp, aplicar_configuracion, guardar_configuracion, obtener_configuracion, crear_configuracion_por_defecto, cargar_configuracion, obtener_configuracion_predeterminada
# from routes.admin import admin_bp
# from routes.profesor import profesor_bp
# from routes.api import api_bp
# from routes.actividades import actividades_bp
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
# from routes.juegos_educativos import juegos_bp

# Inicialización de la aplicación
port = 5000
app = Flask(__name__)
app.secret_key = APP_CONFIG["SECRET_KEY"]
app.config['DEBUG'] = APP_CONFIG["DEBUG"]
app.config['MAX_CONTENT_LENGTH'] = APP_CONFIG["MAX_CONTENT_LENGTH"]

# Registrar el blueprint en la aplicación
app.register_blueprint(config_bp)
app.register_blueprint(code_editor_bp)
# app.register_blueprint(admin_bp)
# app.register_blueprint(profesor_bp)
# app.register_blueprint(api_bp)
# app.register_blueprint(actividades_bp)
# app.register_blueprint(juegos_bp)

# Aplicar configuraciones globalmente
aplicar_configuracion(app)

# Inicializar generador de prácticas
generador = GeneradorPracticasExtendido(DB_CONFIG)

# Inicializar modelo ML
try:
    # Asegurarse de que sentence-transformers esté instalado
    import importlib.util
    sbert_spec = importlib.util.find_spec('sentence_transformers')
    if sbert_spec is None:
        print("ADVERTENCIA: El módulo sentence_transformers no está instalado. Instalando...")
        import subprocess
        subprocess.check_call(["pip", "install", "sentence-transformers"])
        print("sentence-transformers instalado correctamente.")
    
    # Intentar usar el modelo mejorado con SBERT
    from modelo_ml_enhanced import GeneradorPracticasML
    modelo_ml = GeneradorPracticasML(db_config=DB_CONFIG, use_sbert=True)
    print("Modelo inicializado con soporte SBERT")
except Exception as e:
    print(f"Error al inicializar modelo con SBERT: {str(e)}")
    # Fallback al modelo original o mejorado sin SBERT
    try:
        from modelo_ml_enhanced import GeneradorPracticasML
        modelo_ml = GeneradorPracticasML(db_config=DB_CONFIG, use_sbert=False)
        print("Modelo inicializado sin soporte SBERT (explícitamente desactivado)")
    except Exception as e2:
        print(f"Error al inicializar modelo sin SBERT: {str(e2)}")
        # Último recurso: modelo dummy
        modelo_ml = GeneradorPracticasML(db_config=DB_CONFIG, use_sbert=False)
        print("Modelo inicializado con configuración por defecto")

# Configurar entrenamiento automático periódico
def entrenar_modelo_periodicamente():
    """Función para entrenar el modelo periódicamente."""
    try:
        print("Iniciando entrenamiento periódico del modelo...")
        modelo_ml._entrenar_incremental()
        print("Entrenamiento periódico completado.")
    except Exception as e:
        print(f"Error en entrenamiento periódico: {str(e)}")

# Configurar un temporizador para entrenar el modelo cada 6 horas
import threading
import time

def programar_entrenamiento():
    while True:
        # Esperar 6 horas
        time.sleep(6 * 60 * 60)
        entrenar_modelo_periodicamente()

# Iniciar el hilo de entrenamiento periódico
entrenamiento_thread = threading.Thread(target=programar_entrenamiento, daemon=True)
entrenamiento_thread.start()

# También entrenar al inicio si hay suficientes datos
if hasattr(modelo_ml, 'historial_evaluaciones') and len(modelo_ml.historial_evaluaciones) > 5:
    entrenar_modelo_periodicamente()

# Configuración de Flask-Mail
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'sanguineape2189@gmail.com'
app.config['MAIL_PASSWORD'] = 'fvwr iyqg wjlo cskc'

mail = Mail(app)

# Configuración para la subida de archivos
ALLOWED_EXTENSIONS = {'pdf', 'docx', 'txt', 'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = os.path.join('static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configuración de Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Clase para manejar usuarios con Flask-Login
class UserLogin(UserMixin):
    def __init__(self, id, nombre, email, rol, numero_cuenta):
        self.id = id
        self.nombre = nombre
        self.email = email
        self.rol = rol
        self.numero_cuenta = numero_cuenta

# Aplicar configuraciones globalmente
aplicar_configuracion(app)

# Precargar configuraciones para mejorar el rendimiento
from routes.configuracion import cargar_configuraciones
configuraciones_cache = cargar_configuraciones()
print(f"Configuraciones precargadas: {len(configuraciones_cache)} usuarios")

# Función para obtener configuración con caché
def obtener_configuracion_rapida(usuario_id):
    """Obtiene la configuración de un usuario usando caché para mayor velocidad."""
    from routes.configuracion import obtener_configuracion_predeterminada
    usuario_id_str = str(usuario_id)
    if usuario_id_str in configuraciones_cache:
        return configuraciones_cache[usuario_id_str]
    return obtener_configuracion_predeterminada()

@app.context_processor
def inject_user():
    try:
        usuarios_json = cargar_json('usuarios.json')
        usuario_id = str(current_user.id)
        usuario = usuarios_json.get(usuario_id, {})
        return dict(usuario=usuario)
    except Exception:
        return dict(usuario={})

# Función para cargar usuario desde la sesión
@login_manager.user_loader
def load_user(user_id):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    cursor.execute(
        "SELECT id, nombre, email, rol, numero_cuenta FROM usuarios WHERE id = %s",
        (user_id,)
    )
    user_data = cursor.fetchone()

    connection.close()

    if user_data:
        return UserLogin(
            id=user_data['id'],
            nombre=user_data['nombre'],
            email=user_data['email'],
            rol=user_data['rol'],
            numero_cuenta=user_data['numero_cuenta']
        )

    return None


# Registrar eventos de inicio y cierre de sesión
@app.before_request
def before_request():
    g.user = current_user
    
    # Si el usuario está autenticado y no hay marca de tiempo de inicio de sesión
    if current_user.is_authenticated and 'login_time' not in session:
        session['login_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Modificar el procesador de contexto para cargar la configuración de manera más eficiente
@app.context_processor
def inject_global_vars():
    año_actual = datetime.now().year
    
    # Cargar tema desde configuraciones de manera más eficiente
    tema = 'light'  # Valor por defecto
    config_usuario = {}
    
    if current_user.is_authenticated:
        usuario_id = str(current_user.id)
        # Usar la función cargar_configuracion que ya maneja valores por defecto
        from routes.configuracion import cargar_configuracion
        config_usuario = cargar_configuracion(usuario_id)
        tema = config_usuario.get('tema', 'light')
    
    # Guardar tema en la sesión
    session['tema'] = tema
    
    # Verificar notificaciones para usuarios autenticados
    notificaciones = []
    if current_user.is_authenticated:
        # Verificar prácticas pendientes
        practicas_pendientes = verificar_practicas_pendientes(current_user.id) or []
        if practicas_pendientes:
            notificaciones.append({
                'tipo': 'warning',
                'mensaje': f"Tienes {len(practicas_pendientes)} actividades sin entregar."
            })
        
        # Verificar prácticas por caducar
        practicas_por_caducar = verificar_practicas_por_caducar(current_user.id) or []
        if practicas_por_caducar:
            if len(practicas_por_caducar) == 1:
                notificaciones.append({
                    'tipo': 'danger',
                    'mensaje': f"Una actividad está por caducar: {practicas_por_caducar[0]['titulo']}"
                })
            else:
                notificaciones.append({
                    'tipo': 'danger',
                    'mensaje': f"{len(practicas_por_caducar)} actividades están por caducar."
                })
    
    return {
        'año_actual': año_actual,
        'tema': tema,
        'config_usuario': config_usuario,  # Agregar la configuración completa
        'notificaciones': notificaciones
    }
    

# Modificación del procesador de contexto para garantizar que configuracion siempre esté disponible
@app.context_processor
def inject_configuracion():
    """Inyecta la configuración del usuario en todas las plantillas."""
    try:
        if current_user.is_authenticated:
            usuario_id = str(current_user.id)
            config = cargar_configuracion(usuario_id)
            if not config:  # Si no hay configuración, usar la predeterminada
                config = obtener_configuracion_predeterminada()
            return {'configuracion': config}
        return {'configuracion': obtener_configuracion_predeterminada()}  # Configuración predeterminada para usuarios no autenticados
    except Exception as e:
        print(f"Error al cargar configuración: {str(e)}")
        return {'configuracion': obtener_configuracion_predeterminada()}  # En caso de error, usar configuración predeterminada


@app.teardown_appcontext
def teardown_appcontext(exception=None):
    # Si el usuario estaba autenticado y hay una marca de tiempo de inicio de sesión
    if getattr(g, 'user', None) and g.user.is_authenticated and 'login_time' in session:
        # Limpiar marca de tiempo
        session.pop('login_time', None)

def allowed_file(filename):
    """Verifica si el archivo tiene una extensión permitida."""
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'pdf', 'docx', 'txt'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Conexión a la base de datos
def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',
        user='root',
        password='',
        database='generador_practicas'  # Asegúrate de que el nombre de la base de datos sea correcto
    )
    return connection

@app.route('/create-missing-tables', methods=['GET'])
def create_missing_tables():
    """Create missing tables and columns needed for the student profile."""
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Check if perfil_completado column exists in usuarios table
        cursor.execute("SHOW COLUMNS FROM usuarios LIKE 'perfil_completado'")
        if not cursor.fetchone():
            cursor.execute("ALTER TABLE usuarios ADD COLUMN perfil_completado BOOLEAN DEFAULT 0")
            print("Added perfil_completado column to usuarios table")
        
        # Check if perfiles_estudiante table exists
        cursor.execute("SHOW TABLES LIKE 'perfiles_estudiante'")
        if not cursor.fetchone():
            # Create perfiles_estudiante table
            cursor.execute("""
            CREATE TABLE perfiles_estudiante (
                id INT AUTO_INCREMENT PRIMARY KEY,
                estudiante_id INT NOT NULL,
                semestre INT NOT NULL,
                facultad VARCHAR(100) NOT NULL,
                carrera VARCHAR(100) NOT NULL,
                estilos_aprendizaje VARCHAR(255),
                marco_id INT NULL,
                fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
                fecha_actualizacion DATETIME DEFAULT NULL,
                FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE
            )
            """)
            print("Created perfiles_estudiante table")
        
        connection.commit()
        connection.close()
        
        return "Tables and columns created successfully!"
    except Exception as e:
        return f"Error creating tables: {str(e)}"

def generar_numero_cuenta():
    while True:
        numero_cuenta = random.randint(100000000, 999999999)  # Genera un número de 9 dígitos
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM usuarios WHERE numero_cuenta = %s", (numero_cuenta,))
        existe = cursor.fetchone()[0]
        if existe == 0:  # Si no existe, es único
            return numero_cuenta

def enviar_correo(email, nombre, numero_cuenta, password):
    try:
        msg = Message(
            "Cuenta Aprobada",
            sender="tu_correo@gmail.com",
            recipients=[email]
        )
        msg.body = f"""Hola {nombre},

Tu cuenta ha sido aprobada.

No. de cuenta: {numero_cuenta}
Clave: {password}

Saludos."""
        
        # Forzar UTF-8 en el mensaje
        msg.body = msg.body.encode("utf-8").decode("utf-8")
        msg.charset = "utf-8"  # Asegurar que el correo use UTF-8

        mail.send(msg)
        print("Correo enviado correctamente")
    except Exception as e:
        print(f"Error al enviar correo: {str(e)}")

@app.route('/registro', methods=['GET'])
def registro():
    return render_template('registro.html')

# Modificar la función de registro de usuario para crear configuración por defecto
@app.route('/registro', methods=['POST'])
def registrar_solicitud():
    connection = None  # Inicializar la variable
    try:
        data = request.json
        nombre = data['nombre']
        email = data['email']
        password = data['password']
        rol_solicitado = data['rol']  # Asegúrate de que esto esté presente

        # Generar hash de la contraseña
        password_hash = generate_password_hash(password)

        # Insertar la solicitud en la base de datos
        connection = get_db_connection()  # Establecer la conexión
        cursor = connection.cursor()
        query = """
        INSERT INTO solicitudes_registro (nombre, email, password_hash, rol_solicitado, password)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(query, (nombre, email, password_hash, rol_solicitado, password))  # Guardar la contraseña original
        connection.commit()  # Asegúrate de hacer commit

        return jsonify({"message": "Solicitud enviada correctamente"}), 201
    except Exception as e:
        if connection:  # Verificar si la conexión fue establecida
            connection.rollback()  # Asegúrate de hacer rollback en caso de error
        return jsonify({"error": str(e)}), 500
    finally:
        if connection:  # Cerrar la conexión si fue establecida
            connection.close()

# Función para registrar sesiones de usuario
def registrar_sesion_usuario(accion):
    """Registra una sesión de usuario (inicio o cierre)."""
    if not current_user.is_authenticated:
        return
    
    # Cargar sesiones existentes
    sesiones_path = 'sesiones.json'
    if os.path.exists(sesiones_path):
        with open(sesiones_path, 'r', encoding='utf-8') as f:
            try:
                sesiones = json.load(f)
            except json.JSONDecodeError:
                sesiones = {}
    else:
        sesiones = {}
    
    # Obtener o crear registro para el usuario actual
    usuario_id = str(current_user.id)
    if usuario_id not in sesiones:
        sesiones[usuario_id] = []
    
    # Obtener fecha y hora actual
    ahora = datetime.now()
    fecha = ahora.strftime('%Y-%m-%d')
    hora = ahora.strftime('%H:%M:%S')
    
    # Registrar acción
    if accion == 'login':
        # Registrar inicio de sesión
        sesiones[usuario_id].append({
            'accion': 'login',
            'fecha': fecha,
            'hora': hora,
            'detalles': {
                'ip': request.remote_addr,
                'user_agent': request.headers.get('User-Agent')
            }
        })
    elif accion == 'logout':
        # Buscar el último inicio de sesión
        for i in range(len(sesiones[usuario_id]) - 1, -1, -1):
            if sesiones[usuario_id][i]['accion'] == 'login':
                # Calcular duración
                login_hora = datetime.strptime(sesiones[usuario_id][i]['hora'], '%H:%M:%S')
                logout_hora = datetime.strptime(hora, '%H:%M:%S')
                
                # Si el login fue en un día diferente, ajustar
                if sesiones[usuario_id][i]['fecha'] != fecha:
                    # Simplificación: asumimos que fue el día anterior
                    duracion_segundos = (logout_hora - login_hora).total_seconds() + 24*60*60
                else:
                    duracion_segundos = (logout_hora - login_hora).total_seconds()
                
                # Formatear duración como HH:MM:SS
                horas = int(duracion_segundos // 3600)
                minutos = int((duracion_segundos % 3600) // 60)
                segundos = int(duracion_segundos % 60)
                duracion = f"{horas:02d}:{minutos:02d}:{segundos:02d}"
                
                # Actualizar registro de login con la duración
                sesiones[usuario_id][i]['duracion'] = duracion
                break
        
        # Registrar cierre de sesión
        sesiones[usuario_id].append({
            'accion': 'logout',
            'fecha': fecha,
            'hora': hora
        })
    
    # Guardar sesiones
    with open(sesiones_path, 'w', encoding='utf-8') as f:
        json.dump(sesiones, f, indent=4, ensure_ascii=False)

# Ruta para actualizar solo los estilos de aprendizaje
@app.route('/actualizar-estilos', methods=['GET', 'POST'])
@login_required
def actualizar_estilos():
    """Permite al estudiante actualizar sus estilos de aprendizaje."""
    if current_user.rol != 'estudiante':
        flash('Esta función solo está disponible para estudiantes.', 'error')
        return redirect(url_for('inicio'))
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Obtener datos actuales del estudiante
    query = """
    SELECT estilos_aprendizaje
    FROM perfiles_estudiante
    WHERE estudiante_id = %s
    """
    cursor.execute(query, (current_user.id,))
    perfil = cursor.fetchone()
    
    if not perfil:
        flash('Primero debes completar tu perfil.', 'error')
        return redirect(url_for('perfil'))
    
    estilos_actuales = perfil['estilos_aprendizaje'].split(',') if perfil['estilos_aprendizaje'] else []
    
    if request.method == 'POST':
        # Obtener nuevos estilos de aprendizaje
        estilos_aprendizaje = request.form.getlist('estilos_aprendizaje')
        estilos_str = ','.join(estilos_aprendizaje) if estilos_aprendizaje else ''
        
        # Actualizar en la base de datos
        query_update = """
        UPDATE perfiles_estudiante
        SET estilos_aprendizaje = %s, fecha_actualizacion = NOW()
        WHERE estudiante_id = %s
        """
        cursor.execute(query_update, (estilos_str, current_user.id))
        connection.commit()
        
        # Actualizar en el archivo JSON
        usuario_id = str(current_user.id)
        usuarios = cargar_json('usuarios.json')
        
        if usuario_id in usuarios:
            usuarios[usuario_id]['estilos_aprendizaje'] = estilos_str
            guardar_json('usuarios.json', usuarios)
        
        flash('Estilos de aprendizaje actualizados con éxito.', 'success')
        return redirect(url_for('perfil'))
    
    usuario_id = str(current_user.id)
    usuarios = cargar_json('usuarios.json')  # Cargar el archivo de usuarios

    usuario = usuarios.get(usuario_id, {})  # Obtener el usuarioactual
    connection.close()
    
    año_actual = datetime.now().year
    return render_template('actualizar_estilos.html', estilos_actuales=estilos_actuales, año_actual=año_actual, usuario=usuario)

@app.route('/perfil_estudiante', methods=['GET'])
@login_required
def perfil_estudiante():
    """Muestra el formulario de perfil de estudiante si no ha sido completado."""
    # Verificar si el usuario ya completó su perfil
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT perfil_completado FROM usuarios 
    WHERE id = %s AND rol = 'estudiante'
    """
    cursor.execute(query, (current_user.id,))
    usuario = cursor.fetchone()
    connection.close()
    
    # Si no es estudiante o ya completó su perfil, redirigir al inicio
    if not usuario or usuario.get('perfil_completado'):
        return redirect(url_for('inicio'))
    
    año_actual = datetime.now().year
    return render_template('student_profile.html', año_actual=año_actual)

@app.route('/gestionar_solicitudes', methods=['GET', 'POST'])
def gestionar_solicitudes():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    if request.method == 'GET':
        # Obtener todas las solicitudes pendientes
        query = "SELECT * FROM solicitudes_registro WHERE estado = 'pendiente'"
        cursor.execute(query)
        solicitudes = cursor.fetchall()
        usuario_id = str(current_user.id)
        usuarios = cargar_json('usuarios.json')  # Cargar el archivo de usuarios
        usuario = usuarios.get(usuario_id, {})  # Obtener el usuario actual
        return render_template('solicitudes.html', solicitudes=solicitudes, usuario=usuario)  # Renderizar la plantilla

    if request.method == 'POST':
        data = request.json
        solicitud_id = data['id']
        accion = data['accion']  # 'aprobar' o 'rechazar'

        try:
            if accion == 'aprobar':
                # Lógica para aprobar la solicitud
                query = "UPDATE solicitudes_registro SET estado = 'aprobada' WHERE id = %s"
                cursor.execute(query, (solicitud_id,))
                connection.commit()

                # Obtener los detalles de la solicitud
                cursor.execute("SELECT nombre, email, password_hash, rol_solicitado, password FROM solicitudes_registro WHERE id = %s", (solicitud_id,))
                solicitud = cursor.fetchone()

                # Verificar si la solicitud fue encontrada
                if not solicitud:
                    return jsonify({"error": "Solicitud no encontrada."}), 404

                # Generar un número de cuenta único
                numero_cuenta = generar_numero_cuenta()

                # Insertar el nuevo usuario en la tabla usuarios
                insert_query = """
                INSERT INTO usuarios (nombre, email, password_hash, rol, numero_cuenta)
                VALUES (%s, %s, %s, %s, %s)
                """
                cursor.execute(insert_query, (solicitud['nombre'], solicitud['email'], solicitud['password_hash'], solicitud['rol_solicitado'], numero_cuenta))
                connection.commit()

                # Obtener el ID del usuario recién creado
                cursor.execute("SELECT LAST_INSERT_ID() as id")
                resultado = cursor.fetchone()
                nuevo_usuario_id = resultado['id']

                # Crear configuración por defecto para el nuevo usuario
                crear_configuracion_por_defecto(nuevo_usuario_id)

                # Enviar correo electrónico con el número de cuenta y la contraseña original
                enviar_correo(solicitud['email'], solicitud['nombre'], numero_cuenta, solicitud['password'])  # Usar la contraseña original

                print(f"Solicitud {solicitud_id} aprobada y correo enviado.")  # Mensaje de depuración
                return jsonify({"message": "Solicitud aprobada correctamente"}), 200

            elif accion == 'rechazar':
                # Lógica para rechazar la solicitud
                query = "UPDATE solicitudes_registro SET estado = 'rechazada' WHERE id = %s"
                cursor.execute(query, (solicitud_id,))
                connection.commit()
                print(f"Solicitud {solicitud_id} rechazada.")  # Mensaje de depuración
                return jsonify({"message": "Solicitud rechazada correctamente"}), 200

        except Exception as e:
            connection.rollback()  # Asegúrate de hacer rollback en caso de error
            print(f"Error al actualizar la solicitud: {str(e)}")  # Mensaje de depuración
            return jsonify({"error": str(e)}), 500
        finally:
            connection.close()  # Cerrar la conexión

# Ruta modificada para evaluar entregas automáticamente
@app.route('/api/evaluate', methods=['POST'])
def evaluar_entrega_api():
    """API para evaluar entregas de estudiantes"""
    try:
        # Verificar si hay un archivo en la solicitud
        if 'archivo' not in request.files:
            return jsonify({"error": "No se proporcionó ningún archivo"}), 400
            
        archivo = request.files['archivo']
        if archivo.filename == '':
            return jsonify({"error": "No se seleccionó ningún archivo"}), 400
            
        # Verificar si el archivo es de un tipo permitido
        if not allowed_file(archivo.filename):
            return jsonify({"error": "Tipo de archivo no permitido. Solo se aceptan PDF, DOCX y TXT."}), 400
            
        # Verificar tamaño del archivo
        if request.content_length > APP_CONFIG["MAX_CONTENT_LENGTH"]:
            return jsonify({"error": "El archivo es demasiado grande. El tamaño máximo es 10MB."}), 400
            
        # Obtener datos adicionales del formulario
        practica_id = request.form.get('practica_id')
        estudiante_id = request.form.get('estudiante_id')
        
        if not practica_id or not estudiante_id:
            return jsonify({"error": "Faltan datos requeridos (practica_id o estudiante_id)"}), 400
            
        # Guardar el archivo temporalmente
        filename = secure_filename(archivo.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            archivo.save(filepath)
            print(f"Archivo guardado en {filepath}")
        except Exception as e:
            print(f"Error al guardar archivo: {str(e)}")
            return jsonify({"error": f"Error al guardar archivo: {str(e)}"}), 500
            
        # Leer el contenido del archivo
        contenido = leer_contenido_archivo(filepath)
        
        if not contenido:
            return jsonify({"error": "No se pudo leer el contenido del archivo"}), 400
            
        # Obtener detalles de la práctica
        practica = generador.obtener_practica_por_id(int(practica_id))
        if not practica:
            return jsonify({"error": "Práctica no encontrada"}), 404
            
        # Evaluar el contenido con el modelo ML
        resultado = modelo_ml.analizar_contenido(contenido, practica.titulo, practica.objetivo)
        
        # Generar retroalimentación adicional
        retroalimentacion = generar_retroalimentacion(
            resultado['calificacion'],
            contenido,
            practica.titulo,
            practica.objetivo
        )
        
        # Obtener el ID de la evaluación correspondiente
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        query_eval = """
        SELECT id FROM evaluaciones
        WHERE practica_id = %s AND estudiante_id = %s
        """
        cursor.execute(query_eval, (practica_id, estudiante_id))
        eval_result = cursor.fetchone()
        evaluacion_id = eval_result['id'] if eval_result else None
        connection.close()
        
        # Registrar la entrega en la base de datos
        entrega = {
            'practica_id': int(practica_id),
            'estudiante_id': int(estudiante_id),
            'contenido': contenido,
            'fecha_entrega': datetime.now(),
            'estado': 'entregado',
            'archivos_url': filename,
            'evaluacion_id': evaluacion_id  # Agregar el ID de la evaluación
        }
        
        try:
            # Registrar la entrega con estado 'entregado'
            entrega_id = generador.registrar_entrega(entrega)
            
            # Actualizar la evaluación con la calificación generada
            comentarios_completos = {
                "comentarios": resultado['comentarios'],
                "sugerencias": resultado['sugerencias'],
                "retroalimentacion_positiva": retroalimentacion['positivo'],
                "areas_mejora": retroalimentacion['mejoras'],
                "relevancia": resultado['relevancia']
            }
            
            comentarios_json = json.dumps(comentarios_completos)
            
            query_eval = """
                UPDATE evaluaciones 
                SET calificacion = %s, comentarios = %s, estado = 'calificado' 
                WHERE id = %s
            """
            generador.cursor.execute(query_eval, (
                resultado['calificacion'], 
                comentarios_json, 
                evaluacion_id
            ))
            generador.connection.commit()
            
            print(f"Evaluación actualizada con ID: {evaluacion_id}")
            
        except Exception as e:
            print(f"Error al registrar entrega en la base de datos: {str(e)}")
            return jsonify({"error": f"Error al registrar entrega: {str(e)}"}), 500
        
        # Devolver el resultado de la evaluación
        return jsonify({
            "success": True,
            "evaluacion": {
                "calificacion": resultado['calificacion'],
                "comentarios": resultado['comentarios'],
                "sugerencias": resultado['sugerencias'],
                "relevancia": resultado['relevancia'],
                "retroalimentacion_positiva": retroalimentacion['mejoras'],
                "areas_mejora": retroalimentacion['mejoras']
            }
        }), 200
        
    except Exception as e:
        print(f"Error en la API de evaluación: {str(e)}")
        return jsonify({"error": f"Error al procesar la solicitud: {str(e)}"}), 500

@app.route('/evaluar_entrega_automatica', methods=['POST'])
def evaluar_entrega_automatica():
    """Califica automáticamente una entrega usando la red neuronal."""
    try:
        # Obtener datos de la solicitud
        data = request.json
        evaluacion_id = data.get('evaluacion_id')
        print(f"ID de evaluación recibido: {evaluacion_id}")

        if not evaluacion_id:
            return jsonify({"success": False, "error": "No se proporcionó ID de evaluación"}), 400

        print(f"Evaluando entrega automáticamente para evaluación ID: {evaluacion_id}")

        # Obtener la información de la evaluación
        evaluacion = generador.obtener_evaluacion_por_id(evaluacion_id)

        if not evaluacion:
            return jsonify({"success": False, "error": f"Evaluación con ID {evaluacion_id} no encontrada"}), 404

        print(f"Información de evaluación obtenida: {evaluacion}")

        # Buscar la entrega asociada a esta evaluación
        print(f"Buscando entrega asociada a la evaluación con ID: {evaluacion_id}")
        entrega = generador.obtener_entrega_por_evaluacion(evaluacion_id)
        print(f"Resultado de obtener_entrega_por_evaluacion: {entrega}")

        if not entrega:
            # Buscar entrega por práctica y estudiante
            print(f"Intentando buscar entrega por practica_id: {evaluacion['practica_id']} y estudiante_id: {evaluacion['estudiante_id']}")
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            query = """
                SELECT id, archivos_url, contenido
                FROM entregas
                WHERE practica_id = %s AND estudiante_id = %s
                ORDER BY fecha_entrega DESC
                LIMIT 1
            """
            cursor.execute(query, (evaluacion['practica_id'], evaluacion['estudiante_id']))
            entrega = cursor.fetchone()
            connection.close()

            if not entrega:
                return jsonify({
                    "success": False,
                    "error": f"No se encontró una entrega asociada a la evaluación con ID: {evaluacion_id}"
                }), 404

        # Leer el contenido del archivo de texto
        archivo_url = entrega['archivos_url']
        ruta_archivo = os.path.join(app.config['UPLOAD_FOLDER'], archivo_url)

        if not os.path.exists(ruta_archivo):
            return jsonify({"success": False, "error": f"Archivo no encontrado: {archivo_url}"}), 404

        contenido_entrega = leer_contenido_archivo(ruta_archivo)

        if not contenido_entrega:
            return jsonify({"success": False, "error": "No se pudo leer el contenido del archivo"}), 400

        print("Contenido del archivo de texto:")
        print(contenido_entrega[:200] + "..." if len(contenido_entrega) > 200 else contenido_entrega)

        # Obtener estilo de aprendizaje si está disponible
        estilo_aprendizaje = None
        try:
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            query_estilo = """
                SELECT estilos_aprendizaje 
                FROM perfiles_estudiante 
                WHERE estudiante_id = %s
            """
            cursor.execute(query_estilo, (evaluacion['estudiante_id'],))
            resultado_estilo = cursor.fetchone()
            connection.close()

            if resultado_estilo and resultado_estilo['estilos_aprendizaje']:
                estilo_aprendizaje = resultado_estilo['estilos_aprendizaje']
                print(f"Estilo de aprendizaje detectado: {estilo_aprendizaje}")

        except Exception as e:
            print(f"Error al obtener estilo de aprendizaje: {str(e)}")

        # Generar calificación y comentarios usando el modelo ML
        resultado_ml = modelo_ml.analizar_contenido(
            contenido_entrega,
            evaluacion['practica_titulo'],
            evaluacion.get('practica_objetivo', ''),
            estilo_aprendizaje
        )

        calificacion = resultado_ml['calificacion']
        comentarios = resultado_ml['comentarios']

        print("Calificación y comentarios generados:")
        print("Calificación:", calificacion)
        print("Comentarios:", comentarios)

        # Actualizar evaluación
        generador.actualizar_evaluacion(evaluacion_id, comentarios, calificacion)

        # Actualizar entrega
        connection = get_db_connection()
        cursor = connection.cursor()
        query_update_entrega = """
            UPDATE entregas
            SET evaluacion_id = %s, estado = 'calificado'
            WHERE id = %s
        """
        cursor.execute(query_update_entrega, (evaluacion_id, entrega['id']))
        connection.commit()
        connection.close()

        return jsonify({
            "success": True,
            "calificacion": calificacion,
            "comentarios": comentarios
        }), 200

    except Exception as e:
        print(f"Error en evaluar_entrega_automatica: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route('/entrenar_modelo', methods=['POST'])
def entrenar_modelo():
    try:
        # Datos de entrenamiento
        entradas = [
            "Crear una base de datos relacional con MySQL.",
            "Diseñar un sistema de gestión de inventarios.",
            "Escribir consultas SQL para manipular datos en una base de datos.",
            "Crear tablas y relaciones en una base de datos relacional.",
            "Optimizar consultas SQL para mejorar el rendimiento.",
            "Resolver ecuaciones diferenciales de primer orden.",
            "Desarrollar una aplicación web con Django.",
            "Analizar datos utilizando pandas y matplotlib.",
            "Diseñar un circuito electrónico para un sensor de temperatura.",
            "Escribir un ensayo sobre la literatura del siglo XX.",
            "Estudiar las propiedades químicas de los compuestos orgánicos.",
            "Crear un modelo de predicción utilizando machine learning.",
            "Implementar un sistema de autenticación con JWT.",
            "Realizar un análisis estadístico de datos financieros.",
            "Desarrollar un videojuego 2D utilizando Unity.",
            "Diseñar un experimento para medir la velocidad de reacción química.",
            "Crear un chatbot utilizando procesamiento de lenguaje natural."
        ]

        salidas = [
            9.0, 8.5, 9.5, 9.0, 8.5, 8.0, 9.5, 8.0, 8.5, 7.0, 8.0, 9.0, 8.5, 8.0, 9.0, 8.5, 9.0
        ]

        # Entrenar el modelo
        generador.modelo_ml.entrenar_modelo(entradas, salidas, epochs=10, batch_size=4)

        return jsonify({"success": True, "message": "Modelo entrenado correctamente."}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Función para leer diferentes tipos de archivos
def leer_contenido_archivo(filepath):
    """
    Lee el contenido de un archivo según su tipo.
    Soporta: .txt, .pdf, .docx
    """
    try:
        # Determine file type by extension
        extension = os.path.splitext(filepath)[1].lower()
        
        if extension == '.txt':
            # Read text file
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
                
        elif extension == '.pdf':
            try:
                # Use PyPDF2 to read PDF
                from PyPDF2 import PdfReader
                reader = PdfReader(filepath)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                return text
            except ImportError:
                print("PyPDF2 no está instalado. No se puede leer el archivo PDF.")
                return "No se pudo leer el contenido del archivo PDF. PyPDF2 no está instalado."
            
        elif extension == '.docx':
            try:
                # Use python-docx to read DOCX
                import docx
                doc = docx.Document(filepath)
                return " ".join([paragraph.text for paragraph in doc.paragraphs])
            except ImportError:
                print("python-docx no está instalado. No se puede leer el archivo DOCX.")
                return "No se pudo leer el contenido del archivo DOCX. python-docx no está instalado."
            
        else:
            # For unsupported file types, try to read as text
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
                
    except Exception as e:
        print(f"Error al leer el archivo {filepath}: {str(e)}")
        return ""

@app.route('/contacto', methods=['GET', 'POST'])
def contacto():
    if request.method == 'POST':
        # Aquí se procesaría el formulario de contacto
        flash('Tu mensaje ha sido enviado. Te contactaremos pronto.', 'success')
        return redirect(url_for('contacto'))
    
    # Obtener la configuración del usuario actual
    usuario_id = str(current_user.id)
    # Obtener información del usuario desde el archivo JSON
    usuarios = cargar_json('usuarios.json')  # Cargar el archivo de usuarios
    usuario = usuarios.get(usuario_id, {})  # Obtener el usuario actual
    año_actual = datetime.now().year
    return render_template('contacto.html', año_actual=año_actual, usuario=usuario)

@app.route('/ayuda')
def ayuda():
    # Obtener la configuración del usuario actual
    usuario_id = str(current_user.id)
    # Obtener información del usuario desde el archivo JSON
    usuarios = cargar_json('usuarios.json')  # Cargar el archivo de usuarios
    usuario = usuarios.get(usuario_id, {})  # Obtener el usuario actual
    año_actual = datetime.now().year
    return render_template('ayuda.html', año_actual=año_actual, usuario=usuario)

@app.route('/privacidad')
def privacidad():
    # Obtener la configuración del usuario actual
    usuario_id = str(current_user.id)
    # Obtener información del usuario desde el archivo JSON
    usuarios = cargar_json('usuarios.json')  # Cargar el archivo de usuarios
    usuario = usuarios.get(usuario_id, {})  # Obtener el usuario actual
    año_actual = datetime.now().year
    return render_template('privacidad.html', año_actual=año_actual, usuario=usuario)

@app.route('/sobre-nosotros')
def sobre_nosotros():
    # Obtener informacion de las personas que desarrollaron el sistema
    creditos = cargar_json('creditos.json')
    return render_template('sobre_nosotros.html', equipo = creditos.get("creditos", []))

@app.route('/cambiar-password', methods=['POST'])
@login_required
def cambiar_password():
    # Obtener el formulario de cambio de contraseña
    password_actual = request.form.get('password_actual')
    password_nueva = request.form.get('password_nueva')
    password_confirmacion = request.form.get('password_confirmacion')

    # Cargar configuraciones existentes
    configuraciones_path = 'configuraciones.json'
    
    if os.path.exists(configuraciones_path):
        with open(configuraciones_path, 'r') as f:
            configuraciones = json.load(f)
    else:
        configuraciones = {}

    # Obtener la configuración del usuario actual
    usuario_id = str(current_user.id)  # Usar el ID del usuario como clave
    configuracion = configuraciones.get(usuario_id, {})  # Obtener la configuración del usuario o un diccionario vacío

    # Verificar la contraseña actual
    #if not current_user.check_password(password_actual):
    #    flash('La contraseña actual es incorrecta.', 'danger')
    #    return redirect(url_for('perfil'))  # Redirigir al perfil

    # Verificar que la nueva contraseña y la confirmación coincidan
    if password_nueva != password_confirmacion:
        flash('Las contraseñas no coinciden.', 'danger')
        return redirect(url_for('perfil'))  # Redirigir al perfil

    # Establecer la nueva contraseña
    #current_user.set_password(password_nueva)

    # Guardar la nueva contraseña en el archivo JSON
    configuracion['password_hash'] = generate_password_hash(password_nueva)  # Guardar el hash de la nueva contraseña

    # Guardar configuraciones en el archivo JSON
    with open(configuraciones_path, 'w') as f:
        json.dump(configuraciones, f, indent=4)

    flash('Tu contraseña ha sido actualizada.', 'success')
    return redirect(url_for('perfil'))  # Redirigir al perfil

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

def guardar_json(path, data):
    """Guarda datos en un archivo JSON."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def actualizar_configuracion(usuario_id, nuevos_datos):
    """Actualiza la configuración de un usuario en el archivo JSON."""
    configuraciones_path = 'configuraciones.json'
    configuraciones = cargar_json(configuraciones_path)

    # Actualizar la configuración del usuario
    if usuario_id in configuraciones:
        configuraciones[usuario_id].update(nuevos_datos)
    else:
        configuraciones[usuario_id] = nuevos_datos  # Si no existe, crear una nueva entrada

    # Guardar los cambios
    guardar_json(configuraciones_path, configuraciones)

def agregar_usuario(usuario_id, datos_usuario):
    """Agrega un nuevo usuario al archivo usuarios.json."""
    usuarios_path = 'usuarios.json'
    usuarios = cargar_json(usuarios_path)

    # Verificar si el usuario ya existe
    if usuario_id not in usuarios:
        usuarios[usuario_id] = datos_usuario
        guardar_json(usuarios_path, usuarios)
        print(f"Usuario {usuario_id} agregado exitosamente.")
    else:
        print(f"El usuario {usuario_id} ya existe.")

# Ruta para mostrar el perfil del usuario
def agregar_usuario(usuario_id, datos_usuario):
    """Agrega un nuevo usuario al archivo usuarios.json."""
    usuarios_path = 'usuarios.json'
    usuarios = cargar_json(usuarios_path)

    # Verificar si el usuario ya existe
    if usuario_id not in usuarios:
        usuarios[usuario_id] = datos_usuario
        guardar_json(usuarios_path, usuarios)
        print(f"Usuario {usuario_id} agregado exitosamente.")
    else:
        print(f"El usuario {usuario_id} ya existe.")

# Función para calcular el promedio de un estudiante basado en sus evaluaciones
def calcular_promedio_estudiante(estudiante_id):
    """Calcula el promedio de un estudiante basado en sus evaluaciones de clases."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Consulta para obtener todas las evaluaciones del estudiante
    query = """
    SELECT AVG(calificacion) as promedio_general
    FROM evaluaciones
    WHERE estudiante_id = %s AND calificacion IS NOT NULL
    """
    cursor.execute(query, (estudiante_id,))
    resultado = cursor.fetchone()
    connection.close()
    
    # Si no hay evaluaciones, devolver 0
    if not resultado or resultado['promedio_general'] is None:
        return 0.0
    
    return float(resultado['promedio_general'])

# Función para obtener promedios por clase
def obtener_promedios_por_clase(estudiante_id):
    """
    Obtiene los promedios del estudiante por cada grupo/clase.
    Versión corregida que usa grupos en lugar de clases.
    """
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        # Consulta corregida que usa grupos en lugar de clases
        query = """
        SELECT g.id as grupo_id, g.nombre as grupo_nombre, 
               IFNULL(cf.calificacion_final, 0) as promedio
        FROM grupos g
        LEFT JOIN grupo_estudiante ge ON g.id = ge.grupo_id
        LEFT JOIN calificaciones_finales cf ON g.id = cf.grupo_id AND cf.estudiante_id = %s
        WHERE ge.estudiante_id = %s
        """
        
        cursor.execute(query, (estudiante_id, estudiante_id))
        promedios = cursor.fetchall()
        
        return promedios
    except Exception as e:
        print(f"Error al obtener promedios por clase: {e}")
        return []
    finally:
        cursor.close()
        connection.close()

# Función para verificar si un estudiante puede editar su perfil
def puede_editar_perfil(estudiante_id):
    """Verifica si un estudiante puede editar su perfil basado en el límite de ediciones."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Obtener la fecha actual y la fecha hace 6 meses
    fecha_actual = datetime.now()
    fecha_hace_seis_meses = fecha_actual - timedelta(days=180)
    
    # Consulta para contar ediciones en los últimos 6 meses
    query = """
    SELECT COUNT(*) as num_ediciones
    FROM ediciones_perfil
    WHERE estudiante_id = %s AND fecha_edicion > %s
    """
    cursor.execute(query, (estudiante_id, fecha_hace_seis_meses))
    resultado = cursor.fetchone()
    connection.close()
    
    # Si hay menos de 3 ediciones, puede editar
    return resultado['num_ediciones'] < 3 if resultado else True

# Función para registrar una edición de perfil
def registrar_edicion_perfil(estudiante_id):
    """Registra una edición de perfil en la base de datos."""
    connection = get_db_connection()
    cursor = connection.cursor()
    
    # Insertar registro de edición
    query = """
    INSERT INTO ediciones_perfil (estudiante_id, fecha_edicion)
    VALUES (%s, NOW())
    """
    cursor.execute(query, (estudiante_id,))
    connection.commit()
    connection.close()

# Función para obtener los marcos disponibles para un estudiante
def obtener_marcos_disponibles(estudiante_id):
    """Obtiene los marcos que ha desbloqueado un estudiante."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Consulta para obtener marcos desbloqueados
    query = """
    SELECT m.id, m.nombre, m.descripcion, m.imagen_url, m.clase_css
    FROM marcos_perfil m
    JOIN marcos_desbloqueados md ON m.id = md.marco_id
    WHERE md.estudiante_id = %s
    """
    cursor.execute(query, (estudiante_id,))
    marcos = cursor.fetchall()
    
    # Si no hay marcos desbloqueados, obtener al menos el marco básico
    if not marcos:
        query_basico = """
        SELECT id, nombre, descripcion, imagen_url, clase_css
        FROM marcos_perfil
        WHERE id = 1  -- Marco básico
        """
        cursor.execute(query_basico)
        marcos = cursor.fetchall()
        
        # Desbloquear automáticamente el marco básico
        query_insert = """
        INSERT IGNORE INTO marcos_desbloqueados (estudiante_id, marco_id, fecha_desbloqueo)
        VALUES (%s, 1, NOW())
        """
        cursor.execute(query_insert, (estudiante_id,))
        connection.commit()
    
    # Consulta para obtener el marco actualmente seleccionado
    query_seleccionado = """
    SELECT marco_id
    FROM perfiles_estudiante
    WHERE estudiante_id = %s
    """
    cursor.execute(query_seleccionado, (estudiante_id,))
    seleccionado = cursor.fetchone()
    
    connection.close()
    
    # Marcar el marco seleccionado
    marco_seleccionado = seleccionado['marco_id'] if seleccionado and seleccionado['marco_id'] else None
    for marco in marcos:
        marco['seleccionado'] = (marco['id'] == marco_seleccionado)
    
    return marcos

# Función para verificar y desbloquear marcos basados en logros
def verificar_desbloqueo_marcos(estudiante_id):
    """Verifica y desbloquea marcos basados en los logros del estudiante."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Obtener datos del estudiante
    query_estudiante = """
    SELECT semestre, fecha_creacion
    FROM perfiles_estudiante
    WHERE estudiante_id = %s
    """

    cursor.execute(query_estudiante, (estudiante_id,))
    perfil = cursor.fetchone()
    
    if not perfil:
        connection.close()
        return
    
    # Verificar marco por semestre
    semestre = int(perfil['semestre']) if perfil['semestre'] else 0
    for i in range(1, semestre + 1):
        # Verificar si ya tiene el marco desbloqueado
        query_check = """
        SELECT 1 FROM marcos_desbloqueados 
        WHERE estudiante_id = %s AND marco_id = %s
        """
        cursor.execute(query_check, (estudiante_id, i))
        ya_desbloqueado = cursor.fetchone()
        
        if not ya_desbloqueado:
            # Desbloquear marco por semestre
            query_insert = """
            INSERT INTO marcos_desbloqueados (estudiante_id, marco_id, fecha_desbloqueo)
            VALUES (%s, %s, NOW())
            """
            cursor.execute(query_insert, (estudiante_id, i))
    
    # Verificar marco por mejor promedio
    promedio = calcular_promedio_estudiante(estudiante_id)
    if promedio >= 9.5:  # Marco de excelencia académica
        query_check = """
        SELECT 1 FROM marcos_desbloqueados 
        WHERE estudiante_id = %s AND marco_id = %s
        """
        marco_id = 10  # ID del marco de excelencia
        cursor.execute(query_check, (estudiante_id, marco_id))
        ya_desbloqueado = cursor.fetchone()
        
        if not ya_desbloqueado:
            query_insert = """
            INSERT INTO marcos_desbloqueados (estudiante_id, marco_id, fecha_desbloqueo)
            VALUES (%s, %s, NOW())
            """
            cursor.execute(query_insert, (estudiante_id, marco_id))
    
    # Verificar marco por tareas al día (sin tareas atrasadas por 3 meses)
    fecha_hace_tres_meses = datetime.now() - timedelta(days=90)
    query_tareas = """
    SELECT COUNT(*) as tareas_atrasadas
    FROM practicas p
    JOIN asignaciones a ON p.id = a.practica_id
    WHERE a.estudiante_id = %s 
    AND a.fecha_entrega < %s 
    AND a.estado = 'pendiente'
    """
    cursor.execute(query_tareas, (estudiante_id, fecha_hace_tres_meses))
    tareas = cursor.fetchone()
    
    if tareas and tareas['tareas_atrasadas'] == 0:
        query_check = """
        SELECT 1 FROM marcos_desbloqueados 
        WHERE estudiante_id = %s AND marco_id = %s
        """
        marco_id = 11  # ID del marco de responsabilidad
        cursor.execute(query_check, (estudiante_id, marco_id))
        ya_desbloqueado = cursor.fetchone()
        
        if not ya_desbloqueado:
            query_insert = """
            INSERT INTO marcos_desbloqueados (estudiante_id, marco_id, fecha_desbloqueo)
            VALUES (%s, %s, NOW())
            """
            cursor.execute(query_insert, (estudiante_id, marco_id))
    
    connection.commit()
    connection.close()

# Ruta para mostrar y actualizar el perfil del usuario
@app.route('/perfil', methods=['GET', 'POST'])
@login_required
def perfil():
    """Muestra y actualiza el perfil del usuario según su rol."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    if current_user.rol == 'estudiante':
        # Obtener datos del estudiante
        query = """
        SELECT u.*, pe.semestre, pe.facultad, pe.carrera, pe.estilos_aprendizaje, pe.marco_id
        FROM usuarios u
        LEFT JOIN perfiles_estudiante pe ON u.id = pe.estudiante_id
        WHERE u.id = %s
        """
        cursor.execute(query, (current_user.id,))
        usuario = cursor.fetchone()
        
        # Verificar desbloqueo de marcos
        verificar_desbloqueo_marcos(current_user.id)
        
        # Obtener marcos disponibles
        marcos_disponibles = obtener_marcos_disponibles(current_user.id)
        
        # Calcular promedio
        promedio = calcular_promedio_estudiante(current_user.id)
        promedios_por_clase = obtener_promedios_por_clase(current_user.id)
        
        # Contar prácticas asignadas
        query_practicas = """
        SELECT COUNT(*) as total
        FROM asignaciones
        WHERE estudiante_id = %s
        """
        cursor.execute(query_practicas, (current_user.id,))
        resultado = cursor.fetchone()
        total_practicas = resultado['total'] if resultado else 0
        
        # Procesar formulario de actualización
        if request.method == 'POST':
            if not puede_editar_perfil(current_user.id):
                flash('Has alcanzado el límite de 3 ediciones de perfil en este semestre.', 'error')
                return redirect(url_for('perfil'))
            
            # Actualizar solo los campos permitidos
            email = request.form.get('email')
            telefono = request.form.get('telefono')
            marco_id = request.form.get('marco_id')
            
            # Actualizar foto si se proporciona
            if 'foto' in request.files and request.files['foto'].filename:
                foto = request.files['foto']
                if foto and allowed_file(foto.filename):
                    filename = secure_filename(foto.filename)
                    # Generar nombre único para evitar colisiones
                    filename = f"{current_user.id}_{int(time.time())}_{filename}"
                    filepath = os.path.join('static/uploads', filename)
                    
                    # Asegurar que el directorio existe
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    foto.save(filepath)
                    
                    # Actualizar ruta de la foto en usuarios.json
                    usuario_id = str(current_user.id)
                    usuarios = cargar_json('usuarios.json')
                    
                    if usuario_id not in usuarios:
                        usuarios[usuario_id] = {}
                    
                    usuarios[usuario_id]['foto_url'] = f"uploads/{filename}"
                    guardar_json('usuarios.json', usuarios)
                else:
                    flash('Formato de archivo no permitido. Use JPG, PNG o GIF.', 'error')
            
            # Actualizar datos básicos en usuarios.json
            usuario_id = str(current_user.id)
            usuarios = cargar_json('usuarios.json')
            
            if usuario_id not in usuarios:
                usuarios[usuario_id] = {}
            
            usuarios[usuario_id]['nombre'] = usuario['nombre']
            usuarios[usuario_id]['email'] = email
            usuarios[usuario_id]['numero_cuenta'] = usuario.get('numero_cuenta', '')
            usuarios[usuario_id]['telefono'] = telefono
            usuarios[usuario_id]['facultad'] = usuario.get('facultad', '')
            usuarios[usuario_id]['carrera'] = usuario.get('carrera', '')
            usuarios[usuario_id]['semestre'] = usuario.get('semestre', '')
            usuarios[usuario_id]['estilos_aprendizaje'] = usuario.get('estilos_aprendizaje', '')
            
            guardar_json('usuarios.json', usuarios)
            
            # Si seleccionó un marco, actualizarlo
            if marco_id:
                query_marco = "UPDATE perfiles_estudiante SET marco_id = %s WHERE estudiante_id = %s"
                cursor.execute(query_marco, (marco_id, current_user.id))
            
            # Registrar la edición del perfil
            registrar_edicion_perfil(current_user.id)
            
            connection.commit()
            flash('Perfil actualizado con éxito.', 'success')
            return redirect(url_for('perfil'))
        
        # Obtener el número de ediciones restantes
        ediciones_restantes = 3
        fecha_hace_seis_meses = datetime.now() - timedelta(days=180)
        query = """
        SELECT COUNT(*) as num_ediciones
        FROM ediciones_perfil
        WHERE estudiante_id = %s AND fecha_edicion > %s
        """
        cursor.execute(query, (current_user.id, fecha_hace_seis_meses))
        resultado = cursor.fetchone()
        
        ediciones_restantes = 3 - (resultado['num_ediciones'] if resultado else 0)
        
        connection.close()
        
        # Cargar datos del usuario desde usuarios.json
        usuario_id = str(current_user.id)
        usuarios = cargar_json('usuarios.json')
        usuario_json = usuarios.get(usuario_id, {})
        
        # Combinar datos de la base de datos y JSON
        if usuario:
            usuario_combinado = dict(usuario)
        else:
            usuario_combinado = {}
            
        # Actualizar con datos del JSON
        for key, value in usuario_json.items():
            usuario_combinado[key] = value
        
        return render_template(
            'perfil.html', 
            usuario=usuario_combinado, 
            promedio=round(promedio, 2), 
            promedios_por_clase=promedios_por_clase,
            total_practicas=total_practicas,
            marcos_disponibles=marcos_disponibles,
            ediciones_restantes=ediciones_restantes,
            año_actual=datetime.now().year,
            semestre=usuario.get('semestre', 1) if usuario else 1,
            rol=current_user.rol
        )
    
    elif current_user.rol == 'profesor':
        # Obtener datos del profesor
        query = """
        SELECT u.*, pp.departamento, pp.especialidad, pp.grado_academico
        FROM usuarios u
        LEFT JOIN perfiles_profesor pp ON u.id = pp.profesor_id
        WHERE u.id = %s
        """
        cursor.execute(query, (current_user.id,))
        usuario = cursor.fetchone()
        
        # Obtener estadísticas del profesor
        # Contar grupos activos
        query_grupos = """
        SELECT COUNT(*) as total
        FROM grupos
        WHERE profesor_id = %s
        """
        cursor.execute(query_grupos, (current_user.id,))
        resultado = cursor.fetchone()
        total_grupos = resultado['total'] if resultado else 0
        
        # Contar estudiantes en sus grupos
        query_estudiantes = """
        SELECT COUNT(DISTINCT gm.usuario_id) as total
        FROM grupo_miembros gm
        JOIN grupos g ON gm.grupo_id = g.id
        WHERE g.profesor_id = %s
        """

        cursor.execute(query_estudiantes, (current_user.id,))
        resultado = cursor.fetchone()
        total_estudiantes = resultado['total'] if resultado else 0
        
        # Contar prácticas creadas
        query_practicas = """
        SELECT COUNT(*) as total
        FROM practicas
        WHERE autor_id = %s
        """
        cursor.execute(query_practicas, (current_user.id,))
        resultado = cursor.fetchone()
        total_practicas = resultado['total'] if resultado else 0
        
        # Procesar formulario de actualización
        if request.method == 'POST':
            # Actualizar datos básicos
            email = request.form.get('email')
            telefono = request.form.get('telefono')
            departamento = request.form.get('departamento', '')
            especialidad = request.form.get('especialidad', '')
            grado_academico = request.form.get('grado_academico', '')
            
            # Actualizar foto si se proporciona
            if 'foto' in request.files and request.files['foto'].filename:
                foto = request.files['foto']
                if foto and allowed_file(foto.filename):
                    filename = secure_filename(foto.filename)
                    # Generar nombre único para evitar colisiones
                    filename = f"{current_user.id}_{int(time.time())}_{filename}"
                    filepath = os.path.join('static/uploads', filename)
                    
                    # Asegurar que el directorio existe
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    foto.save(filepath)
                    
                    # Actualizar ruta de la foto en usuarios.json
                    usuario_id = str(current_user.id)
                    usuarios = cargar_json('usuarios.json')
                    
                    if usuario_id not in usuarios:
                        usuarios[usuario_id] = {}
                    
                    usuarios[usuario_id]['foto_url'] = f"uploads/{filename}"
                    guardar_json('usuarios.json', usuarios)
                else:
                    flash('Formato de archivo no permitido. Use JPG, PNG o GIF.', 'error')
            
            # Actualizar datos básicos en usuarios.json
            usuario_id = str(current_user.id)
            usuarios = cargar_json('usuarios.json')
            
            if usuario_id not in usuarios:
                usuarios[usuario_id] = {}
            
            usuarios[usuario_id]['nombre'] = usuario['nombre']
            usuarios[usuario_id]['email'] = email
            usuarios[usuario_id]['numero_cuenta'] = usuario.get('numero_cuenta', '')
            usuarios[usuario_id]['telefono'] = telefono
            usuarios[usuario_id]['departamento'] = departamento
            usuarios[usuario_id]['especialidad'] = especialidad
            usuarios[usuario_id]['grado_academico'] = grado_academico
            
            guardar_json('usuarios.json', usuarios)
            
            # Actualizar o crear perfil de profesor
            cursor.execute("""
                SELECT 1 FROM perfiles_profesor WHERE profesor_id = %s
            """, (current_user.id,))
            
            if cursor.fetchone():
                # Actualizar perfil existente
                query_perfil = """
                UPDATE perfiles_profesor 
                SET departamento = %s, especialidad = %s, grado_academico = %s, fecha_actualizacion = NOW()
                WHERE profesor_id = %s
                """
                cursor.execute(query_perfil, (departamento, especialidad, grado_academico, current_user.id))
            else:
                # Crear nuevo perfil
                query_perfil = """
                INSERT INTO perfiles_profesor 
                (profesor_id, departamento, especialidad, grado_academico, fecha_creacion)
                VALUES (%s, %s, %s, %s, NOW())
                """
                cursor.execute(query_perfil, (current_user.id, departamento, especialidad, grado_academico))
            
            connection.commit()
            flash('Perfil actualizado con éxito.', 'success')
            return redirect(url_for('perfil'))
        
        connection.close()
        
        # Cargar datos del usuario desde usuarios.json
        usuario_id = str(current_user.id)
        usuarios = cargar_json('usuarios.json')
        usuario_json = usuarios.get(usuario_id, {})
        
        # Combinar datos de la base de datos y JSON
        if usuario:
            usuario_combinado = dict(usuario)
        else:
            usuario_combinado = {
                'id': current_user.id,
                'nombre': current_user.nombre,
                'email': current_user.email,
                'rol': current_user.rol,
                'numero_cuenta': getattr(current_user, 'numero_cuenta', ''),
                'departamento': '',
                'especialidad': '',
                'grado_academico': ''
            }
            
        # Actualizar con datos del JSON
        for key, value in usuario_json.items():
            usuario_combinado[key] = value
        
        return render_template(
            'perfil.html', 
            usuario=usuario_combinado, 
            total_grupos=total_grupos,
            total_estudiantes=total_estudiantes,
            total_practicas=total_practicas,
            año_actual=datetime.now().year,
            rol=current_user.rol
        )
    
    else:  # Administrador
        # Obtener datos del administrador
        query = """
        SELECT u.*, pa.departamento, pa.cargo
        FROM usuarios u
        LEFT JOIN perfiles_administrador pa ON u.id = pa.admin_id
        WHERE u.id = %s
        """
        cursor.execute(query, (current_user.id,))
        usuario = cursor.fetchone()
        
        # Obtener estadísticas del sistema
        # Contar usuarios por rol
        query_usuarios = """
        SELECT rol, COUNT(*) as total
        FROM usuarios
        GROUP BY rol
        """
        cursor.execute(query_usuarios)
        usuarios_por_rol = cursor.fetchall()
        
        # Contar grupos activos
        query_grupos = """
        SELECT COUNT(*) as total
        FROM grupos
        """
        cursor.execute(query_grupos)
        resultado = cursor.fetchone()
        total_grupos = resultado['total'] if resultado else 0
        
        # Procesar formulario de actualización
        if request.method == 'POST':
            # Actualizar datos básicos
            email = request.form.get('email')
            telefono = request.form.get('telefono')
            departamento = request.form.get('departamento', '')
            cargo = request.form.get('cargo', '')
            
            # Actualizar foto si se proporciona
            if 'foto' in request.files and request.files['foto'].filename:
                foto = request.files['foto']
                if foto and allowed_file(foto.filename):
                    filename = secure_filename(foto.filename)
                    # Generar nombre único para evitar colisiones
                    filename = f"{current_user.id}_{int(time.time())}_{filename}"
                    filepath = os.path.join('static/uploads', filename)
                    
                    # Asegurar que el directorio existe
                    os.makedirs(os.path.dirname(filepath), exist_ok=True)
                    
                    foto.save(filepath)
                    
                    # Actualizar ruta de la foto en usuarios.json
                    usuario_id = str(current_user.id)
                    usuarios = cargar_json('usuarios.json')
                    
                    if usuario_id not in usuarios:
                        usuarios[usuario_id] = {}
                    
                    usuarios[usuario_id]['foto_url'] = f"uploads/{filename}"
                    guardar_json('usuarios.json', usuarios)
                else:
                    flash('Formato de archivo no permitido. Use JPG, PNG o GIF.', 'error')
            
            # Actualizar datos básicos en usuarios.json
            usuario_id = str(current_user.id)
            usuarios = cargar_json('usuarios.json')
            
            if usuario_id not in usuarios:
                usuarios[usuario_id] = {}
            
            usuarios[usuario_id]['nombre'] = usuario['nombre']
            usuarios[usuario_id]['email'] = email
            usuarios[usuario_id]['numero_cuenta'] = usuario.get('numero_cuenta', '')
            usuarios[usuario_id]['telefono'] = telefono
            usuarios[usuario_id]['departamento'] = departamento
            usuarios[usuario_id]['cargo'] = cargo
            
            guardar_json('usuarios.json', usuarios)
            
            # Actualizar o crear perfil de administrador
            cursor.execute("""
                SELECT 1 FROM perfiles_administrador WHERE admin_id = %s
            """, (current_user.id,))
            
            if cursor.fetchone():
                # Actualizar perfil existente
                query_perfil = """
                UPDATE perfiles_administrador 
                SET departamento = %s, cargo = %s, fecha_actualizacion = NOW()
                WHERE admin_id = %s
                """
                cursor.execute(query_perfil, (departamento, cargo, current_user.id))
            else:
                # Crear nuevo perfil
                query_perfil = """
                INSERT INTO perfiles_administrador 
                (admin_id, departamento, cargo, fecha_creacion)
                VALUES (%s, %s, %s, NOW())
                """
                cursor.execute(query_perfil, (current_user.id, departamento, cargo))
            
            connection.commit()
            flash('Perfil actualizado con éxito.', 'success')
            return redirect(url_for('perfil'))
        
        connection.close()
        
        # Cargar datos del usuario desde usuarios.json
        usuario_id = str(current_user.id)
        usuarios = cargar_json('usuarios.json')
        usuario_json = usuarios.get(usuario_id, {})
        
        # Combinar datos de la base de datos y JSON
        if usuario:
            usuario_combinado = dict(usuario)
        else:
            usuario_combinado = {
                'id': current_user.id,
                'nombre': current_user.nombre,
                'email': current_user.email,
                'rol': current_user.rol,
                'numero_cuenta': getattr(current_user, 'numero_cuenta', ''),
                'departamento': '',
                'cargo': ''
            }
            
        # Actualizar con datos del JSON
        for key, value in usuario_json.items():
            usuario_combinado[key] = value
        
        # Actividades recientes (ejemplo)
        actividades_recientes = [
            {
                'icono': 'user-plus',
                'titulo': 'Nuevo usuario registrado',
                'detalles': 'Juan Pérez (Estudiante)',
                'tiempo': 'Hace 2 horas'
            },
            {
                'icono': 'edit',
                'titulo': 'Materia actualizada',
                'detalles': 'Algoritmos y Estructuras de Datos',
                'tiempo': 'Hace 5 horas'
            },
            {
                'icono': 'trash-alt',
                'titulo': 'Grupo eliminado',
                'detalles': 'Grupo A-03 (Cálculo Diferencial)',
                'tiempo': 'Ayer'
            },
            {
                'icono': 'cog',
                'titulo': 'Configuración actualizada',
                'detalles': 'Parámetros de evaluación',
                'tiempo': 'Hace 2 días'
            }
        ]
        
        return render_template(
            'perfil.html', 
            usuario=usuario_combinado, 
            usuarios_por_rol=usuarios_por_rol,
            total_grupos=total_grupos,
            actividades_recientes=actividades_recientes,
            año_actual=datetime.now().year,
            rol=current_user.rol
        )

@app.route('/api/evaluaciones/<int:practica_id>', methods=['GET'])
@login_required
def get_evaluaciones(practica_id):
    """
    Endpoint para obtener las evaluaciones de una práctica específica.
    """
    try:
        # Obtener los estudiantes (evaluaciones) para esta práctica
        estudiantes = generador.obtener_estudiantes_por_practica(practica_id)
        
        # Obtener información de la práctica
        practica = generador.obtener_practica_por_id(practica_id)
        
        if not practica:
            return jsonify({"error": "Práctica no encontrada"}), 404
        
        # Formatear las evaluaciones para la respuesta
        evaluaciones = []
        for estudiante in estudiantes:
            # Convertir la fecha a formato ISO para JSON
            fecha_entrega = practica.fecha_entrega.isoformat() if hasattr(practica, 'fecha_entrega') else None
            
            # Asegúrate de que todos los campos se obtengan correctamente
            evaluacion = {
                "estudiante_id": estudiante["estudiante_id"],
                "estudiante_nombre": estudiante["estudiante_nombre"],
                "estado": estudiante["estado"],
                "calificacion": estudiante.get("calificacion", None),  # Asegúrate de que este campo esté presente
                "comentarios": estudiante.get("comentarios", None),    # Asegúrate de que este campo esté presente
                "archivo_url": estudiante.get("archivo_url", ""),      # Asegúrate de que este campo esté presente
                "evaluacion_id": estudiante.get("evaluacion_id", ""),  # Asegúrate de que este campo esté presente
                "uso_ia": estudiante.get("uso_ia", 0),                 # Asegúrate de que este campo esté presente
                "fecha_entrega": fecha_entrega
            }
            evaluaciones.append(evaluacion)
        
        return jsonify({
            "success": True,
            "practica": {
                "id": practica.id,
                "titulo": practica.titulo,
                "objetivo": practica.objetivo,
                "fecha_entrega": fecha_entrega
            },
            "evaluaciones": evaluaciones
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def actualizar_usuario(usuario_id, nuevos_datos):
    """Actualiza la información de un usuario en el archivo usuarios.json."""
    usuarios_path = 'usuarios.json'
    
    if os.path.exists(usuarios_path):
        with open(usuarios_path, 'r') as f:
            usuarios = json.load(f)
    else:
        usuarios = {}

    # Actualizar o crear la información del usuario
    if usuario_id not in usuarios:
        usuarios[usuario_id] = {}

    # Actualizar información del usuario
    usuarios[usuario_id]['nombre'] = current_user.nombre  # Mantener el nombre del usuario actual
    usuarios[usuario_id]['email'] = current_user.email     # Mantener el email del usuario actual
    usuarios[usuario_id]['numero_cuenta'] = current_user.numero_cuenta  # Mantener el número de cuenta
    usuarios[usuario_id]['telefono'] = nuevos_datos.get('telefono', '')
    usuarios[usuario_id]['facultad'] = nuevos_datos.get('facultad', '')
    usuarios[usuario_id]['carrera'] = nuevos_datos.get('carrera', '')
    usuarios[usuario_id]['semestre'] = nuevos_datos.get('semestre', '')
    usuarios[usuario_id]['promedio'] = nuevos_datos.get('promedio', '')

    # Manejar la carga de la imagen
    if 'foto' in request.files:
        foto = request.files['foto']
        if foto and foto.filename != '':
            foto_filename = secure_filename(foto.filename)
            foto_path = os.path.join(app.config['UPLOAD_FOLDER'], foto_filename)
            foto.save(foto_path)
            usuarios[usuario_id]['foto_url'] = foto_path  # Guardar la ruta de la foto

    # Guardar usuarios en el archivo JSON
    with open(usuarios_path, 'w') as f:
        json.dump(usuarios, f, indent=4)

# Ruta para guardar la información del perfil del usuario
@app.route('/guardar-perfil', methods=['POST'])
@login_required
def guardar_perfil():
    # Cargar usuarios existentes
    usuarios_path = 'usuarios.json'
    
    if os.path.exists(usuarios_path):
        with open(usuarios_path, 'r') as f:
            usuarios = json.load(f)
    else:
        usuarios = {}

    # Actualizar o crear la información del usuario
    usuario_id = str(current_user.id)  # Usar el ID del usuario como clave
    if usuario_id not in usuarios:
        usuarios[usuario_id] = {}

    # Actualizar información del usuario
    usuarios[usuario_id]['nombre'] = current_user.nombre  # Mantener el nombre del usuario actual
    usuarios[usuario_id]['email'] = current_user.email     # Mantener el email del usuario actual
    usuarios[usuario_id]['numero_cuenta'] = current_user.numero_cuenta  # Mantener el número de cuenta
    usuarios[usuario_id]['telefono'] = request.form.get('telefono', '')
    usuarios[usuario_id]['facultad'] = request.form.get('facultad', '')
    usuarios[usuario_id]['carrera'] = request.form.get('carrera', '')
    usuarios[usuario_id]['semestre'] = request.form.get('semestre', '')
    usuarios[usuario_id]['promedio'] = request.form.get('promedio', '')

    # Manejar la carga de la imagen
    if 'foto' in request.files:
        file = request.files['foto']
        if file and file.filename != '':
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            usuarios[usuario_id]['foto_url'] = file_path  # Guardar la ruta de la imagen

    # Guardar usuarios en el archivo JSON
    with open(usuarios_path, 'w') as f:
        json.dump(usuarios, f, indent=4)

    flash('Tu perfil ha sido actualizado.', 'success')
    return redirect(url_for('perfil'))
def obtener_practicas_por_estudiante(estudiante_id):
    """Obtiene las prácticas asignadas a un estudiante."""
    configuraciones_path = 'configuraciones.json'
    usuarios_path = 'usuarios.json'

    practicas = []

    # Cargar configuraciones y usuarios
    configuraciones = cargar_json(configuraciones_path)
    usuarios = cargar_json(usuarios_path)

    # Filtrar las prácticas del estudiante
    usuario = usuarios.get(estudiante_id, {})  # Cambiado para buscar por ID de estudiante
    if str(estudiante_id) in configuraciones:
        practicas = configuraciones[str(estudiante_id)].get('practicas', [])

    return practicas

def obtener_evaluaciones_por_estudiante(estudiante_id):
    """Obtiene las evaluaciones de un estudiante."""
    evaluaciones = cargar_json('evaluaciones.json')
    # Filtrar evaluaciones por estudiante_id
    return [eval for eval in evaluaciones if eval['estudiante_id'] == estudiante_id]

def obtener_entregas_por_estudiante(estudiante_id):
    """Obtiene las entregas de un estudiante."""
    entregas = cargar_json('entregas.json')
    # Filtrar entregas por estudiante_id
    return [entrega for entrega in entregas if entrega['estudiante_id'] == estudiante_id]

# Ruta para actualizar el perfil
@app.route('/actualizar_perfil', methods=['POST'])
def actualizar_perfil():
    telefono = request.form.get('telefono')
    facultad = request.form.get('facultad')
    carrera = request.form.get('carrera')
    
    # Leer el archivo JSON
    with open('usuarios.json', 'r') as f:
        usuarios = json.load(f)

    # Obtener el ID del usuario actual
    usuario_id = str(current_user.id)

    # Verificar si el usuario existe en el archivo JSON
    if usuario_id not in usuarios:
        flash('El usuario no existe.', 'danger')
        return redirect(url_for('perfil'))

    # Manejar la carga de la fotografía
    if 'foto' in request.files:
        foto = request.files['foto']
        if foto and allowed_file(foto.filename):
            foto_filename = secure_filename(foto.filename)
            foto_path = os.path.join(app.config['UPLOAD_FOLDER'], foto_filename)
            foto.save(foto_path)
            usuarios[usuario_id]['foto_url'] = foto_path  # Guardar la ruta de la foto en la configuración

    # Actualizar los datos del usuario
    usuarios[usuario_id]['telefono'] = telefono
    usuarios[usuario_id]['facultad'] = facultad
    usuarios[usuario_id]['carrera'] = carrera

    # Guardar los cambios en el archivo JSON
    with open('usuarios.json', 'w') as f:
        json.dump(usuarios, f)

    flash('Perfil actualizado con éxito.')
    return redirect(url_for('perfil'))

# Ruta para guardar configuración (para compatibilidad con el código existente)
@app.route('/guardar', methods=['POST'])
@login_required
def guardar_configuracion_compat():
    # Llama a la función guardar_configuracion y pasa los datos del formulario
    return guardar_configuracion()

# Modificación de la función horario para obtener actividades directamente de la tabla practicas
@app.route('/horario')
@login_required
def horario():
    # Obtener el ID del estudiante actual
    estudiante_id = current_user.id
    
    # Obtener la semana seleccionada (o usar la actual)
    semana_seleccionada = request.args.get('semana')
    
    # Obtener semanas con prácticas
    semanas = obtener_semanas_con_practicas(estudiante_id)
    
    # Si no se especificó semana o la semana no existe, usar la primera semana disponible
    if not semana_seleccionada or semana_seleccionada not in semanas:
        semana_seleccionada = list(semanas.keys())[0] if semanas else None
    
    # Obtener clases para la semana seleccionada
    clases = obtener_clases_horario(estudiante_id, semana_seleccionada)
    # Obtener información del usuario desde el archivo JSON
    usuarios = cargar_json('usuarios.json')  # Cargar el archivo de usuarios
    usuario_id = str(current_user.id)  # Obtener el ID del usuario actual
    usuario = usuarios.get(usuario_id, {})  # Obtener el usuario actual
    año_actual = datetime.now().year
    return render_template('horario.html', 
                          semanas=semanas,
                          semana_seleccionada=semana_seleccionada,
                          clases=clases,
                          año_actual=año_actual,
                          usuario=usuario)  # Pasar la variable usuario


def obtener_clases_horario(estudiante_id, semana=None):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Si no se especifica semana, usar la semana actual
        if not semana:
            hoy = datetime.now()
            inicio_semana = hoy - timedelta(days=hoy.weekday())
            fin_semana = inicio_semana + timedelta(days=6)
        else:
            # Parsear la semana en formato "YYYY-MM-DD"
            inicio_semana = datetime.strptime(semana, "%Y-%m-%d")
            fin_semana = inicio_semana + timedelta(days=6)
        
        # Consulta para obtener prácticas en el rango de fechas
        query = """
        SELECT 
            p.id, 
            p.titulo, 
            p.fecha_entrega,
            DAYOFWEEK(p.fecha_entrega) as dia,
            HOUR(p.fecha_entrega) as hora,
            HOUR(p.fecha_entrega) + 2 as hora_fin,
            m.nombre as materia_nombre,
            u.nombre as profesor_nombre,
            g.nombre as grupo_nombre
        FROM practicas p
        JOIN materias m ON p.materia_id = m.id
        JOIN usuarios u ON p.autor_id = u.id
        JOIN grupos g ON p.grupo_id = g.id
        JOIN evaluaciones e ON p.id = e.practica_id
        WHERE e.estudiante_id = %s 
        AND p.fecha_entrega BETWEEN %s AND %s
        """
        cursor.execute(query, (estudiante_id, inicio_semana, fin_semana))
        practicas = cursor.fetchall()
        
        # Formatear los resultados para el horario
        clases = []
        for practica in practicas:
            # En MySQL, DAYOFWEEK() devuelve 1 para domingo, 2 para lunes, etc.
            # Ajustamos para que 1 sea lunes, 2 martes, etc.
            dia = practica['dia'] % 7  # Convertir domingo (7) a 0
            if dia == 0:
                dia = 7  # Domingo es 7 en nuestro sistema
            
            clases.append({
                'dia': dia,
                'hora_inicio': practica['hora'],
                'hora_fin': practica['hora_fin'],
                'materia': {'nombre': practica['materia_nombre']},
                'profesor': {'nombre': practica['profesor_nombre']},
                'aula': practica['grupo_nombre']
            })
        
        connection.close()
        return clases
    except Exception as e:
        print(f"Error al obtener clases del horario: {str(e)}")
        return []

# Función para obtener semanas con prácticas
def obtener_semanas_con_practicas(estudiante_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Consulta para obtener fechas de prácticas
        query = """
        SELECT DISTINCT DATE(p.fecha_entrega) as fecha
        FROM practicas p
        JOIN evaluaciones e ON p.id = e.practica_id
        WHERE e.estudiante_id = %s
        ORDER BY fecha
        """
        cursor.execute(query, (estudiante_id,))
        fechas = cursor.fetchall()
        
        connection.close()
        
        # Agrupar por semanas
        semanas = {}
        for fecha_dict in fechas:
            fecha = fecha_dict['fecha']
            inicio_semana = fecha - timedelta(days=fecha.weekday())
            fin_semana = inicio_semana + timedelta(days=6)
            
            semana_key = inicio_semana.strftime("%Y-%m-%d")
            semana_label = f"{inicio_semana.strftime('%d/%m/%Y')} - {fin_semana.strftime('%d/%m/%Y')}"
            
            semanas[semana_key] = semana_label
        
        # Si no hay semanas, agregar la semana actual
        if not semanas:
            hoy = datetime.now()
            inicio_semana = hoy - timedelta(days=hoy.weekday())
            fin_semana = inicio_semana + timedelta(days=6)
            
            semana_key = inicio_semana.strftime("%Y-%m-%d")
            semana_label = f"{inicio_semana.strftime('%d/%m/%Y')} - {fin_semana.strftime('%d/%m/%Y')}"
            
            semanas[semana_key] = semana_label
        
        return semanas
    except Exception as e:
        print(f"Error al obtener semanas con prácticas: {str(e)}")
        # Valor por defecto en caso de error
        hoy = datetime.now()
        inicio_semana = hoy - timedelta(days=hoy.weekday())
        fin_semana = inicio_semana + timedelta(days=6)
        
        semana_key = inicio_semana.strftime("%Y-%m-%d")
        semana_label = f"{inicio_semana.strftime('%d/%m/%Y')} - {fin_semana.strftime('%d/%m/%Y')}"
        
        return {semana_key: semana_label}

# Ruta para mostrar calificaciones
@app.route('/calificaciones', methods=['GET'])
@login_required
def calificaciones():
    """Muestra las calificaciones según el rol del usuario."""

    usuarios = cargar_json('usuarios.json')  # Cargar el archivo de usuarios
    usuario_id = str(current_user.id)  # Obtener el ID del usuario actual
    usuario = usuarios.get(usuario_id, {})  # Obtener el usuario actual

    semestre_id = request.args.get('semestre_id', 'all')  # Semestre seleccionado

    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # Obtener todos los semestres
    cursor.execute("SELECT id, nombre FROM semestres ORDER BY fecha_inicio DESC")
    semestres = cursor.fetchall()

    if current_user.rol == 'estudiante':
        # Vista de estudiante: solo sus calificaciones
        if semestre_id != 'all':
            query = """
            SELECT cf.*, g.nombre as grupo_nombre, m.nombre as materia_nombre, 
                   m.creditos, s.nombre as semestre_nombre, s.id as semestre_id
            FROM calificaciones_finales cf
            JOIN grupos g ON cf.grupo_id = g.id
            JOIN materias m ON g.materia_id = m.id
            JOIN semestres s ON g.semestre_id = s.id
            WHERE cf.estudiante_id = %s AND g.semestre_id = %s
            ORDER BY cf.calificacion_final DESC
            """
            cursor.execute(query, (current_user.id, semestre_id))
        else:
            query = """
            SELECT cf.*, g.nombre as grupo_nombre, m.nombre as materia_nombre, 
                   m.creditos, s.nombre as semestre_nombre, s.id as semestre_id
            FROM calificaciones_finales cf
            JOIN grupos g ON cf.grupo_id = g.id
            JOIN materias m ON g.materia_id = m.id
            JOIN semestres s ON g.semestre_id = s.id
            WHERE cf.estudiante_id = %s
            ORDER BY s.fecha_inicio DESC, cf.calificacion_final DESC
            """
            cursor.execute(query, (current_user.id,))

        calificaciones = cursor.fetchall()

        # Calcular estadísticas
        materias_aprobadas = sum(1 for c in calificaciones if c['calificacion_final'] >= 6)
        creditos_acumulados = sum(c['creditos'] for c in calificaciones if c['calificacion_final'] >= 6)

        # Obtener créditos totales del plan de estudios
        cursor.execute("SELECT SUM(creditos) as total FROM materias")
        result = cursor.fetchone()
        creditos_totales = result['total'] if result and result['total'] else 0

        # Calcular promedio general
        if calificaciones:
            promedio_general = sum(c['calificacion_final'] for c in calificaciones) / len(calificaciones)
            promedio_general = round(promedio_general, 2)
        else:
            promedio_general = 0

        # Preparar datos para gráfico
        if semestre_id != 'all' and calificaciones:
            materias_json = json.dumps([c['materia_nombre'] for c in calificaciones])
            calificaciones_json = json.dumps([float(c['calificacion_final']) for c in calificaciones])
        else:
            ultimo_semestre_calificaciones = []
            if calificaciones:
                ultimo_semestre_id = calificaciones[0]['semestre_id']
                ultimo_semestre_calificaciones = [c for c in calificaciones if c['semestre_id'] == ultimo_semestre_id]

            materias_json = json.dumps([c['materia_nombre'] for c in ultimo_semestre_calificaciones])
            calificaciones_json = json.dumps([float(c['calificacion_final']) for c in ultimo_semestre_calificaciones])

        semestre_actual = semestre_id if semestre_id != 'all' else (calificaciones[0]['semestre_id'] if calificaciones else None)

        connection.close()

        return render_template(
            'calificaciones.html',
            calificaciones=calificaciones,
            semestres=semestres,
            usuario=usuario,
            promedio_general=promedio_general,
            materias_aprobadas=materias_aprobadas,
            creditos_acumulados=creditos_acumulados,
            creditos_totales=creditos_totales,
            materias_json=materias_json,
            calificaciones_json=calificaciones_json,
            semestre_actual=semestre_actual,
            año_actual=datetime.now().year,
            rol=current_user.rol
        )
    
    elif current_user.rol == 'profesor':
        # Vista de profesor: calificaciones de sus grupos
        if semestre_id != 'all':
            query = """
            SELECT g.id as grupo_id, g.nombre as grupo_nombre, m.nombre as  materia_nombre,
                   s.nombre as semestre_nombre, s.id as semestre_id,
                   COUNT(DISTINCT gm.usuario_id) as total_estudiantes,
                   AVG(cf.calificacion_final) as promedio_grupo
            FROM grupos g
            JOIN materias m ON g.materia_id = m.id
            JOIN semestres s ON g.semestre_id = s.id
            LEFT JOIN grupo_miembros gm ON g.id = gm.grupo_id
            LEFT JOIN calificaciones_finales cf ON g.id = cf.grupo_id AND   gm.usuario_id = cf.estudiante_id
            WHERE g.profesor_id = %s AND g.semestre_id = %s
            GROUP BY g.id, g.nombre, m.nombre, s.nombre, s.id
            ORDER BY s.fecha_inicio DESC, promedio_grupo DESC
            """

            cursor.execute(query, (current_user.id, semestre_id))
        else:
            # Vista de profesor: obtener las calificaciones de los estudiantes de los grupos que enseña
            query = """
            SELECT g.id as grupo_id, g.nombre as grupo_nombre, m.nombre as          materia_nombre,
                   s.nombre as semestre_nombre, s.id as semestre_id,
                   COUNT(DISTINCT gm.usuario_id) as total_estudiantes,
                   AVG(cf.calificacion_final) as promedio_grupo
            FROM grupos g
            JOIN materias m ON g.materia_id = m.id
            JOIN semestres s ON g.semestre_id = s.id
            LEFT JOIN grupo_miembros gm ON g.id = gm.grupo_id
            LEFT JOIN calificaciones_finales cf ON g.id = cf.grupo_id AND cf.           estudiante_id = gm.usuario_id
            WHERE g.profesor_id = %s
            GROUP BY g.id, g.nombre, m.nombre, s.nombre, s.id
            ORDER BY s.fecha_inicio DESC, promedio_grupo DESC
            """
            cursor.execute(query, (current_user.id,))
        
        grupos = cursor.fetchall()
        
        # Calcular estadísticas
        total_grupos = len(grupos)
        total_estudiantes = sum(g['total_estudiantes'] for g in grupos)
        
        # Calcular promedio general de todos los grupos
        promedios_validos = [g['promedio_grupo'] for g in grupos if g['promedio_grupo'] is not None]
        if promedios_validos:
            promedio_general = sum(promedios_validos) / len(promedios_validos)
            promedio_general = round(promedio_general, 2)
        else:
            promedio_general = 0
        
        # Preparar datos para el gráfico
        if semestre_id != 'all' and grupos:
            grupos_json = json.dumps([g['grupo_nombre'] for g in grupos])
            promedios_json = json.dumps([float(g['promedio_grupo']) if g['promedio_grupo'] is not None else 0 for g in grupos])
        else:
            # Si no hay semestre seleccionado o no hay grupos, mostrar datos del último semestre
            ultimo_semestre_grupos = []
            if grupos:
                ultimo_semestre_id = grupos[0]['semestre_id']
                ultimo_semestre_grupos = [g for g in grupos if g['semestre_id'] == ultimo_semestre_id]
            
            grupos_json = json.dumps([g['grupo_nombre'] for g in ultimo_semestre_grupos])
            promedios_json = json.dumps([float(g['promedio_grupo']) if g['promedio_grupo'] is not None else 0 for g in ultimo_semestre_grupos])
        
        # Establecer el semestre actual para la visualización
        semestre_actual = semestre_id if semestre_id != 'all' else (grupos[0]['semestre_id'] if grupos else None)
        
        # Obtener detalles de estudiantes por grupo para mostrar en detalle
        grupos_detalle = {}
        for grupo in grupos:
            cursor.execute("""
                SELECT u.nombre as estudiante_nombre, cf.calificacion_final,
                       cf.practicas_promedio, cf.examenes_promedio, cf.proyectos_promedio, cf.asistencia_porcentaje
                FROM calificaciones_finales cf
                JOIN usuarios u ON cf.estudiante_id = u.id
                WHERE cf.grupo_id = %s
                ORDER BY cf.calificacion_final DESC
            """, (grupo['grupo_id'],))
            estudiantes = cursor.fetchall()
            grupos_detalle[grupo['grupo_id']] = estudiantes
        
        connection.close()
        
        return render_template(
            'calificaciones_profesor.html',
            grupos=grupos,
            grupos_detalle=grupos_detalle,
            semestres=semestres,
            promedio_general=promedio_general,
            total_grupos=total_grupos,
            total_estudiantes=total_estudiantes,
            grupos_json=grupos_json,
            promedios_json=promedios_json,
            semestre_actual=semestre_actual,
            año_actual=datetime.now().year,
            rol=current_user.rol,
            usuario=usuario
        )

# Función para obtener el promedio de calificaciones
def obtener_promedio_calificaciones(estudiante_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Consulta para obtener calificaciones
        query = """
        SELECT AVG(calificacion) as promedio
        FROM evaluaciones
        WHERE estudiante_id = %s AND calificacion IS NOT NULL
        """
        cursor.execute(query, (estudiante_id,))
        resultado = cursor.fetchone()
        
        connection.close()
        
        if resultado and resultado['promedio']:
            return round(float(resultado['promedio']), 2)
        return 0.0
    except Exception as e:
        print(f"Error al obtener promedio de calificaciones: {str(e)}")
        return 0.0

# Función para obtener materias aprobadas
def obtener_materias_aprobadas(estudiante_id, calificacion_minima=6.0):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Consulta para obtener materias aprobadas
        query = """
        SELECT COUNT(DISTINCT p.materia_id) as materias_aprobadas
        FROM evaluaciones e
        JOIN practicas p ON e.practica_id = p.id
        WHERE e.estudiante_id = %s AND e.calificacion >= %s
        """
        cursor.execute(query, (estudiante_id, calificacion_minima))
        resultado = cursor.fetchone()
        
        connection.close()
        
        if resultado:
            return resultado['materias_aprobadas']
        return 0
    except Exception as e:
        print(f"Error al obtener materias aprobadas: {str(e)}")
        return 0

# Función para obtener créditos acumulados
def obtener_creditos_acumulados(estudiante_id, calificacion_minima=6.0):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Consulta para obtener créditos acumulados
        # Nota: Asumimos que hay una columna 'creditos' en la tabla 'materias'
        query = """
        SELECT SUM(m.creditos) as creditos_acumulados
        FROM evaluaciones e
        JOIN practicas p ON e.practica_id = p.id
        JOIN materias m ON p.materia_id = m.id
        WHERE e.estudiante_id = %s AND e.calificacion >= %s
        """
        cursor.execute(query, (estudiante_id, calificacion_minima))
        resultado = cursor.fetchone()
        
        connection.close()
        
        if resultado and resultado['creditos_acumulados']:
            return resultado['creditos_acumulados']
        return 0
    except Exception as e:
        print(f"Error al obtener créditos acumulados: {str(e)}")
        # Valor por defecto en caso de error o si no hay columna de créditos
        return 30

# Función para obtener créditos totales
def obtener_creditos_totales():
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Consulta para obtener créditos totales
        query = """
        SELECT SUM(creditos) as creditos_totales
        FROM materias
        """
        cursor.execute(query)
        resultado = cursor.fetchone()
        
        connection.close()
        
        if resultado and resultado['creditos_totales']:
            return resultado['creditos_totales']
        return 0
    except Exception as e:
        print(f"Error al obtener créditos totales: {str(e)}")
        # Valor por defecto en caso de error o si no hay columna de créditos
        return 400

# Función para obtener calificaciones por semestre
def obtener_calificaciones_por_semestre(estudiante_id, semestre_id=None):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Consulta base
        query = """
        SELECT 
            m.id as materia_id, 
            m.nombre as materia_nombre, 
            m.creditos as materia_creditos,
            s.id as semestre_id, 
            s.nombre as semestre_nombre,
            AVG(e.calificacion) as calificacion
        FROM evaluaciones e
        JOIN practicas p ON e.practica_id = p.id
        JOIN materias m ON p.materia_id = p.id
        JOIN grupos g ON p.grupo_id = g.id
        JOIN semestres s ON g.semestre_id = s.id
        WHERE e.estudiante_id = %s AND e.calificacion IS NOT NULL
        """
        
        # Añadir filtro por semestre si se especifica
        if semestre_id:
            query += " AND s.id = %s"
            cursor.execute(query, (estudiante_id, semestre_id))
        else:
            cursor.execute(query, (estudiante_id,))
        
        calificaciones = cursor.fetchall()
        
        # Formatear los resultados
        resultados = []
        for cal in calificaciones:
            resultados.append({
                'materia': {
                    'nombre': cal['materia_nombre'],
                    'creditos': cal['materia_creditos'] or 8  # Valor por defecto si es NULL
                },
                'semestre': {
                    'id': cal['semestre_id'],
                    'nombre': cal['semestre_nombre']
                },
                'calificacion': round(float(cal['calificacion']), 1)
            })
        
        connection.close()
        return resultados
    except Exception as e:
        print(f"Error al obtener calificaciones por semestre: {str(e)}")
        return []


def obtener_semestres():
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = "SELECT id, nombre FROM semestres ORDER BY id"
        cursor.execute(query)
        semestres = cursor.fetchall()
        
        connection.close()
        
        if not semestres:  # Si no hay semestres en la BD, usar valores por defecto
            return [
                {'id': 1, 'nombre': 'Semestre 1'},
                {'id': 2, 'nombre': 'Semestre 2'},
                {'id': 3, 'nombre': 'Semestre 3'}
            ]
        
        return semestres
    except Exception as e:
        print(f"Error al obtener semestres: {str(e)}")
        # Valores por defecto en caso de error
        return [
            {'id': 1, 'nombre': 'Semestre 1'},
            {'id': 2, 'nombre': 'Semestre 2'},
            {'id': 3, 'nombre': 'Semestre 3'}
        ]


def dividir_texto_en_fragmentos(texto, max_palabras=100):
    """Divide el texto en fragmentos de un número máximo de palabras."""
    palabras = texto.split()
    for i in range(0, len(palabras), max_palabras):
        yield ' '.join(palabras[i:i + max_palabras])

def preprocesar_texto(texto):
    """Elimina caracteres especiales y convierte el texto a minúsculas."""
    texto = re.sub(r'[^\w\s]', '', texto.lower())
    return texto.strip()

@app.route('/cargar_modelo', methods=['GET'])
def cargar_modelo():
    """
    Carga el modelo entrenado desde un archivo.
    """
    try:
        generador.modelo_ml.cargar_modelo()
        return jsonify({"success": True, "message": "Modelo cargado correctamente."}), 200
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

# Opción 1: Verificar si la ruta existe en app.view_functions
@app.route('/estudiante/<int:estudiante_id>')
def vista_estudiante(estudiante_id):
    # Verificar si la ruta 'perfil_usuario' existe
    tiene_perfil_usuario = 'perfil_usuario' in app.view_functions
    
    """Muestra las prácticas activas de un estudiante y sus entregas."""
    practicas = generador.obtener_practicas_estudiante(estudiante_id)
    entregas = generador.obtener_entregas_por_estudiante(estudiante_id)  # Obtener entregas del estudiante
    
    # Obtener evaluaciones calificadas para mostrarlas en la misma página
    evaluaciones_calificadas = generador.obtener_evaluaciones_calificadas(estudiante_id)
    
    # Obtener información del usuario desde el archivo JSON
    usuario = cargar_json('usuarios.json').get(str(estudiante_id), {})  # Asegúrate de que el ID esté en formato de cadena
    
    año_actual = datetime.now().year
    return render_template('estudiante.html',
                          usuario=usuario,  # Pasar la variable usuario
                          practicas=practicas, 
                          entregas=entregas, 
                          evaluaciones_calificadas=evaluaciones_calificadas,
                          año_actual=año_actual,
                          tiene_perfil_usuario=tiene_perfil_usuario)  # Pasar la variable al template

@app.route('/retroalimentacion/<int:practica_id>')
@login_required
def retroalimentacion(practica_id):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)

    # Verificar práctica
    cursor.execute("SELECT * FROM practicas WHERE id = %s", (practica_id,))
    practica = cursor.fetchone()
    if not practica:
        flash('Práctica no encontrada', 'error')
        return redirect(url_for('vista_estudiante', estudiante_id=current_user.id))

    # Verificar permisos (solo estudiantes)
    if current_user.rol != 'estudiante':
        flash('Acceso no autorizado', 'error')
        return redirect(url_for('dashboard'))

    cursor.execute("SELECT * FROM entregas WHERE practica_id = %s AND estudiante_id = %s", (practica_id, current_user.id))
    entrega = cursor.fetchone()

    if not entrega or not entrega.get('evaluacion_id'):
        flash('No se encontró la entrega o evaluación asociada.', 'warning')
        return redirect(url_for('vista_estudiante', estudiante_id=current_user.id))

    try:
        try:
            from modelo_ml_enhanced import EnhancedModeloEvaluacionInteligente as ModeloML
            modelo = ModeloML(db_config=DB_CONFIG, use_sbert=True)
        except ImportError:
            from modelo_ml_scikit import ModeloEvaluacionInteligente as ModeloML
            modelo = ModeloML()

        cursor.execute("SELECT estilos_aprendizaje FROM perfiles_estudiante WHERE estudiante_id = %s", (current_user.id,))
        resultado = cursor.fetchone()
        estilo_aprendizaje = resultado['estilos_aprendizaje'] if resultado else None

        evaluacion = modelo.analizar_contenido(
            entrega['contenido'],
            practica['titulo'],
            practica['objetivo'],
            estilo_aprendizaje
        )
    except Exception as e:
        app.logger.error(f"Error al generar retroalimentación: {str(e)}")
        evaluacion = {
            "calificacion": 0,
            "comentarios": "No hay comentarios disponibles",
            "sugerencias": "No hay sugerencias disponibles",
            "relevancia": 0,
            "metricas": {},
            "fortalezas": [],
            "debilidades": [],
            "recursos_recomendados": []
        }

    connection.close()

    return render_template(
        'retroalimentacion.html',
        practica=practica,
        evaluacion=evaluacion
    )



# Contexto global para todas las plantillas
@app.context_processor
def inject_global_vars():
    año_actual = datetime.now().year
    tema = session.get('tema', 'light')

    notificaciones = []
    if current_user.is_authenticated:
        # Verificar prácticas pendientes
        practicas_pendientes = verificar_practicas_pendientes(current_user.id) or []
        print(f"Prácticas pendientes: {practicas_pendientes}")  # Depuración
        if practicas_pendientes:
            notificaciones.append({
                'tipo': 'warning',
                'mensaje': f"Tienes {len(practicas_pendientes)} actividades sin entregar."
            })
        
        # Verificar prácticas por caducar
        practicas_por_caducar = verificar_practicas_por_caducar(current_user.id) or []
        print(f"Prácticas por caducar: {practicas_por_caducar}")  # Depuración
        if practicas_por_caducar:
            if len(practicas_por_caducar) == 1:
                notificaciones.append({
                    'tipo': 'danger',
                    'mensaje': f"Una actividad está por caducar: {practicas_por_caducar[0]['titulo']}"
                })
            else:
                notificaciones.append({
                    'tipo': 'danger',
                    'mensaje': f"{len(practicas_por_caducar)} actividades están por caducar."
                })

    return {
        'año_actual': año_actual,
        'tema': tema,
        'notificaciones': notificaciones
    }


@app.route('/')
def index():
    año_actual = datetime.now().year
    return render_template('login.html', año_actual=año_actual)

@app.route('/inicio')
@login_required
def inicio():

    # Obtener información del usuario desde el archivo JSON
    usuarios = cargar_json('usuarios.json')  # Cargar el archivo de usuarios
    usuario_id = str(current_user.id)  # Obtener el ID del usuario actual
    usuario = usuarios.get(usuario_id, {})  # Obtener el usuario actual

    # Determinar qué dashboard mostrar según el rol del usuario
    if current_user.rol == 'administrador':
        return render_template('inicio.html', rol='administrador', usuario=usuario)
    elif current_user.rol == 'profesor':
        return render_template('inicio.html', rol='profesor', usuario=usuario)
    else:  # Estudiante u otro rol
        # Obtener el ID del estudiante
        estudiante_id = current_user.id
        configuracion = obtener_configuracion(usuario_id)  # <- asegúrate de tener esta función
        # Obtener prácticas pendientes y por caducar
        practicas_pendientes = verificar_practicas_pendientes(estudiante_id) or []
        practicas_por_caducar = verificar_practicas_por_caducar(estudiante_id) or []
        # Dentro de la vista estudiante en la ruta /inicio
        practicas_atrasadas = verificar_practicas_atrasadas(estudiante_id) or []
        # Pasar las variables correctamente a la plantilla
        return render_template('inicio.html', 
                               rol='estudiante',
                               estudiante_id=estudiante_id,
                               practicas_pendientes=practicas_pendientes,
                               practicas_por_caducar=practicas_por_caducar, configuracion=configuracion, practicas_atrasadas=practicas_atrasadas,
                               usuario=usuario)

def verificar_practicas_atrasadas(estudiante_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT p.id, p.titulo, p.fecha_entrega, e.estado
        FROM practicas p
        JOIN evaluaciones e ON p.id = e.practica_id
        WHERE e.estudiante_id = %s AND e.estado = 'pendiente' AND p.fecha_entrega < NOW()
        ORDER BY p.fecha_entrega ASC
        """
        cursor.execute(query, (estudiante_id,))
        practicas_atrasadas = cursor.fetchall()
        connection.close()
        return practicas_atrasadas
    except Exception as e:
        print(f"Error al verificar prácticas atrasadas: {str(e)}")
        return []


def verificar_practicas_pendientes(estudiante_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Consulta para obtener prácticas pendientes
        query = """
        SELECT p.id, p.titulo, p.fecha_entrega, e.estado
        FROM practicas p
        JOIN evaluaciones e ON p.id = e.practica_id
        WHERE e.estudiante_id = %s AND e.estado = 'pendiente' AND p.fecha_entrega > NOW()
        ORDER BY p.fecha_entrega ASC
        """
        cursor.execute(query, (estudiante_id,))
        practicas_pendientes = cursor.fetchall()
        
        connection.close()
        return practicas_pendientes
    except Exception as e:
        print(f"Error al verificar prácticas pendientes: {str(e)}")
        return []


# Función para verificar si hay prácticas por caducar
def verificar_practicas_por_caducar(estudiante_id):
    try:
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        # Consulta para obtener prácticas que están por caducar
        query = """
        SELECT p.id, p.titulo, p.fecha_entrega, e.estado
        FROM practicas p
        JOIN evaluaciones e ON p.id = e.practica_id
        WHERE e.estudiante_id = %s AND e.estado = 'pendiente' 
        AND p.fecha_entrega > NOW() AND p.fecha_entrega <= DATE_ADD(NOW(), INTERVAL 3 DAY)
        ORDER BY p.fecha_entrega ASC
        """
        cursor.execute(query, (estudiante_id,))
        practicas_por_caducar = cursor.fetchall()
        
        connection.close()
        return practicas_por_caducar
    except Exception as e:
        print(f"Error al verificar prácticas por caducar: {str(e)}")
        return []

@app.route('/practicas', methods=['POST'])
def crear_practica():
    """Guarda una nueva práctica con la opción de calificación automática."""
    titulo = request.form['titulo']
    materia_id = int(request.form['materia_id'])
    nivel_id = int(request.form['nivel_id'])
    autor_id = int(request.form['autor_id'])
    objetivo = request.form['objetivo']
    fecha_entrega = request.form['fecha_entrega']
    tiempo_estimado = int(request.form['tiempo_estimado'])
    concepto_id = int(request.form['concepto_id'])
    herramienta_id = int(request.form['herramienta_id'])
    uso_ia = bool(int(request.form['uso_ia']))  # Convertir a booleano
    grupo_id = int(request.form['grupo_id'])

    practica = Practica(
        titulo=titulo,
        materia_id=materia_id,
        nombre_materia="",  # Se asigna después
        nivel_id=nivel_id,
        autor_id=int(autor_id),  # Convertir a entero
        objetivo=objetivo,
        fecha_entrega=datetime.strptime(fecha_entrega, '%Y-%m-%d'),
        estado='pendiente',
        concepto_id=concepto_id,
        herramienta_id=herramienta_id,
        tiempo_estimado=tiempo_estimado,
        uso_ia=uso_ia,  # Mantener la opción de calificación con IA
        grupo_id=grupo_id,
    )

    usuario_id = str(current_user.id)

    usuarios = cargar_json('usuarios.json')  # Cargar el archivo de usuarios
    usuario = usuarios.get(usuario_id, {})  # Obtener el usuario actual
    generador.crear_practica(practica)
    flash("Práctica creada exitosamente", "success")
    return redirect(url_for('practicas', usuario=usuario))

# Modificar la función crear_practica para guardar correctamente el tipo_asignacion
@app.route('/practicas', methods=['GET', 'POST'])
@login_required
def practicas():
    """Gestión de prácticas académicas."""
    if current_user.rol not in ['administrador', 'profesor']:
        flash('No tienes permisos para acceder a esta página.', 'error')
        return redirect(url_for('inicio'))
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Procesar formulario para crear práctica
    if request.method == 'POST':
        titulo = request.form['titulo']
        grupo_id = request.form['grupo_id']
        nivel_id = request.form['nivel_id']
        autor_id = request.form.get('autor_id', current_user.id)
        objetivo = request.form['objetivo']
        materia_id = request.form['materia_id']
        fecha_entrega = request.form['fecha_entrega']
        tiempo_estimado = request.form['tiempo_estimado']
        concepto_id = request.form['concepto_id']
        herramienta_id = request.form['herramienta_id']
        uso_ia = request.form.get('uso_ia', '0')
        
        # Obtener tipo de práctica y tipo de juego
        tipo = request.form.get('tipo', 'juego')
        tipo = request.form.get('tipo') if tipo == 'juego' else None
        
        # Establecer tipo_asignacion según el tipo de práctica
        tipo_asignacion = 'juego' if tipo == 'juego' else request.form.get('tipo_asignacion', 'practica')
        
        # Insertar la práctica en la base de datos
        cursor.execute("""
            INSERT INTO practicas 
            (titulo, grupo_id, nivel_id, autor_id, objetivo, materia_id, 
             fecha_entrega, tiempo_estimado, concepto_id, herramienta_id, uso_ia, 
             tipo_asignacion, tipo)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            titulo, grupo_id, nivel_id, autor_id, objetivo, materia_id,
            fecha_entrega, tiempo_estimado, concepto_id, herramienta_id, uso_ia, 
            tipo_asignacion, tipo
        ))
        
        practica_id = cursor.lastrowid
        
        # Asignar la práctica a todos los estudiantes del grupo
        cursor.execute("""
            INSERT INTO asignaciones (practica_id, estudiante_id, fecha_entrega)
            SELECT %s, ge.estudiante_id, %s
            FROM grupo_estudiante ge
            WHERE ge.grupo_id = %s
        """, (practica_id, fecha_entrega, grupo_id))
        
        connection.commit()
        flash('Práctica creada exitosamente.', 'success')
        
        # Si es un juego, redirigir a la página de creación de juegos
        if tipo == 'juego':
            return redirect(url_for('juegos.crear_juego', practica_id=practica_id))
        else:
            return redirect(url_for('practicas'))
    
    # Obtener datos para la vista
    if current_user.rol == 'administrador':
        cursor.execute("""
            SELECT p.*, m.nombre as nombre_materia, g.nombre as nombre_grupo
            FROM practicas p
            JOIN materias m ON p.materia_id = m.id
            JOIN grupos g ON p.grupo_id = g.id
            ORDER BY p.fecha_entrega DESC
        """)
    else:  # Profesor
        cursor.execute("""
            SELECT p.*, m.nombre as nombre_materia, g.nombre as nombre_grupo
            FROM practicas p
            JOIN materias m ON p.materia_id = m.id
            JOIN grupos g ON p.grupo_id = g.id
            WHERE p.autor_id = %s
            ORDER BY p.fecha_entrega DESC
        """, (current_user.id,))
    
    practicas = cursor.fetchall()
    
    # Obtener datos para los formularios
    cursor.execute("""
        SELECT g.id, g.nombre, m.nombre as materia_nombre
        FROM grupos g
        JOIN materias m ON g.materia_id = m.id
        WHERE g.profesor_id = %s
    """, (current_user.id,))
    grupos = cursor.fetchall()
    
    cursor.execute("SELECT id, nombre FROM materias")
    materias = cursor.fetchall()
    
    cursor.execute("SELECT id, nombre FROM niveles")
    niveles = cursor.fetchall()
    
    cursor.execute("SELECT id, nombre FROM conceptos")
    conceptos = cursor.fetchall()
    
    cursor.execute("SELECT id, nombre FROM herramientas")
    herramientas = cursor.fetchall()
    
    if current_user.rol == 'administrador':
        cursor.execute("""
            SELECT id, nombre FROM usuarios
            WHERE rol IN ('administrador', 'profesor')
        """)
        autorizados = cursor.fetchall()
    else:
        autorizados = []
    
    usuario_id = str(current_user.id)
    usuarios = cargar_json('usuarios.json')  # Cargar el archivo de usuarios
    usuario = usuarios.get(usuario_id, {})  # Obtener el usuario actual

    connection.close()
    
    return render_template(
        'practicas.html',
        practicas=practicas,
        usuario=usuario,
        grupos=grupos,
        materias=materias,
        niveles=niveles,
        conceptos=conceptos,
        herramientas=herramientas,
        autorizados=autorizados,
        rol=current_user.rol,
        usuario_nombre=current_user.nombre,
        año_actual=datetime.now().year
    )

# Modificar la función ver_practica para manejar juegos
@app.route('/ver_practica/<int:practica_id>')
@login_required
def ver_practica(practica_id):
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Obtener información de la práctica
    cursor.execute("""
        SELECT p.*, m.nombre as nombre_materia, g.nombre as nombre_grupo
        FROM practicas p
        JOIN materias m ON p.materia_id = m.id
        JOIN grupos g ON p.grupo_id = g.id
        WHERE p.id = %s
    """, (practica_id,))
    
    practica = cursor.fetchone()
    
    if not practica:
        flash('Práctica no encontrada', 'error')
        return redirect(url_for('practicas'))
    
    # Si es un estudiante y la práctica es un juego, redirigir al juego
    if current_user.rol == 'estudiante' and practica['tipo_asignacion'] == 'juego':
        # Determine the game type and redirect accordingly
        cursor.execute("""
            SELECT * FROM contenido_generado
            WHERE practica_id = %s
        """, (practica_id,))
        contenido = cursor.fetchone()

        if contenido:
            tipo_juego = contenido['tipo']
            if tipo_juego == 'memorama':
                return redirect(url_for('juegos.memorama', practica_id=practica_id))
            elif tipo_juego == 'quiz':
                return redirect(url_for('juegos.quiz', practica_id=practica_id))
            else:
                return redirect(url_for('juegos.menu_juegos', practica_id=practica_id))
        else:
            flash('No se ha especificado el tipo de juego.', 'error')
            return redirect(url_for('practicas'))
    
    # Si es profesor y la práctica es un juego, mostrar opciones de edición
    if (current_user.rol == 'profesor' or current_user.rol == 'administrador') and practica['tipo_asignacion'] == 'juego':
        # Verificar si ya existe contenido para este juego
        cursor.execute("""
            SELECT id FROM contenido_generado
            WHERE practica_id = %s
        """, (practica_id,))
        
        tiene_contenido = cursor.fetchone() is not None
        
        # Si no tiene contenido, redirigir a la página de creación
        if not tiene_contenido:
            flash('Esta práctica requiere configuración adicional. Por favor, configura el juego.', 'info')
            return redirect(url_for('juegos.crear_juego', practica_id=practica_id))
    
    connection.close()
    
    # Renderizar la plantilla correspondiente
    return render_template('ver_practica.html', practica=practica)

# Función para calcular calificaciones finales de un grupo
def calcular_calificaciones_finales(grupo_id):
    """Calcula las calificaciones finales de todos los estudiantes en un grupo."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Obtener criterios de evaluación del grupo
    cursor.execute("""
        SELECT * FROM criterios_evaluacion WHERE grupo_id = %s
    """, (grupo_id,))
    criterios = cursor.fetchone()
    
    if not criterios:
        # Si no hay criterios definidos, usar valores predeterminados
        criterios = {
            'practicas_porcentaje': 40,
            'examenes_porcentaje': 30,
            'proyectos_porcentaje': 20,
            'asistencia_porcentaje': 10
        }
    
    # Obtener todos los estudiantes del grupo
    cursor.execute("""
        SELECT DISTINCT u.id, u.nombre
        FROM usuarios u
        JOIN grupo_estudiante ge ON u.id = ge.estudiante_id
        WHERE ge.grupo_id = %s
    """, (grupo_id,))
    estudiantes = cursor.fetchall()
    
    # Para cada estudiante, calcular su calificación final
    for estudiante in estudiantes:
        estudiante_id = estudiante['id']
        
        # Calcular promedio de prácticas
        cursor.execute("""
            SELECT AVG(e.calificacion) as promedio
            FROM evaluaciones e
            JOIN practicas p ON e.practica_id = p.id
            WHERE e.estudiante_id = %s AND p.grupo_id = %s AND p.tipo_asignacion = 'practica'
        """, (estudiante_id, grupo_id))
        result = cursor.fetchone()
        practicas_promedio = float(result['promedio']) if result and result['promedio'] else 0
        
        # Calcular promedio de exámenes
        cursor.execute("""
            SELECT AVG(e.calificacion) as promedio
            FROM evaluaciones e
            JOIN practicas p ON e.practica_id = p.id
            WHERE e.estudiante_id = %s AND p.grupo_id = %s AND p.tipo_asignacion = 'examen'
        """, (estudiante_id, grupo_id))
        result = cursor.fetchone()
        examenes_promedio = float(result['promedio']) if result and result['promedio'] else 0
        
        # Calcular promedio de proyectos
        cursor.execute("""
            SELECT AVG(e.calificacion) as promedio
            FROM evaluaciones e
            JOIN practicas p ON e.practica_id = p.id
            WHERE e.estudiante_id = %s AND p.grupo_id = %s AND p.tipo_asignacion = 'proyecto'
        """, (estudiante_id, grupo_id))
        result = cursor.fetchone()
        proyectos_promedio = float(result['promedio']) if result and result['promedio'] else 0
        
        # Calcular porcentaje de asistencia
        cursor.execute("""
            SELECT 
                COUNT(CASE WHEN estado = 'presente' OR estado = 'justificado' THEN 1 END) as asistencias,
                COUNT(*) as total_clases
            FROM asistencias
            WHERE estudiante_id = %s AND grupo_id = %s
        """, (estudiante_id, grupo_id))
        result = cursor.fetchone()
        
        if result and result['total_clases'] > 0:
            asistencia_porcentaje = (result['asistencias'] / result['total_clases']) * 100
        else:
            asistencia_porcentaje = 0
        
        # Calcular calificación final ponderada
        calificacion_final = (
            (practicas_promedio * criterios['practicas_porcentaje'] / 100) +
            (examenes_promedio * criterios['examenes_porcentaje'] / 100) +
            (proyectos_promedio * criterios['proyectos_porcentaje'] / 100) +
            (asistencia_porcentaje * criterios['asistencia_porcentaje'] / 100 / 10)  # Convertir a escala de 10
        )
        
        # Redondear a 2 decimales
        calificacion_final = round(calificacion_final, 2)
        
        # Guardar o actualizar la calificación final
        cursor.execute("""
            INSERT INTO calificaciones_finales 
            (estudiante_id, grupo_id, calificacion_final, practicas_promedio, 
             examenes_promedio, proyectos_promedio, asistencia_porcentaje)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
            calificacion_final = %s,
            practicas_promedio = %s,
            examenes_promedio = %s,
            proyectos_promedio = %s,
            asistencia_porcentaje = %s,
            fecha_calificacion = NOW()
        """, (
            estudiante_id, grupo_id, calificacion_final, practicas_promedio,
            examenes_promedio, proyectos_promedio, asistencia_porcentaje,
            calificacion_final, practicas_promedio, examenes_promedio,
            proyectos_promedio, asistencia_porcentaje
        ))
    
    # Marcar el grupo como evaluado
    cursor.execute("""
        UPDATE grupos SET evaluacion_finalizada = TRUE WHERE id = %s
    """, (grupo_id,))
    
    connection.commit()
    connection.close()

@app.route('/calificar_automatico', methods=['POST'])
def calificar_automatico():
    """Califica automáticamente las entregas usando la red neuronal y actualiza evaluaciones."""
    try:
        # Obtener todas las entregas pendientes de calificación
        query = "SELECT id, practica_id, estudiante_id, archivos_url FROM entregas WHERE estado = 'pendiente'"
        generador.cursor.execute(query)
        entregas = generador.cursor.fetchall()
        
        if not entregas:
            flash("No hay entregas pendientes para calificar.", "info")
            return redirect(url_for('evaluacion'))
        
        for entrega in entregas:
            entrega_id = entrega['id']
            practica_id = entrega['practica_id']
            estudiante_id = entrega['estudiante_id']
            archivos_url = entrega['archivos_url']  # Asegúrate de que este campo esté poblado
            
            # Obtener detalles de la práctica
            practica = generador.obtener_practica_por_id(practica_id)
            
            if not practica:
                flash(f"Práctica ID {practica_id} no encontrada.", "warning")
                continue  # Si la práctica no existe, saltar a la siguiente entrega
            
            if not practica.uso_ia:
                continue  # Solo calificar si la práctica tiene activada la IA
            
            # Verificar si el archivo existe
            ruta_archivo = os.path.join(app.config['UPLOAD_FOLDER'], archivos_url)
            if not os.path.exists(ruta_archivo):
                flash(f"Archivo no encontrado para la entrega ID {entrega_id}: {archivos_url}", "warning")
                continue
                
            # Leer el contenido del archivo
            contenido_entrega = leer_contenido_archivo(ruta_archivo)
            
            if not contenido_entrega:
                flash(f"No se pudo leer el contenido del archivo para la entrega ID {entrega_id}", "warning")
                continue
            
            # Generar calificación y comentarios con la IA
            resultado_ml = modelo_ml.analizar_contenido(
                contenido_entrega,
                practica.titulo,
                practica.objetivo
            )
            
            calificacion = resultado_ml.get('calificacion')
            comentarios = resultado_ml.get('comentarios')
            
            # Verificar si se generó una calificación válida
            if calificacion is None or comentarios is None:
                flash(f"No se pudo generar calificación para la entrega ID {entrega_id}.", "warning")
                continue
            
            # Al calificar la entrega
            query_eval = """
                UPDATE evaluaciones 
                SET calificacion = %s, comentarios = %s, estado = 'calificado' 
                WHERE practica_id = %s AND estudiante_id = %s
            """
            generador.cursor.execute(query_eval, (calificacion, comentarios, practica_id, estudiante_id))
            
            # Marcar la entrega como calificada
            query_update = "UPDATE entregas SET estado = 'calificado' WHERE id = %s"
            generador.cursor.execute(query_update, (entrega_id,))
        
        generador.connection.commit()
        flash("Entregas calificadas automáticamente.", "success")
    except Exception as e:
        generador.connection.rollback()
        flash(f"Error en la calificación automática: {str(e)}", "error")
    
    return redirect(url_for('evaluacion'))

@app.route('/api/calificar/<int:evaluacion_id>', methods=['POST'])
@login_required
def calificar_entrega(evaluacion_id):
    """Califica manualmente una entrega."""
    data = request.get_json()
    calificacion = data.get('calificacion')
    comentarios = data.get('comentarios')

    try:
        # Actualizar la evaluación en la base de datos
        query = """
            UPDATE evaluaciones 
            SET calificacion = %s, comentarios = %s, estado = 'calificado' 
            WHERE id = %s
        """
        generador.cursor.execute(query, (calificacion, comentarios, evaluacion_id))
        generador.connection.commit()

        return jsonify({"success": True}), 200
    except Exception as e:
        generador.connection.rollback()
        return jsonify({"error": str(e)}), 500

@app.route('/eliminar_evaluacion/<int:evaluacion_id>', methods=['POST'])
def eliminar_evaluacion(evaluacion_id):
    try:
        generador.eliminar_evaluacion(evaluacion_id)  # Ahora usa una función en generador_practicas.py
        flash("Evaluación eliminada correctamente", "success")
    except Exception as e:
        flash(f"Error al eliminar evaluación: {str(e)}", "error")

    return redirect(url_for('evaluacion'))

@app.route('/eliminar_practica/<int:practica_id>', methods=['POST'])
@login_required
def eliminar_practica(practica_id):
    """Elimina una práctica."""
    if current_user.rol not in ['administrador', 'profesor']:
        flash('No tienes permisos para realizar esta acción.', 'error')
        return redirect(url_for('practicas'))
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Verificar que la práctica exista y pertenezca al profesor actual (si no es administrador)
    if current_user.rol == 'profesor':
        cursor.execute("""
            SELECT id FROM practicas WHERE id = %s AND autor_id = %s
        """, (practica_id, current_user.id))
        practica = cursor.fetchone()
        if not practica:
            connection.close()
            flash('No tienes permisos para eliminar esta práctica.', 'error')
            return redirect(url_for('practicas'))
    
    try:
        # Eliminar asignaciones relacionadas
        cursor.execute("DELETE FROM asignaciones WHERE practica_id = %s", (practica_id,))
        
        # Eliminar evaluaciones relacionadas
        cursor.execute("DELETE FROM evaluaciones WHERE practica_id = %s", (practica_id,))
        
        # Eliminar la práctica
        cursor.execute("DELETE FROM practicas WHERE id = %s", (practica_id,))
        
        connection.commit()
        flash('Práctica eliminada exitosamente.', 'success')
    except Exception as e:
        connection.rollback()
        flash(f'Error al eliminar la práctica: {str(e)}', 'error')
    finally:
        connection.close()
    
    return redirect(url_for('practicas'))


@app.route('/generar_practica', methods=['GET', 'POST'])
def generar_practica_view():
    if request.method == 'POST':
        titulo = request.form['titulo']
        objetivo = request.form['objetivo']
        
        try:
            # Generar práctica utilizando el modelo ML
            practica_generada = modelo_ml.generar_practica(titulo, objetivo)
            
            # Guardar los datos generados para posterior presentación
            return render_template('resultado.html', 
                                   practica=practica_generada, 
                                   titulo=titulo, 
                                   objetivo=objetivo)
        except Exception as e:
            flash(f"Error al generar práctica: {str(e)}", "error")
            return redirect(url_for('generar_practica_view'))
    
    # Para solicitud GET
    return render_template('generar_practica.html')

@app.route('/usuarios', methods=['GET', 'POST'])
def usuarios():
    if request.method == 'POST':
        nombre = request.form['nombre']
        email = request.form['email']
        rol = request.form['rol']
        
        usuario = Usuario(
            id=None,
            nombre=nombre,
            email=email,
            rol=rol
        )
        
        try:
            generador.agregar_usuario(usuario)
            flash("Usuario agregado exitosamente", "success")
        except Exception as e:
            flash(f"Error al agregar usuario: {str(e)}", "error")
        
        return redirect(url_for('usuarios'))
    
    # Para solicitud GET
    usuarios = generador.obtener_usuarios()
    return render_template('usuarios.html', usuarios=usuarios)

@app.route('/evaluacion', methods=['GET', 'POST'])
def evaluacion():
    if request.method == 'POST':
        # Manejar la actualización de evaluaciones
        evaluacion_id = int(request.form['evaluacion_id'])
        comentario = request.form['comentario']
        
        calificacion_str = request.form.get('calificacion', '').strip()
        calificacion = float(calificacion_str) if calificacion_str else None  # Evitar error de conversión
        
        try:
            generador.actualizar_evaluacion(evaluacion_id, comentario, calificacion)
            flash("Evaluación actualizada exitosamente", "success")
        except Exception as e:
            flash(f"Error al actualizar evaluación: {str(e)}", "error")
        
        return redirect(url_for('evaluacion'))

    # Método GET: Mostrar evaluación o vista general
    entrega_id = request.args.get('entrega_id')
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    try:
        if entrega_id:
            # Obtener la entrega
            cursor.execute("""
                SELECT e.*, p.titulo as practica_titulo, p.descripcion as practica_descripcion
                FROM entregas e
                JOIN practicas p ON e.practica_id = p.id
                WHERE e.id = %s
            """, (entrega_id,))
            entrega = cursor.fetchone()
            if not entrega:
                flash('Entrega no encontrada', 'error')
                return redirect(url_for('dashboard'))

            # Obtener información del estudiante
            cursor.execute("""
                SELECT u.id, u.nombre, u.email, u.numero_cuenta
                FROM usuarios u
                WHERE u.id = %s
            """, (entrega['estudiante_id'],))
            estudiante = cursor.fetchone()

            if not estudiante:
                flash('Estudiante no encontrado', 'error')
                return redirect(url_for('dashboard'))

            if 'id' not in estudiante:
                estudiante = {
                    'id': entrega['estudiante_id'],
                    'nombre': 'Estudiante ID: ' + str(entrega['estudiante_id']),
                    'email': 'No disponible',
                    'numero_cuenta': 'No disponible'
                }

            return render_template('evaluacion.html', entrega=entrega, estudiante=estudiante)

        # Si no hay entrega_id, mostrar vista general de evaluaciones
        todas_practicas = generador.obtener_practicas()
        practicas_por_materia = defaultdict(list)

        for practica in todas_practicas:
            materia_id = practica.materia_id
            estudiantes = generador.obtener_estudiantes_por_practica(practica.id)

            for estudiante in estudiantes:
                estudiante['archivo_url'] = estudiante.get('archivos_url', '')
                estudiante['estado'] = estudiante.get('estado', 'pendiente')
            practica.estudiantes = estudiantes
            practicas_por_materia[materia_id].append(practica)

        cursor.execute("SELECT id, nombre FROM materias")
        materias = cursor.fetchall()

        cursor.execute("""
            SELECT e.id AS entrega_id, e.practica_id, e.estudiante_id, e.evaluacion_id, e.fecha_entrega, e.estado, 
                   e.archivos_url, e.calificacion, e.contenido, 
                   p.titulo AS practica_titulo, p.uso_ia, u.nombre AS estudiante_nombre
            FROM entregas e
            JOIN practicas p ON e.practica_id = p.id
            JOIN usuarios u ON e.estudiante_id = u.id
        """)
        entregas = cursor.fetchall()

        for entrega in entregas:
            practica_id = entrega['practica_id']
            for practica in practicas_por_materia.get(practica_id, []):
                for estudiante in practica.estudiantes:
                    estudiante_id = estudiante.get('id')
                    if estudiante_id is not None and estudiante_id == entrega['estudiante_id']:
                        estudiante['estado'] = entrega['estado']
                        estudiante['archivo_url'] = entrega['archivos_url']
                        estudiante['evaluacion_id'] = entrega['evaluacion_id']
                        estudiante['uso_ia'] = entrega['uso_ia']

        todas_evaluaciones = generador.obtener_evaluaciones()
        evaluaciones_pendientes = [ev for ev in todas_evaluaciones if ev['calificacion'] is None]
        evaluaciones_calificadas = [ev for ev in todas_evaluaciones if ev['calificacion'] is not None]

        usuario_id = str(current_user.id)
        usuarios = cargar_json('usuarios.json')
        usuario = usuarios.get(usuario_id, {})

        return render_template(
            'evaluacion.html',
            practicas_por_materia=practicas_por_materia,
            materias=materias,
            evaluaciones=evaluaciones_pendientes,
            evaluaciones_calificadas=evaluaciones_calificadas,
            usuario=usuario
        )

    except Exception as e:
        flash(f'Error al cargar la evaluación: {str(e)}', 'error')
        return redirect(url_for('inicio'))
    
    finally:
        cursor.close()
        connection.close()

# Add this route for saving the profile
@app.route('/guardar_perfil_estudiante', methods=['POST'])
@login_required
def guardar_perfil_estudiante():
    """Guarda el perfil del estudiante en la base de datos."""
    if current_user.rol != 'estudiante':
        flash('Solo los estudiantes pueden completar este perfil.', 'error')
        return redirect(url_for('inicio'))
    
    try:
        # Obtener datos del formulario
        semestre = request.form.get('semestre')
        facultad = request.form.get('facultad')
        carrera = request.form.get('carrera')
        
        # Obtener estilos de aprendizaje seleccionados
        estilos_aprendizaje = request.form.getlist('estilos_aprendizaje')
        estilos_str = ','.join(estilos_aprendizaje) if estilos_aprendizaje else ''
        
        # Guardar en la base de datos
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Verificar si ya existe un perfil para este estudiante
        query_check = "SELECT id FROM perfiles_estudiante WHERE estudiante_id = %s"
        cursor.execute(query_check, (current_user.id,))
        perfil_existente = cursor.fetchone()
        
        if perfil_existente:
            # Actualizar perfil existente
            query = """
            UPDATE perfiles_estudiante 
            SET semestre = %s, facultad = %s, carrera = %s, estilos_aprendizaje = %s, 
                fecha_actualizacion = NOW() 
            WHERE estudiante_id = %s
            """
            cursor.execute(query, (semestre, facultad, carrera, estilos_str, current_user.id))
            print(f"Perfil actualizado para estudiante_id={current_user.id}")
        else:
            # Crear nuevo perfil
            query = """
            INSERT INTO perfiles_estudiante 
            (estudiante_id, semestre, facultad, carrera, estilos_aprendizaje, fecha_creacion) 
            VALUES (%s, %s, %s, %s, %s, NOW())
            """
            cursor.execute(query, (current_user.id, semestre, facultad, carrera, estilos_str))
            print(f"Nuevo perfil creado para estudiante_id={current_user.id}")
        
        # Marcar como completado en la tabla usuarios - IMPORTANTE: Usar UPDATE DIRECTO
        query_update_user = """
        UPDATE usuarios 
        SET perfil_completado = 1 
        WHERE id = %s
        """
        cursor.execute(query_update_user, (current_user.id,))
        affected_rows = cursor.rowcount
        print(f"Actualización de perfil_completado afectó a {affected_rows} filas")
        
        connection.commit()
        
        # Verificar que el perfil_completado se haya actualizado correctamente
        cursor.execute("SELECT perfil_completado FROM usuarios WHERE id = %s", (current_user.id,))
        updated_user = cursor.fetchone()
        print(f"Valor de perfil_completado después de la actualización: {updated_user[0] if updated_user else 'No encontrado'}")
        
        connection.close()
        
        # Actualizar la información en el archivo JSON de usuarios
        usuario_id = str(current_user.id)
        usuarios_path = 'usuarios.json'
        
        # Asegurarse de que el archivo existe
        if not os.path.exists(usuarios_path):
            with open(usuarios_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=4, ensure_ascii=False)
                print(f"Archivo {usuarios_path} creado")
        
        # Leer el archivo existente
        try:
            with open(usuarios_path, 'r', encoding='utf-8') as f:
                try:
                    usuarios = json.load(f)
                    print(f"Archivo {usuarios_path} cargado correctamente")
                except json.JSONDecodeError:
                    print(f"Error al decodificar {usuarios_path}, creando nuevo diccionario")
                    usuarios = {}
        except Exception as e:
            print(f"Error al abrir {usuarios_path}: {str(e)}")
            usuarios = {}
        
        # Actualizar o crear la entrada del usuario
        if usuario_id not in usuarios:
            usuarios[usuario_id] = {}
            print(f"Creando nueva entrada para usuario_id={usuario_id}")
        
        # Actualizar datos
        usuarios[usuario_id]['nombre'] = current_user.nombre
        usuarios[usuario_id]['email'] = current_user.email
        usuarios[usuario_id]['numero_cuenta'] = current_user.numero_cuenta
        usuarios[usuario_id]['semestre'] = semestre
        usuarios[usuario_id]['facultad'] = facultad
        usuarios[usuario_id]['carrera'] = carrera
        usuarios[usuario_id]['estilos_aprendizaje'] = estilos_str
        
        # Guardar el archivo actualizado
        try:
            with open(usuarios_path, 'w', encoding='utf-8') as f:
                json.dump(usuarios, f, indent=4, ensure_ascii=False)
                print(f"Archivo {usuarios_path} actualizado correctamente")
        except Exception as e:
            print(f"Error al guardar {usuarios_path}: {str(e)}")
        
        flash('¡Perfil completado con éxito! Bienvenido a IntelliTutor UNAM.', 'success')
        return redirect(url_for('inicio'))
        
    except Exception as e:
        print(f"Error en guardar_perfil_estudiante: {str(e)}")
        flash(f'Error al guardar el perfil: {str(e)}', 'error')
        return redirect(url_for('perfil_estudiante'))

    
@app.route('/guardar_calificacion', methods=['POST'])
def guardar_calificacion():
    """Guarda la calificación y los comentarios de una evaluación manual."""
    try:
        evaluacion_id = request.form['evaluacion_id']
        calificacion = request.form['calificacion']
        comentarios = request.form['comentarios']

        # Lógica para guardar la calificación en la base de datos
        with closing(get_db_connection()) as connection:
            cursor = connection.cursor()
            query = """
            UPDATE evaluaciones
            SET calificacion = %s, comentarios = %s, estado = 'calificado'
            WHERE id = %s
            """
            cursor.execute(query, (calificacion, comentarios, evaluacion_id))
            connection.commit()

        flash("Calificación guardada exitosamente.", "success")
    except Exception as e:
        flash(f"Error al guardar la calificación: {str(e)}", "error")

    return redirect(url_for('evaluacion'))

# @app.route('/practica/<int:practica_id>', methods=['GET'])
# def ver_practica(practica_id):
#     practica = generador.obtener_practica_por_id(practica_id)
#     contenido_generado = generador.obtener_contenido_generado(practica_id)
    
#     if not practica:
#         flash("Práctica no encontrada.", "error")
#         return redirect(url_for('practicas'))  # Redirigir a la lista de prácticas

#     # Generar datos adicionales con el modelo ML
#     titulo = practica.titulo
#     objetivo = practica.objetivo
#     practica_generada = modelo_ml.generar_practica(titulo, objetivo)
    
#     prediccion_exito = 'Alta'  # Ejemplo de predicción de éxito
#     recomendaciones_personalizadas = 'Recomendaciones personalizadas generadas por IA'  # Ejemplo de recomendaciones
    
#     modelo_ml_data = {
#         'prediccion_exito': prediccion_exito,
#         'recomendaciones_personalizadas': recomendaciones_personalizadas
#     }
    
#     # Crear el diccionario contenido_generado
#     contenido_generado = {
#         'descripcion': practica_generada['descripcion'],
#         'objetivos_aprendizaje': practica_generada['objetivos_aprendizaje'],
#         'actividades': practica_generada['actividades'],
#         'recursos': practica_generada['recursos'],
#         'criterios_evaluacion': practica_generada['criterios_evaluacion'],
#         'recomendaciones': practica_generada['recomendaciones']
#     }
    
#     print(f"Práctica: {practica}")
#     print(f"Contenido generado: {contenido_generado}")
#     print(f"Datos generados por IA: {modelo_ml_data}")
    
#     return render_template('ver_practica.html', practica=practica, contenido=contenido_generado, modelo_ml=modelo_ml_data)

# Añade esta ruta a tu archivo app.py

@app.route('/ver_practica_detalle/<int:practica_id>')
@login_required
def ver_practica_detalle(practica_id):
    """Muestra los detalles de una práctica."""
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
            flash("Práctica no encontrada.", "error")
            return redirect(url_for('practicas'))
        
        # Obtener contenido generado si existe
        cursor.execute("""
            SELECT contenido FROM contenido_generado WHERE practica_id = %s
        """, (practica_id,))
        contenido_generado_row = cursor.fetchone()
        
        # Obtener evaluación si existe
        cursor.execute("""
            SELECT * FROM evaluaciones WHERE practica_id = %s AND estudiante_id = %s
        """, (practica_id, current_user.id))
        evaluacion = cursor.fetchone()
        
        connection.close()
        
        # Si hay contenido generado, procesarlo
        if contenido_generado_row and contenido_generado_row['contenido']:
            contenido_generado = json.loads(contenido_generado_row['contenido'])
        else:
            # Generar datos de ejemplo si no hay contenido
            contenido_generado = {
                'descripcion': f"Descripción de la práctica: {practica['titulo']}",
                'objetivos_aprendizaje': [
                    "Comprender los conceptos básicos relacionados con el tema",
                    "Aplicar los conocimientos adquiridos en situaciones prácticas",
                    "Desarrollar habilidades de análisis y resolución de problemas"
                ],
                'actividades': [
                    "Investigar sobre el tema asignado",
                    "Realizar ejercicios prácticos",
                    "Elaborar un informe con los resultados obtenidos"
                ],
                'recursos': [
                    {'titulo': 'Material de clase', 'descripcion': 'Consultar las presentaciones y apuntes de clase'},
                    {'titulo': 'Biblioteca digital', 'descripcion': 'Acceder a los recursos electrónicos de la biblioteca UNAM'}
                ],
                'criterios_evaluacion': [
                    {'titulo': 'Contenido', 'descripcion': 'Profundidad y precisión del contenido (40%)'},
                    {'titulo': 'Metodología', 'descripcion': 'Aplicación correcta de la metodología (30%)'},
                    {'titulo': 'Presentación', 'descripcion': 'Claridad y organización del trabajo (20%)'},
                    {'titulo': 'Originalidad', 'descripcion': 'Aportaciones originales y creatividad (10%)'}
                ],
                'recomendaciones': [
                    "Comenzar con suficiente anticipación",
                    "Consultar múltiples fuentes de información",
                    "Revisar cuidadosamente el trabajo antes de entregarlo"
                ]
            }
        
        # Procesar evaluación si existe
        if evaluacion:
            # Convertir campos JSON si existen
            if evaluacion.get('metricas') and isinstance(evaluacion['metricas'], str):
                evaluacion['metricas'] = json.loads(evaluacion['metricas'])
            
            if evaluacion.get('fortalezas') and isinstance(evaluacion['fortalezas'], str):
                evaluacion['fortalezas'] = json.loads(evaluacion['fortalezas'])
            else:
                evaluacion['fortalezas'] = ["Buena estructura", "Contenido relevante", "Claridad en la exposición"]
            
            if evaluacion.get('debilidades') and isinstance(evaluacion['debilidades'], str):
                evaluacion['debilidades'] = json.loads(evaluacion['debilidades'])
            else:
                evaluacion['debilidades'] = ["Podría mejorar la profundidad", "Algunas secciones requieren más desarrollo"]
            
            if evaluacion.get('recursos_recomendados') and isinstance(evaluacion['recursos_recomendados'], str):
                evaluacion['recursos_recomendados'] = json.loads(evaluacion['recursos_recomendados'])
            else:
                evaluacion['recursos_recomendados'] = [
                    {'titulo': 'Guía de estudio', 'url': 'https://www.unam.mx/recursos/guia-estudio'},
                    {'titulo': 'Biblioteca digital UNAM', 'url': 'https://www.bibliotecadigital.unam.mx'}
                ]
            
            # Asegurar que existan campos necesarios
            if 'relevancia' not in evaluacion:
                evaluacion['relevancia'] = 85
            
            if 'comentarios' not in evaluacion:
                evaluacion['comentarios'] = "Buen trabajo en general. La práctica cumple con los objetivos establecidos."
            
            if 'sugerencias' not in evaluacion:
                evaluacion['sugerencias'] = "Para mejorar, considera profundizar más en los conceptos teóricos y añadir más ejemplos prácticos."
        
        # Registrar actividad
        from routes.actividades import registrar_actividad_sistema
        registrar_actividad_sistema(
            f"Visualización de detalles de la práctica: {practica['titulo']}",
            'practica',
            practica_id
        )
        
        return render_template(
            'ver_practica_detalle.html',
            practica=practica,
            contenido=contenido_generado,
            evaluacion=evaluacion
        )
    except Exception as e:
        flash(f"Error al cargar los detalles de la práctica: {str(e)}", "error")
        return redirect(url_for('practicas'))

@app.route('/api/generar_practica', methods=['POST'])
def api_generar_practica():
    """API para generar prácticas automáticamente"""
    datos = request.get_json()
    
    if not datos or 'titulo' not in datos or 'objetivo' not in datos:
        return jsonify({'error': 'Se requieren título y objetivo'}), 400
    
    try:
        practica_generada = modelo_ml.generar_practica(datos['titulo'], datos['objetivo'])
        return jsonify({'practica': practica_generada}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/usuarios', methods=['GET'])
@login_required
def get_usuarios():
    """Obtiene la lista de usuarios para los selectores."""
    if current_user.rol not in ['administrador', 'profesor']:
        return jsonify({'success': False, 'error': 'No tienes permisos para realizar esta acción.'}), 403
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    cursor.execute("""
        SELECT id, nombre, rol FROM usuarios WHERE rol = 'estudiante'
    """)
    
    usuarios = cursor.fetchall()
    connection.close()
    
    return jsonify({'success': True, 'usuarios': usuarios})


# Modificar la función de login para crear configuración por defecto si no existe
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        numero_cuenta = request.form['numero_cuenta']
        password = request.form['password']

        # Intento de autenticación
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute(
            "SELECT id, nombre, rol, password_hash, email, numero_cuenta, perfil_completado FROM usuarios WHERE numero_cuenta = %s",
            (numero_cuenta,)
        )
        usuario = cursor.fetchone()
        connection.close()

        if usuario and check_password_hash(usuario['password_hash'], password):
            # Si las credenciales son correctas, iniciar sesión
            session['usuario_id'] = usuario['id']
            session['usuario_nombre'] = usuario['nombre']
            session['rol'] = usuario['rol']
            session['inicio_sesion'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # Crear objeto UserLogin para Flask-Login
            user = UserLogin(
                id=usuario['id'],
                nombre=usuario['nombre'],
                email=usuario['email'],
                rol=usuario['rol'],
                numero_cuenta=usuario['numero_cuenta']
            )
            login_user(user)  # Iniciar sesión con Flask-Login
            
            # Verificar si el usuario tiene configuración, si no, crear una por defecto
            crear_configuracion_por_defecto(usuario['id'])
            
            registrar_sesion_usuario('login')
            
            # Si es estudiante y no ha completado su perfil, redirigir al formulario de perfil
            if usuario['rol'] == 'estudiante' and not usuario.get('perfil_completado'):
                return redirect(url_for('perfil_estudiante'))  # Usar la ruta correcta
                        
            return redirect(url_for('inicio'))
        else:
            flash('Credenciales inválidas', 'error')
    año_actual = datetime.now().year
    return render_template('login.html', año_actual=año_actual)


@app.route('/aulas', methods=['GET', 'POST'])
@login_required
def aulas():
    """Gestión de aulas virtuales."""
    usuarios = cargar_json('usuarios.json')  # Cargar el archivo de usuarios
    usuario_id = str(current_user.id)  # Obtener el ID del usuario actual
    usuario = usuarios.get(usuario_id, {})  # Obtener el usuario actual
    if current_user.rol not in ['administrador', 'profesor']:
        flash('No tienes permisos para acceder a esta página.', 'error')
        return redirect(url_for('inicio'))
    
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Procesar formulario para crear grupo
    if request.method == 'POST':
        if 'materia_id' in request.form:  # Crear nuevo grupo
            materia_id = request.form['materia_id']
            semestre_id = request.form['semestre_id']
            turno = request.form['turno']
            grupo_numero = request.form['grupo_numero']
            descripcion = request.form['descripcion']
            fecha_inicio = request.form.get('fecha_inicio')
            fecha_fin = request.form.get('fecha_fin')
        
            # Criterios de evaluación
            practicas_porcentaje = request.form.get('practicas_porcentaje', 40)
            examenes_porcentaje = request.form.get('examenes_porcentaje', 30)
            proyectos_porcentaje = request.form.get('proyectos_porcentaje', 20)
            asistencia_porcentaje = request.form.get('asistencia_porcentaje', 10)
        
            # Validar que los porcentajes sumen 100%
            total_porcentaje = float(practicas_porcentaje) + float(examenes_porcentaje) + float(proyectos_porcentaje) + float(asistencia_porcentaje)
            if total_porcentaje != 100:
                flash('Los porcentajes de evaluación deben sumar 100%.', 'error')
                return redirect(url_for('aulas'))
        
            # Obtener nombres desde la BD
            cursor.execute("SELECT nombre FROM materias WHERE id = %s", (materia_id,))
            materia = cursor.fetchone()
            
            cursor.execute("SELECT nombre FROM semestres WHERE id = %s", (semestre_id,))
            semestre = cursor.fetchone()
            
            if not materia or not semestre:
                flash('Materia o semestre no válidos.', 'error')
                return redirect(url_for('aulas'))

            nombre_grupo = f"{materia['nombre']} {semestre['nombre']}{grupo_numero}"
        
            cursor.execute("""
                INSERT INTO grupos (nombre, descripcion, profesor_id, materia_id, semestre_id, turno, fecha_inicio, fecha_fin)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (nombre_grupo, descripcion, current_user.id, materia_id, semestre_id, turno, fecha_inicio, fecha_fin))
        
            grupo_id = cursor.lastrowid
        
            cursor.execute("""
                INSERT INTO criterios_evaluacion 
                (grupo_id, practicas_porcentaje, examenes_porcentaje, proyectos_porcentaje, asistencia_porcentaje)
                VALUES (%s, %s, %s, %s, %s)
            """, (grupo_id, practicas_porcentaje, examenes_porcentaje, proyectos_porcentaje, asistencia_porcentaje))
        
            connection.commit()
            flash('Grupo creado exitosamente.', 'success')
            return redirect(url_for('aulas'))

            
        elif 'editar_grupo_id' in request.form:  # Editar grupo existente
            grupo_id = request.form['editar_grupo_id']

            if request.form.get('descripcion'):
                descripcion = request.form['descripcion']
                turno = request.form['turno']
                semestre_id = request.form['semestre_id']
                fecha_inicio = request.form.get('fecha_inicio')
                fecha_fin = request.form.get('fecha_fin')
            
            # Criterios de evaluación
            practicas_porcentaje = request.form.get('practicas_porcentaje')
            examenes_porcentaje = request.form.get('examenes_porcentaje')
            proyectos_porcentaje = request.form.get('proyectos_porcentaje')
            asistencia_porcentaje = request.form.get('asistencia_porcentaje')
            
            # Validar que los porcentajes sumen 100% si se proporcionan
            if practicas_porcentaje and examenes_porcentaje and proyectos_porcentaje and asistencia_porcentaje:
                total_porcentaje = float(practicas_porcentaje) + float(examenes_porcentaje) + float(proyectos_porcentaje) + float(asistencia_porcentaje)
                if total_porcentaje != 100:
                    flash('Los porcentajes de evaluación deben sumar 100%.', 'error')
                    return redirect(url_for('aulas'))
                
                # Actualizar criterios de evaluación
                cursor.execute("""
                               UPDATE criterios_evaluacion
                               SET practicas_porcentaje = %s, examenes_porcentaje = %s, proyectos_porcentaje = %s, asistencia_porcentaje = %s
                               WHERE grupo_id = %s
                """, (practicas_porcentaje, examenes_porcentaje, proyectos_porcentaje, asistencia_porcentaje, grupo_id))
            
            if request.form.get('descripcion'):
                # Actualizar grupo
                cursor.execute("""
                    UPDATE grupos 
                    SET descripcion = %s, semestre_id = %s, fecha_inicio = %s, fecha_fin = %s, turno = %s
                    WHERE id = %s
                """, (descripcion, semestre_id, fecha_inicio, fecha_fin, turno, grupo_id))

            
            connection.commit()
            flash('Grupo actualizado exitosamente.', 'success')
            return redirect(url_for('aulas'))
    
    # Obtener datos para la vista
    if current_user.rol == 'administrador':
        cursor.execute("""
            SELECT g.*, m.nombre as materia_nombre, u.nombre as profesor_nombre
            FROM grupos g
            JOIN materias m ON g.materia_id = m.id
            JOIN usuarios u ON g.profesor_id = u.id
        """)
    else:  # Profesor
        cursor.execute("""
            SELECT g.*, m.nombre as materia_nombre, u.nombre as profesor_nombre
            FROM grupos g
            JOIN materias m ON g.materia_id = m.id
            JOIN usuarios u ON g.profesor_id = u.id
            WHERE g.profesor_id = %s
        """, (current_user.id,))
    
    grupos_raw = cursor.fetchall()
    
    # Procesar los datos de los grupos
    grupos = {}
    for grupo in grupos_raw:
        grupo_id = str(grupo['id'])
        grupos[grupo_id] = {
            'id': grupo['id'],
            'nombre': grupo['nombre'],
            'descripcion': grupo['descripcion'],
            'turno': grupo['turno'],
            'materia_nombre': grupo['materia_nombre'],
            'profesor_nombre': grupo['profesor_nombre'],
            'fecha_inicio': grupo['fecha_inicio'],
            'fecha_fin': grupo['fecha_fin'],
            'semestre_id': grupo.get('semestre_id'),
            'estudiantes': [],
            'total_practicas': 0,
            'practicas': []
        }
        
        # Obtener miembros del grupo desde la tabla grupo_miembros
        cursor.execute("""
            SELECT u.id, u.nombre
            FROM usuarios u
            JOIN grupo_miembros gm ON u.id = gm.usuario_id
            WHERE gm.grupo_id = %s
        """, (grupo['id'],))
        miembros = cursor.fetchall()
        grupos[grupo_id]['estudiantes'] = [m['nombre'] for m in miembros]
        grupos[grupo_id]['estudiantes_data'] = miembros
        
        # Obtener prácticas del grupo
        cursor.execute("""
            SELECT id, titulo, tipo_asignacion
            FROM practicas
            WHERE grupo_id = %s
        """, (grupo['id'],))
        practicas = cursor.fetchall()
        grupos[grupo_id]['practicas'] = practicas
        grupos[grupo_id]['total_practicas'] = len(practicas)
        
        # Obtener criterios de evaluación
        cursor.execute("""
            SELECT * FROM criterios_evaluacion WHERE grupo_id = %s
        """, (grupo['id'],))
        criterios = cursor.fetchone()
        cursor.fetchall()
        if criterios:
            grupos[grupo_id]['criterios'] = criterios
    
    # Obtener materias y semestres para los formularios
    cursor.execute("SELECT id, nombre FROM materias")
    materias = cursor.fetchall()
    
    cursor.execute("SELECT id, nombre FROM semestres")
    semestres = cursor.fetchall()
    
    connection.close()
    
    return render_template(
        'aulas.html',
        grupos=grupos,
        usuario=usuario,
        materias=materias,
        semestres=semestres,
        año_actual=datetime.now().year
    )

@app.route('/editar_grupo/<int:grupo_id>', methods=['GET', 'POST'])
def editar_grupo(grupo_id):
    if request.method == 'POST':
        # Obtener los datos del formulario
        descripcion = request.form['descripcion']
        turno = request.form['turno']
        semestre_id = request.form['semestre_id']

        # Actualizar el grupo en la base de datos
        try:
            query = """
            UPDATE grupos
            SET descripcion = %s, turno = %s, semestre_id = %s
            WHERE id = %s
            """
            connection = get_db_connection()
            cursor = connection.cursor()
            cursor.execute(query, (descripcion, turno, semestre_id, grupo_id))
            connection.commit()
            flash("Grupo actualizado exitosamente", "success")
        except Exception as e:
            connection.rollback()
            flash(f"Error al actualizar grupo: {str(e)}", "error")
        finally:
            connection.close()

        return redirect(url_for('aulas'))  # Redirigir a la lista de aulas

    # Manejo de la visualización del formulario (GET)
    grupo = generador.obtener_grupo_por_id(grupo_id)  # Asegúrate de tener esta función
    semestres = generador.obtener_semestres()
    return render_template('editar_grupo.html', grupo=grupo, semestres=semestres)

@app.route('/ver_detalles_grupo/<int:grupo_id>', methods=['GET'])
def ver_detalles_grupo(grupo_id):
    grupo = generador.obtener_grupo_por_id(grupo_id)  # Asegúrate de tener este método
    estudiantes = generador.obtener_estudiantes_por_grupo(grupo_id)  # Método para obtener estudiantes en el grupo

    return render_template('ver_detalles_grupo.html', grupo=grupo, estudiantes=estudiantes)

@app.route('/gestionar_estudiantes/<int:grupo_id>', methods=['POST'])
@login_required
def gestionar_estudiantes(grupo_id):
    """Gestiona los estudiantes de un grupo."""
    if current_user.rol not in ['administrador', 'profesor']:
        return jsonify({'success': False, 'error': 'No tienes permisos para realizar esta acción.'}), 403
    
    # Verificar que el grupo exista y pertenezca al profesor actual (si no es administrador)
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    if current_user.rol == 'profesor':
        cursor.execute("""
            SELECT id FROM grupos WHERE id = %s AND profesor_id = %s
        """, (grupo_id, current_user.id))
        grupo = cursor.fetchone()
        if not grupo:
            connection.close()
            return jsonify({'success': False, 'error': 'No tienes permisos para gestionar este grupo.'}), 403
    
    data = request.json

    # Asegurar que usuario_id es entero
    try:
        usuario_id = int(data.get('usuario_id'))
    except (ValueError, TypeError):
        usuario_id = None

    accion = data.get('accion')

    print("Datos recibidos:", data)
    print("usuario_id:", usuario_id, "| accion:", accion)

    if not usuario_id or not accion:
        connection.close()
        return jsonify({'success': False, 'error': 'Datos incompletos.'}), 400

    
    try:
        if accion == 'agregar':
            # Verificar que el usuario exista y sea estudiante
            cursor.execute("""
                SELECT id, nombre FROM usuarios WHERE id = %s AND rol = 'estudiante'
            """, (usuario_id,))
            estudiante = cursor.fetchone()
            
            if not estudiante:
                connection.close()
                return jsonify({'success': False, 'error': 'El estudiante no existe o no es un estudiante.'}), 404
            
            # Verificar que el estudiante no esté ya en el grupo
            cursor.execute("""
                SELECT 1 FROM grupo_miembros WHERE grupo_id = %s AND usuario_id = %s
            """, (grupo_id, usuario_id))
            
            if cursor.fetchone():
                connection.close()
                return jsonify({'success': False, 'error': 'El estudiante ya está inscrito en este grupo.'}), 400
            
            # Agregar el estudiante al grupo
            cursor.execute("""
                INSERT INTO grupo_miembros (grupo_id, usuario_id, rol) VALUES (%s, %s, 'estudiante')
            """, (grupo_id, usuario_id))
            
            connection.commit()
            
            # Devolver los datos del estudiante para actualizar la UI
            return jsonify({
                'success': True, 
                'estudiante': {
                    'id': estudiante['id'],
                    'nombre': estudiante['nombre']
                }
            })
            
        elif accion == 'eliminar':
            # Verificar que el estudiante esté en el grupo
            cursor.execute("""
                SELECT u.id, u.nombre 
                FROM usuarios u
                JOIN grupo_miembros gm ON u.id = gm.usuario_id
                WHERE gm.grupo_id = %s AND u.id = %s
            """, (grupo_id, usuario_id))
            
            estudiante = cursor.fetchone()
            if not estudiante:
                connection.close()
                return jsonify({'success': False, 'error': 'El estudiante no está inscrito en este grupo.'}), 404
            
            # Eliminar el estudiante del grupo
            cursor.execute("""
                DELETE FROM grupo_miembros WHERE grupo_id = %s AND usuario_id = %s
            """, (grupo_id, usuario_id))
            
            connection.commit()
            
            return jsonify({
                'success': True,
                'estudiante': {
                    'id': estudiante['id'],
                    'nombre': estudiante['nombre']
                }
            })
        else:
            connection.close()
            return jsonify({'success': False, 'error': 'Acción no válida.'}), 400
            
    except Exception as e:
        connection.rollback()
        connection.close()
        return jsonify({'success': False, 'error': str(e)}), 500
    
    finally:
        connection.close()
    
@app.route('/profesor_usuarios', methods=['GET', 'POST'])
def profesor_usuarios():
    if request.method == 'POST':
        try:
            # Obtener los datos del formulario
            nombre = request.form['nombre']
            email = request.form['email']
            password = request.form['password']

            # Validar que los campos no estén vacíos
            if not nombre.strip() or not email.strip() or not password.strip():
                flash("Todos los campos son obligatorios.", "error")
                return redirect(url_for('profesor_usuarios'))

            # Generar un hash seguro para la contraseña
            password_hash = generate_password_hash(password)

            # Registrar el usuario como estudiante
            rol = 'estudiante'  # Forzar el rol a 'estudiante'
            generador.registrar_usuario(nombre, email, password_hash, rol)
            flash("Estudiante agregado correctamente.", "success")
        except Exception as e:
            flash(f"Error al agregar estudiante: {str(e)}", "error")

        return redirect(url_for('profesor_usuarios'))

    # Obtener la lista de estudiantes registrados
    estudiantes = generador.obtener_usuarios_por_rol('estudiante')  # Filtrar solo estudiantes
    return render_template('profesor_usuarios.html', estudiantes=estudiantes)

# Ruta para ver sesiones (solo administradores)
@app.route('/admin/sesiones')
@login_required
def admin_sesiones():
    """Muestra las sesiones de los usuarios (solo para administradores)."""
    if current_user.rol != 'administrador':
        flash('No tienes permisos para acceder a esta página.', 'error')
        return redirect(url_for('inicio'))
    
    # Cargar sesiones
    sesiones_path = 'sesiones.json'
    if os.path.exists(sesiones_path):
        with open(sesiones_path, 'r', encoding='utf-8') as f:
            try:
                sesiones = json.load(f)
            except json.JSONDecodeError:
                sesiones = {}
    else:
        sesiones = {}
    
    # Obtener información de usuarios
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT id, nombre, rol FROM usuarios")
    usuarios_db = cursor.fetchall()
    connection.close()
    
    # Crear un diccionario de usuarios para búsqueda rápida
    usuarios_dict = {str(u['id']): u for u in usuarios_db}
    
    # Procesar sesiones para mostrar
    sesiones_procesadas = []
    for usuario_id, registros in sesiones.items():
        usuario = usuarios_dict.get(usuario_id, {'nombre': 'Usuario desconocido', 'rol': 'N/A'})
        
        for registro in registros:
            if registro['accion'] in ['login', 'logout'] and registro.get('duracion'):
                sesiones_procesadas.append({
                    'usuario_id': usuario_id,
                    'nombre': usuario['nombre'],
                    'rol': usuario['rol'],
                    'fecha': registro['fecha'],
                    'hora_inicio': registro.get('hora', 'N/A'),
                    'duracion': registro.get('duracion', 'N/A'),
                    'detalles': registro.get('detalles', {})
                })
    
    # Ordenar por fecha y hora (más recientes primero)
    sesiones_procesadas.sort(key=lambda x: (x['fecha'], x['hora_inicio']), reverse=True)
    
    return render_template(
        'admin_sesiones.html',
        sesiones=sesiones_procesadas,
        año_actual=datetime.now().year
    )

def registrar_usuario(nombre, email, password_hash, rol):
    numero_cuenta = generar_numero_cuenta()
    connection = get_db_connection()
    cursor = connection.cursor()
    query = """
        INSERT INTO usuarios (nombre, email, password_hash, rol, numero_cuenta)
        VALUES (%s, %s, %s, %s, %s)
    """
    cursor.execute(query, (nombre, email, password_hash, rol, numero_cuenta))
    connection.commit()
    cursor.close()
    connection.close()

def obtener_usuarios():
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT id, nombre, email, rol FROM usuarios")
    usuarios = cursor.fetchall()
    cursor.close()
    connection.close()
    return usuarios

@app.route('/admin_usuarios', methods=['GET', 'POST'])
def admin_usuarios():
    if request.method == 'POST':
        try:
            nombre = request.form['nombre']
            email = request.form['email']
            rol = request.form['rol']
            password = request.form['password']

            if not nombre.strip() or not email.strip() or not rol.strip() or not password.strip():
                flash("Todos los campos son obligatorios.", "error")
                return redirect(url_for('admin_usuarios'))

            password_hash = generate_password_hash(password)
            registrar_usuario(nombre, email, password_hash, rol)
            flash("Usuario agregado correctamente.", "success")
        except Exception as e:
            flash(f"Error al agregar usuario: {str(e)}", "error")

        return redirect(url_for('admin_usuarios'))

    usuarios = obtener_usuarios()
    return render_template('admin_usuarios.html', usuarios=usuarios)

@app.route('/actualizar_rol', methods=['POST'])
def actualizar_rol():
    usuario_id = request.form['usuario_id']
    nuevo_rol = request.form['nuevo_rol']
    try:
        connection = get_db_connection()
        cursor = connection.cursor()
        cursor.execute("UPDATE usuarios SET rol = %s WHERE id = %s", (nuevo_rol, usuario_id))
        connection.commit()
        cursor.close()
        connection.close()
        return jsonify({"success": True, "message": "Rol actualizado correctamente."})
    except Exception as e:
        return jsonify({"success": False, "message": f"Error al actualizar el rol: {str(e)}"}), 500

# Ruta para cerrar sesión
@app.route('/logout')
@login_required
def logout():
    """Cierra la sesión del usuario actual."""
    # Registrar cierre de sesión
    registrar_sesion_usuario('logout')
    
    logout_user()
    flash('Has cerrado sesión correctamente.', 'success')
    return redirect(url_for('login'))

@app.before_request
def actualizar_sesion_usuario():
    if 'usuario_id' in session:
        usuario_id = session['usuario_id']
        generador = GeneradorPracticasExtendido(DB_CONFIG)
        nuevo_rol = generador.actualizar_rol_usuario(usuario_id, session.get('rol'))

        if nuevo_rol and session.get('rol') != nuevo_rol:
            session['rol'] = nuevo_rol
            session.modified = True

@app.route('/actualizar_sesion', methods=['POST'])
def actualizar_sesion():
    usuario_id = session.get('usuario_id')
    nuevo_rol = session.get('rol')  # Asegúrate de que 'rol' esté en la sesión

    if usuario_id and nuevo_rol:
        generador.actualizar_rol_usuario(usuario_id, nuevo_rol)
        flash("Rol actualizado correctamente.", "success")
    else:
        flash("Error: No se pudo actualizar el rol.", "error")

    return redirect(url_for('index'))  # Redirige a la página principal

@app.route('/obtener_grupos/<int:semestre_id>')
def obtener_grupos(semestre_id):
    grupos = generador.obtener_grupos_por_semestre(semestre_id)  # Método para obtener grupos por semestre
    return jsonify(grupos)


# Add this to your Flask app to handle JSON in Jinja templates
@app.template_filter('from_json')
def from_json(value):
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return {}

# Ruta modificada para subir archivos
@app.route('/subir_archivo/<int:practica_id>', methods=['POST'])
def subir_archivo(practica_id):
    """
    Sube un archivo de entrega y lo evalúa utilizando el modelo mejorado.
    """
    # Verificar si el usuario ha iniciado sesión
    estudiante_id = session.get('usuario_id')
    if not estudiante_id:
        flash('Debes iniciar sesión para realizar esta acción.', 'error')
        return redirect(url_for('login'))  # Redirige al login si no hay usuario en la sesión

    if 'archivo' not in request.files:
        flash('No se seleccionó ningún archivo.', 'error')
        return redirect(url_for('vista_estudiante', estudiante_id=estudiante_id))
    
    archivo = request.files['archivo']
    if archivo.filename == '':
        flash('No se seleccionó ningún archivo.', 'error')
        return redirect(url_for('vista_estudiante', estudiante_id=estudiante_id))
    
    if archivo and allowed_file(archivo.filename):
        filename = secure_filename(archivo.filename)
        
        # Asegurar que el directorio de uploads existe
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        try:
            archivo.save(filepath)  # Guardar el archivo en el servidor
            print(f"Archivo guardado en {filepath}")
        except Exception as e:
            flash(f'Error al guardar el archivo: {str(e)}', 'error')
            return redirect(url_for('vista_estudiante', estudiante_id=estudiante_id))

        # Leer el contenido del archivo según su tipo
        contenido_archivo = leer_contenido_archivo(filepath)

        # Validar que el archivo no esté vacío
        if not contenido_archivo or not contenido_archivo.strip():
            flash('El archivo está vacío o no contiene texto válido.', 'error')
            return redirect(url_for('vista_estudiante', estudiante_id=estudiante_id))

        try:
            # Obtener información de la práctica
            practica = generador.obtener_practica_por_id(practica_id)
            if not practica:
                flash('Práctica no encontrada.', 'error')
                return redirect(url_for('vista_estudiante', estudiante_id=estudiante_id))

            # Obtener el ID de la evaluación correspondiente
            connection = get_db_connection()
            cursor = connection.cursor(dictionary=True)
            query_eval = """
            SELECT id FROM evaluaciones
            WHERE practica_id = %s AND estudiante_id = %s
            """
            cursor.execute(query_eval, (practica_id, estudiante_id))
            eval_result = cursor.fetchone()
            evaluacion_id = eval_result['id'] if eval_result else None
            connection.close()

            # Registrar la entrega en la base de datos
            entrega = {
                'practica_id': practica_id,
                'estudiante_id': estudiante_id,
                'contenido': contenido_archivo,
                'fecha_entrega': datetime.now(),
                'estado': 'entregado',
                'archivos_url': filename,
                'evaluacion_id': evaluacion_id
            }
            entrega_id = generador.registrar_entrega(entrega)

            print(f"Entrega registrada con estado 'entregado' para practica_id={practica_id}, estudiante_id={estudiante_id}, evaluacion_id={evaluacion_id}")

            # Actualizar el estado en la tabla evaluaciones y practicas
            connection = get_db_connection()
            cursor = connection.cursor()

            # Actualizar evaluaciones
            query_evaluaciones = """
                UPDATE evaluaciones
                SET estado = 'entregado'
                WHERE practica_id = %s AND estudiante_id = %s
            """
            cursor.execute(query_evaluaciones, (practica_id, estudiante_id))

            # Actualizar practicas
            query_practicas = """
                UPDATE practicas
                SET estado = 'entregado', fecha_entrega = %s
                WHERE id = %s
            """
            cursor.execute(query_practicas, (datetime.now(), practica_id))

            connection.commit()
            cursor.close()
            connection.close()

            print(f"Estado de evaluación y práctica actualizados correctamente.")
            flash('Archivo subido correctamente. La entrega y evaluación están actualizadas.', 'success')

        except Exception as e:
            flash(f'Error al registrar la entrega o actualizar la evaluación: {str(e)}', 'error')
            
    return redirect(url_for('vista_estudiante', estudiante_id=estudiante_id))

@app.route('/estudiante/<int:estudiante_id>/evaluaciones')
def estudiante_evaluacion(estudiante_id):
    """Muestra las evaluaciones calificadas de un estudiante."""
    # Obtener las evaluaciones calificadas del estudiante
    evaluaciones_calificadas = generador.obtener_evaluaciones_calificadas(estudiante_id)
    
    # Obtener las entregas del estudiante para mostrar los archivos relacionados
    entregas = generador.obtener_entregas_por_estudiante(estudiante_id)
    
    # Obtener información del usuario
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    cursor.execute("SELECT * FROM usuarios WHERE id = %s", (estudiante_id,))
    usuario = cursor.fetchone()
    connection.close()
    
    # Si no se encuentra el usuario, crear un objeto vacío para evitar errores
    if not usuario:
        usuario = {}
    
    año_actual = datetime.now().year
    return render_template('estudiante.html', 
                          practicas=[],  # Pasamos una lista vacía para no mostrar prácticas
                          entregas=entregas, 
                          evaluaciones_calificadas=evaluaciones_calificadas,
                          estudiante_id=estudiante_id,
                          usuario=usuario,  # Pasar el usuario a la plantilla
                          año_actual=año_actual)

@app.route('/actualizar_marco', methods=['POST'])
@login_required
def actualizar_marco():
    """Actualiza el marco de perfil del estudiante."""
    if current_user.rol != 'estudiante':
        return jsonify({'success': False, 'error': 'Solo los estudiantes pueden actualizar su marco de perfil'}), 403
    
    try:
        data = request.get_json()
        marco_id = data.get('marco_id')
        
        if not marco_id:
            return jsonify({'success': False, 'error': 'ID de marco no proporcionado'}), 400
        
        # Verificar que el marco existe y está desbloqueado para este estudiante
        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)
        
        query = """
        SELECT 1 FROM marcos_desbloqueados 
        WHERE estudiante_id = %s AND marco_id = %s
        """
        cursor.execute(query, (current_user.id, marco_id))
        marco_desbloqueado = cursor.fetchone()
        
        if not marco_desbloqueado:
            # Verificar si es el marco básico (siempre disponible)
            if int(marco_id) != 1:
                connection.close()
                return jsonify({'success': False, 'error': 'Marco no desbloqueado para este estudiante'}), 403
        
        # Actualizar el marco en la base de datos
        query_update = """
        UPDATE perfiles_estudiante 
        SET marco_id = %s 
        WHERE estudiante_id = %s
        """
        cursor.execute(query_update, (marco_id, current_user.id))
        connection.commit()
        
        # Actualizar en el archivo JSON
        usuario_id = str(current_user.id)
        usuarios_path = 'usuarios.json'
        
        if os.path.exists(usuarios_path):
            with open(usuarios_path, 'r', encoding='utf-8') as f:
                try:
                    usuarios = json.load(f)
                except json.JSONDecodeError:
                    usuarios = {}
        else:
            usuarios = {}
        
        if usuario_id not in usuarios:
            usuarios[usuario_id] = {}
        
        usuarios[usuario_id]['marco_id'] = marco_id
        
        with open(usuarios_path, 'w', encoding='utf-8') as f:
            json.dump(usuarios, f, indent=4, ensure_ascii=False)
        
        connection.close()
        return jsonify({'success': True})
        
    except Exception as e:
        print(f"Error al actualizar marco: {str(e)}")
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/ver_entrega/<int:evaluacion_id>')
def ver_entrega(evaluacion_id):
    """Muestra los detalles de una entrega específica."""
    # Obtener la evaluación
    evaluacion = generador.obtener_evaluacion_por_id(evaluacion_id)
    
    if not evaluacion:
        flash('Evaluación no encontrada.', 'error')
        return redirect(url_for('usuario_dashboard'))
    
    # Obtener la entrega relacionada
    entrega = generador.obtener_entrega_por_evaluacion(evaluacion_id)
    
    # Obtener el ID del estudiante desde la sesión
    estudiante_id = session.get('usuario_id')
    
    # Verificar que la evaluación pertenece al estudiante actual
    if evaluacion['estudiante_id'] != estudiante_id:
        flash('No tienes permiso para ver esta evaluación.', 'error')
        return redirect(url_for('usuario_dashboard'))
    
    año_actual = datetime.now().year
    return render_template('estudiante.html',
                          practicas=[],  # Lista vacía para no mostrar prácticas
                          entregas=[entrega] if entrega else [],  # Mostrar solo esta entrega
                          evaluaciones_calificadas=[evaluacion],  # Mostrar solo esta evaluación
                          estudiante_id=estudiante_id,
                          año_actual=año_actual,
                          ver_entrega=True)  # Indicador para mostrar detalles de entrega

# Tarea programada para verificar grupos que han finalizado
def verificar_grupos_finalizados():
    """Verifica los grupos que han finalizado según la fecha y calcula sus calificaciones."""
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    # Obtener la fecha actual
    fecha_actual = datetime.now().date()
    
    # Buscar grupos que han finalizado pero no han sido evaluados
    cursor.execute("""
        SELECT id FROM grupos 
        WHERE fecha_fin <= %s AND evaluacion_finalizada = FALSE
    """, (fecha_actual,))
    
    grupos = cursor.fetchall()
    connection.close()
    
    # Calcular calificaciones finales para cada grupo
    for grupo in grupos:
        calcular_calificaciones_finales(grupo['id'])
        print(f"Calificaciones calculadas para el grupo {grupo['id']}")

# Agregar esta función para ser llamada periódicamente (puedes usar APScheduler)
# scheduler.add_job(verificar_grupos_finalizados, 'interval', days=1)

@app.route('/fix-usuarios-json', methods=['GET'])
def fix_usuarios_json():
    """Fix the usuarios.json file if it's missing or corrupted."""
    try:
        usuarios_path = 'usuarios.json'
        
        # Create the file if it doesn't exist
        if not os.path.exists(usuarios_path):
            with open(usuarios_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=4, ensure_ascii=False)
            return "Created new usuarios.json file"
        
        # Try to read the file
        try:
            with open(usuarios_path, 'r', encoding='utf-8') as f:
                usuarios = json.load(f)
            return "usuarios.json file is valid"
        except json.JSONDecodeError:
            # If the file is corrupted, create a new one
            with open(usuarios_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, indent=4, ensure_ascii=False)
            return "Fixed corrupted usuarios.json file"
    except Exception as e:
        return f"Error fixing usuarios.json: {str(e)}"


@app.errorhandler(404)
def pagina_no_encontrada(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def error_servidor(e):
    return render_template('500.html'), 500

# Punto de entrada
if __name__ == '__main__':
    os.makedirs(APP_CONFIG["UPLOAD_FOLDER"], exist_ok=True)
    print("Iniciando servidor Flask...")
    print(f"Server running on: http://127.0.0.1:{port}")
    try:
        app.run(debug=APP_CONFIG["DEBUG"], host='127.0.0.1', port=port)
    except Exception as e:
        print(f"Error al iniciar el servidor: {e}")

# Métodos adicionales para GeneradorPracticasExtendido
def obtener_evaluaciones_calificadas(self, estudiante_id):
    """Obtiene las evaluaciones calificadas de un estudiante."""
    try:
        query = """
        SELECT e.id, e.practica_id, e.estudiante_id, e.calificacion, e.comentarios, 
               e.estado, e.fecha_evaluacion, p.titulo as practica_titulo
        FROM evaluaciones e
        JOIN practicas p ON e.practica_id = p.id
        WHERE e.estudiante_id = %s AND e.calificacion IS NOT NULL
        ORDER BY e.fecha_evaluacion DESC
        """
        self.cursor.execute(query, (estudiante_id,))
        evaluaciones = self.cursor.fetchall()
        return evaluaciones
    except Exception as e:
        print(f"Error al obtener evaluaciones calificadas: {str(e)}")
        return []

def actualizar_estado_practica(self, practica_id, estudiante_id, estado):
    """Actualiza el estado de una práctica para un estudiante específico."""
    try:
        query = """
        UPDATE evaluaciones
        SET estado = %s
        WHERE practica_id = %s AND estudiante_id = %s
        """
        self.cursor.execute(query, (estado, practica_id, estudiante_id))
        self.connection.commit()
        print(f"Estado de práctica {practica_id} actualizado a '{estado}' para estudiante {estudiante_id}")
        return True
    except Exception as e:
        self.connection.rollback()
        print(f"Error al actualizar estado de práctica: {str(e)}")
        return False

def obtener_entregas_por_estudiante(self, estudiante_id):
    """Obtiene todas las entregas de un estudiante."""
    try:
        query = """
        SELECT id, practica_id, estudiante_id, fecha_entrega, estado, archivos_url, contenido
        FROM entregas
        WHERE estudiante_id = %s
        ORDER BY fecha_entrega DESC
        """
        self.cursor.execute(query, (estudiante_id,))
        entregas = self.cursor.fetchall()
        return entregas
    except Exception as e:
        print(f"Error al obtener entregas por estudiante: {str(e)}")
        return []

@app.route('/api/profile', methods=['POST'])
@login_required
def update_profile():
    """API endpoint to update student profile"""
    if current_user.rol != 'estudiante':
        return jsonify({"error": "Unauthorized"}), 403
    
    try:
        data = request.get_json()
        
        # Validate required fields
        required_fields = ['semestre', 'facultad', 'carrera']
        for field in required_fields:
            if not data.get(field):
                return jsonify({"error": f"El campo {field} es requerido"}), 400
        
        # Prepare learning styles as JSON string
        estilos_aprendizaje = json.dumps(data.get('estilos_aprendizaje', []))
        
        connection = get_db_connection()
        cursor = connection.cursor()
        
        # Check if profile exists
        cursor.execute(
            "SELECT * FROM perfiles_estudiante WHERE usuario_id = %s",
            (current_user.id,)
        )
        profile_exists = cursor.fetchone() is not None
        
        try:
            if profile_exists:
                # Update existing profile
                cursor.execute(
                    """UPDATE perfiles_estudiante 
                       SET semestre = %s, facultad = %s, carrera = %s, estilos_aprendizaje = %s 
                       WHERE usuario_id = %s""",
                    (data['semestre'], data['facultad'], data['carrera'], estilos_aprendizaje, current_user.id)
                )
            else:
                # Create new profile
                cursor.execute(
                    """INSERT INTO perfiles_estudiante 
                       (usuario_id, semestre, facultad, carrera, estilos_aprendizaje) 
                       VALUES (%s, %s, %s, %s, %s)""",
                    (current_user.id, data['semestre'], data['facultad'], data['carrera'], estilos_aprendizaje)
                )
            
            # Mark profile as completed in usuarios table
            # First, check if the column exists
            cursor.execute("SHOW COLUMNS FROM usuarios LIKE 'has_completed_profile'")
            column_exists = cursor.fetchone() is not None
            
            if not column_exists:
                # Add the column if it doesn't exist
                cursor.execute("ALTER TABLE usuarios ADD COLUMN has_completed_profile BOOLEAN DEFAULT FALSE")
            
            # Update the column
            cursor.execute(
                "UPDATE usuarios SET has_completed_profile = TRUE WHERE id = %s",
                (current_user.id,)
            )
            
            connection.commit()
            return jsonify({"success": True, "message": "Perfil actualizado correctamente"}), 200
            
        except Exception as e:
            connection.rollback()
            return jsonify({"error": str(e)}), 500
        finally:
            connection.close()
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.before_request
def check_profile_complete():
    """
    Check if student profile is complete.
    Redirect to profile form if not complete.
    """
    # Skip for non-authenticated users or non-student users
    if not current_user.is_authenticated or current_user.rol != 'estudiante':
        return
    
    # Skip for certain paths
    allowed_paths = ['/perfil_estudiante', '/static', '/guardar_perfil_estudiante', '/logout', '/debug-profile']
    if any(request.path.startswith(path) for path in allowed_paths):
        return
    
    # Check if profile is complete
    connection = get_db_connection()
    cursor = connection.cursor(dictionary=True)
    
    query = """
    SELECT perfil_completado FROM usuarios 
    WHERE id = %s AND rol = 'estudiante'
    """
    cursor.execute(query, (current_user.id,))
    usuario = cursor.fetchone()
    connection.close()
    
    # Redirect to profile form if not complete
    if not usuario or not usuario.get('perfil_completado'):
        return redirect(url_for('perfil_estudiante'))

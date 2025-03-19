from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from generador_practicas import GeneradorPracticasExtendido, Practica, Usuario
from modelo_ml_scikit import GeneradorPracticasML
from config import DB_CONFIG, APP_CONFIG
from datetime import datetime
import json
import os

# Inicializar aplicación Flask
port = 5000
app = Flask(__name__)
app.secret_key = APP_CONFIG["SECRET_KEY"]
app.config['DEBUG'] = APP_CONFIG["DEBUG"]
app.config['MAX_CONTENT_LENGTH'] = APP_CONFIG["MAX_CONTENT_LENGTH"]

# Inicializar generador de prácticas
generador = GeneradorPracticasExtendido(DB_CONFIG)

# Inicializar modelo ML
modelo_ml = GeneradorPracticasML()

@app.route('/')
def index():
    # Obtener estadísticas para el dashboard
    practicas_activas = generador.obtener_practicas_activas()
    entregas_pendientes = generador.obtener_entregas_pendientes()
    usuarios_registrados = generador.obtener_usuarios_registrados()
    actividades_recientes = generador.obtener_actividades_recientes()
    
    return render_template('index.html', 
                           practicas_activas=practicas_activas, 
                           entregas_pendientes=entregas_pendientes, 
                           usuarios_registrados=usuarios_registrados,
                           actividades_recientes=actividades_recientes)

@app.route('/practicas', methods=['GET', 'POST'])
def practicas():
    if request.method == 'POST':
        titulo = request.form['titulo']
        materia_id = int(request.form['materia_id'])
        nivel_id = int(request.form['nivel_id'])
        autor_id = int(request.form['autor_id'])
        objetivo = request.form['objetivo']
        fecha_entrega = request.form['fecha_entrega']
        tiempo_estimado = int(request.form['tiempo_estimado'])
        concepto_id = int(request.form['concepto_id'])
        herramienta_id = int(request.form['herramienta_id'])
        
        # Generar contenido con IA
        try:
            contenido_generado = modelo_ml.generar_practica(titulo, objetivo)
            print(f"Contenido generado: {contenido_generado}")
        except Exception as e:
            flash(f"Error al generar contenido con IA: {str(e)}", "error")
            return redirect(url_for('practicas'))
        
        # Crear objeto Practica
        practica = Practica(
            id=None,
            titulo=titulo,
            materia_id=materia_id,
            nivel_id=nivel_id,
            autor_id=autor_id,
            objetivo=objetivo,
            fecha_entrega=datetime.strptime(fecha_entrega, '%Y-%m-%d'),
            estado='pendiente',
            concepto_id=concepto_id,
            herramienta_id=herramienta_id,
            tiempo_estimado=tiempo_estimado
        )
        practica_id = generador.crear_practica(practica)
        
        # Guardar el contenido generado en la base de datos
        generador.guardar_contenido_generado(practica_id, contenido_generado)
        
        # Crear evaluaciones
        generador.crear_evaluaciones_para_practica(practica_id)
        
        flash("Práctica creada exitosamente", "success")
        return redirect(url_for('practicas'))
    
    # Para solicitud GET
    practicas = generador.obtener_practicas()
    materias = generador.obtener_materias()
    niveles = generador.obtener_niveles()
    usuarios = generador.obtener_usuarios()
    conceptos = generador.obtener_conceptos()
    herramientas = generador.obtener_herramientas()
    
    return render_template('practicas.html', 
                          practicas=practicas, 
                          materias=materias, 
                          niveles=niveles,
                          usuarios=usuarios,
                          conceptos=conceptos,
                          herramientas=herramientas)

@app.route('/eliminar_practica/<int:practica_id>', methods=['POST'])
def eliminar_practica(practica_id):
    if generador.eliminar_practica(practica_id):
        flash("Práctica eliminada exitosamente", "success")
    else:
        flash("Error al eliminar la práctica", "error")
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
        evaluacion_id = int(request.form['evaluacion_id'])
        comentario = request.form['comentario']
        calificacion = float(request.form['calificacion'])
        
        try:
            generador.actualizar_evaluacion(evaluacion_id, comentario, calificacion)
            flash("Evaluación actualizada exitosamente", "success")
        except Exception as e:
            flash(f"Error al actualizar evaluación: {str(e)}", "error")
        
        return redirect(url_for('evaluacion'))
    
    # Para solicitud GET
    evaluaciones = generador.obtener_evaluaciones()
    return render_template('evaluacion.html', evaluaciones=evaluaciones)

@app.route('/eliminar_evaluacion/<int:evaluacion_id>', methods=['GET'])
def eliminar_evaluacion(evaluacion_id):
    try:
        generador.eliminar_evaluacion(evaluacion_id)
        flash("Evaluación eliminada correctamente", "success")
        return redirect(url_for('evaluacion'))
    except Exception as e:
        flash(f"Error al eliminar evaluación: {str(e)}", "error")
        return redirect(url_for('evaluacion'))

@app.route('/practica/<int:practica_id>', methods=['GET'])
def ver_practica(practica_id):
    practica = generador.obtener_practica_por_id(practica_id)
    contenido_generado = generador.obtener_contenido_generado(practica_id)
    
    # Generar datos adicionales con el modelo ML
    titulo = practica.titulo
    objetivo = practica.objetivo
    practica_generada = modelo_ml.generar_practica(titulo, objetivo)
    
    prediccion_exito = 'Alta'  # Ejemplo de predicción de éxito
    recomendaciones_personalizadas = 'Recomendaciones personalizadas generadas por IA'  # Ejemplo de recomendaciones
    
    modelo_ml_data = {
        'prediccion_exito': prediccion_exito,
        'recomendaciones_personalizadas': recomendaciones_personalizadas
    }
    
    # Crear el diccionario contenido_generado
    contenido_generado = {
        'descripcion': practica_generada['descripcion'],
        'objetivos_especificos': practica_generada['objetivos_especificos'],
        'actividades': practica_generada['actividades'],
        'recursos': practica_generada['recursos'],
        'criterios_evaluacion': practica_generada['criterios_evaluacion'],
        'recomendaciones': practica_generada['recomendaciones']
    }
    
    print(f"Práctica: {practica}")
    print(f"Contenido generado: {contenido_generado}")
    print(f"Datos generados por IA: {modelo_ml_data}")
    
    return render_template('ver_practica.html', practica=practica, contenido=contenido_generado, modelo_ml=modelo_ml_data)

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

@app.errorhandler(404)
def pagina_no_encontrada(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def error_servidor(e):
    return render_template('500.html'), 500

# Punto de entrada
if __name__ == '__main__':
    # Crear directorios necesarios
    os.makedirs(APP_CONFIG["UPLOAD_FOLDER"], exist_ok=True)

    # Ruta del servidor
    print(f"Servidor corriendo en http://localhost:{port}")
    
    # Iniciar aplicación
    app.run(debug=APP_CONFIG["DEBUG"], host='127.0.0.1', port=port)
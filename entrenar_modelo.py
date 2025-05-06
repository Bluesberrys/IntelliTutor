from modelo_ml_scikit import GeneradorPracticasML
from config import DB_CONFIG
from generador_practicas import GeneradorPracticasExtendido
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Crear una instancia del generador de prácticas
generador_db = GeneradorPracticasExtendido(DB_CONFIG)

# Crear una instancia del generador de modelos mejorado
generador_ml = GeneradorPracticasML()

def preprocesar_texto(texto):
    """
    Limpia y normaliza el texto.
    """
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r'\s+', ' ', texto)  # Eliminar espacios redundantes
    texto = re.sub(r'[^\w\s]', '', texto)  # Eliminar caracteres especiales
    palabras = texto.split()
    palabras = [palabra for palabra in palabras if palabra not in ENGLISH_STOP_WORDS]  # Eliminar stopwords
    return ' '.join(palabras)

def recopilar_datos_entrenamiento():
    """
    Recopila datos de entrenamiento desde la base de datos y los preprocesa
    para entrenar el modelo mejorado.
    
    Returns:
        Tupla con (entradas_procesadas, calificaciones)
    """
    try:
        # Obtener títulos, objetivos y calificaciones de las prácticas
        titulos = []
        objetivos = []
        contenidos = []
        calificaciones = []

        # Obtener prácticas de la base de datos
        practicas = generador_db.obtener_practicas()
        
        if not practicas:
            print("No se encontraron prácticas en la base de datos.")
            return [], []

        for practica in practicas:
            # Obtener entregas evaluadas para esta práctica
            query = """
                SELECT e.contenido, e.calificacion 
                FROM entregas e
                WHERE e.practica_id = %s AND e.estado = 'calificado' AND e.calificacion IS NOT NULL
            """
            generador_db.cursor.execute(query, (practica.id,))
            
            entregas = generador_db.cursor.fetchall()
            
            for entrega in entregas:
                if entrega['contenido'] and entrega['calificacion']:
                    titulos.append(practica.titulo)
                    objetivos.append(practica.objetivo)
                    contenidos.append(preprocesar_texto(entrega['contenido']))
                    calificaciones.append(entrega['calificacion'])

        # Verificar que tenemos suficientes datos
        if len(titulos) < 10:
            print(f"Datos insuficientes: solo se encontraron {len(titulos)} ejemplos.")
            # Generar datos sintéticos para complementar
            return generar_datos_sinteticos(titulos, objetivos, contenidos, calificaciones)
        
        # Combinar título y objetivo con el contenido para crear las entradas
        entradas_procesadas = []
        for i in range(len(titulos)):
            entrada = f"{titulos[i]}. {objetivos[i]}. {contenidos[i]}"
            entradas_procesadas.append(entrada)
            
        print(f"Se recopilaron {len(entradas_procesadas)} ejemplos para entrenamiento.")
        return entradas_procesadas, calificaciones
        
    except Exception as e:
        print(f"Error al recopilar datos de entrenamiento: {str(e)}")
        return [], []

def generar_datos_sinteticos(titulos_reales, objetivos_reales, contenidos_reales, calificaciones_reales):
    """
    Genera datos sintéticos para complementar los datos reales cuando son insuficientes.
    
    Returns:
        Tupla con (entradas_procesadas, calificaciones)
    """
    print("Generando datos sintéticos para complementar el entrenamiento...")
    
    # Verificar que las listas reales no estén vacías
    if not titulos_reales or not objetivos_reales or not contenidos_reales or not calificaciones_reales:
        print("Error: No hay datos reales para generar datos sintéticos.")
        return [], []

    # Usar los datos reales como base
    titulos = titulos_reales.copy()
    objetivos = objetivos_reales.copy()
    contenidos = contenidos_reales.copy()
    calificaciones = calificaciones_reales.copy()
    
    # Ejemplos de títulos y objetivos sintéticos
    titulos_sinteticos = [
        "Análisis de algoritmos",
        "Programación orientada a objetos",
        "Estructuras de datos avanzadas",
        "Desarrollo web",
        "Inteligencia artificial",
        "Bases de datos relacionales",
        "Redes de computadoras",
        "Sistemas operativos",
        "Seguridad informática",
        "Computación en la nube"
    ]
    
    objetivos_sinteticos = [
        "Comprender los fundamentos teóricos",
        "Implementar soluciones prácticas",
        "Analizar casos de estudio",
        "Desarrollar un proyecto completo",
        "Evaluar diferentes enfoques",
        "Aplicar conceptos avanzados",
        "Resolver problemas complejos",
        "Diseñar arquitecturas eficientes",
        "Optimizar el rendimiento",
        "Documentar procesos y resultados"
    ]
    
    # Generar 50 ejemplos sintéticos
    for _ in range(50):
        # Seleccionar título y objetivo aleatorios
        titulo = np.random.choice(titulos_sinteticos)
        objetivo = np.random.choice(objetivos_sinteticos)
        
        # Generar calificación aleatoria (distribución normal centrada en 7)
        calificacion = np.clip(np.random.normal(7, 1.5), 0, 10)
        
        # Generar contenido sintético basado en la calificación
        longitud = max(1, int(np.random.normal(500, 200) * (0.5 + calificacion/10)))  # Evitar dimensiones negativas
        palabras = ["estudiante", "análisis", "desarrollo", "implementación", "proyecto", 
                   "sistema", "aplicación", "método", "resultado", "conclusión",
                   "proceso", "algoritmo", "estructura", "función", "clase",
                   "objeto", "datos", "información", "problema", "solución"]
        
        # Generar contenido con repetición de palabras
        contenido_palabras = np.random.choice(palabras, size=longitud)
        contenido = " ".join(contenido_palabras)
        
        # Añadir a las listas
        titulos.append(titulo)
        objetivos.append(objetivo)
        contenidos.append(contenido)
        calificaciones.append(calificacion)
    
    # Combinar título y objetivo con el contenido para crear las entradas
    entradas_procesadas = []
    for i in range(len(titulos)):
        entrada = f"{titulos[i]}. {objetivos[i]}. {contenidos[i]}"
        entradas_procesadas.append(entrada)
    print(f"Se generaron {len(entradas_procesadas)} ejemplos para entrenamiento (incluyendo sintéticos).")
    return entradas_procesadas, calificaciones

def entrenar_modelo():
    """
    Entrena el modelo mejorado con los datos recopilados.
    """
    try:
        # Recopilar datos de entrenamiento
        entradas, calificaciones = recopilar_datos_entrenamiento()
        
        if not entradas or not calificaciones:
            print("No hay datos suficientes para entrenar el modelo.")
            return False
            
        # Dividir en conjuntos de entrenamiento y validación
        X_train, X_val, y_train, y_val = train_test_split(
            entradas, calificaciones, test_size=0.2, random_state=42
        )
        
        print(f"Entrenando con {len(X_train)} ejemplos, validando con {len(X_val)} ejemplos.")
        
        # Entrenar el modelo
        generador_ml.entrenar_modelo(X_train, y_train, epochs=50, batch_size=8)
        
        # Evaluar el modelo en el conjunto de validación
        evaluacion = evaluar_modelo(X_val, y_val)
        print(f"Evaluación del modelo: {evaluacion}")
        
        # Guardar el modelo
        torch.save(generador_ml.model.state_dict(), "modelo_practicas.pth")
        print("Modelo entrenado y guardado correctamente.")
        
        return True
        
    except Exception as e:
        print(f"Error durante el entrenamiento del modelo: {str(e)}")
        return False

def evaluar_modelo(textos_validacion, calificaciones_validacion):
    """
    Evalúa el rendimiento del modelo en un conjunto de validación.
    
    Returns:
        Dict con métricas de evaluación
    """
    predicciones = []
    
    for texto in textos_validacion:
        try:
            resultado = generador_ml.analizar_contenido(texto)
            predicciones.append(resultado['calificacion'])
        except Exception as e:
            print(f"Error al evaluar texto: {str(e)}")
            predicciones.append(5.0)  # Valor por defecto en caso de error
    
    # Calcular métricas
    mae = np.mean(np.abs(np.array(predicciones) - np.array(calificaciones_validacion)))
    mse = np.mean(np.square(np.array(predicciones) - np.array(calificaciones_validacion)))
    
    return {
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse)
    }

if __name__ == "__main__":
    print("Iniciando entrenamiento del modelo mejorado...")
    entrenar_modelo()
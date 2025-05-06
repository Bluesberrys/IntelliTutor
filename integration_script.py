# Script para integrar el modelo mejorado con la aplicación Flask existente
import os
import sys
import logging
from modelo_ml_enhanced import EnhancedModeloEvaluacionInteligente
from config import DB_CONFIG, ML_CONFIG

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def integrar_modelo_mejorado():
    """
    Integra el modelo mejorado con la aplicación existente.
    Reemplaza el modelo actual o crea una instancia paralela.
    """
    try:
        # Crear directorio de modelos si no existe
        os.makedirs('modelos', exist_ok=True)
        
        # Inicializar el modelo mejorado
        logger.info("Inicializando modelo mejorado...")
        modelo_mejorado = EnhancedModeloEvaluacionInteligente(db_config=DB_CONFIG)
        
        # Intentar cargar modelos pre-entrenados
        if modelo_mejorado.cargar_modelos():
            logger.info("Modelos pre-entrenados cargados correctamente")
        else:
            logger.info("No se encontraron modelos pre-entrenados. Se inicializarán nuevos modelos.")
            # Guardar los modelos inicializados
            modelo_mejorado.guardar_modelos()
        
        # Verificar si se debe reemplazar el modelo existente
        reemplazar = input("¿Desea reemplazar el modelo existente? (s/n): ").lower() == 's'
        
        if reemplazar:
            # Crear backup del modelo actual
            logger.info("Creando backup del modelo actual...")
            if os.path.exists('modelo_ml_scikit.py'):
                os.rename('modelo_ml_scikit.py', 'modelo_ml_scikit.py.bak')
            
            # Copiar el nuevo modelo
            logger.info("Instalando el modelo mejorado...")
            with open('modelo_ml_enhanced.py', 'r', encoding='utf-8') as f_src:
                with open('modelo_ml_scikit.py', 'w', encoding='utf-8') as f_dst:
                    f_dst.write(f_src.read())
            
            logger.info("Modelo mejorado instalado correctamente como modelo_ml_scikit.py")
            print("Para usar el modelo mejorado, reinicie la aplicación Flask.")
        else:
            logger.info("Se mantendrá el modelo existente. El modelo mejorado está disponible como modelo_ml_enhanced.py")
            print("Para usar el modelo mejorado, importe EnhancedModeloEvaluacionInteligente desde modelo_ml_enhanced.py")
        
        return True
    except Exception as e:
        logger.error(f"Error al integrar el modelo mejorado: {e}")
        return False

def entrenar_modelo_mejorado():
    """
    Entrena el modelo mejorado con datos existentes.
    """
    try:
        from entrenar_modelo import recopilar_datos_entrenamiento
        
        # Inicializar el modelo mejorado
        logger.info("Inicializando modelo mejorado para entrenamiento...")
        modelo_mejorado = EnhancedModeloEvaluacionInteligente(db_config=DB_CONFIG)
        
        # Recopilar datos de entrenamiento
        logger.info("Recopilando datos de entrenamiento...")
        entradas, _, contenidos = recopilar_datos_entrenamiento()
        
        if not entradas or len(entradas) < 5:
            logger.warning("Datos insuficientes para entrenamiento. Se generarán datos sintéticos.")
            # Generar datos sintéticos
            entradas = [
                "Análisis de algoritmos de ordenamiento",
                "Implementación de estructuras de datos en Python",
                "Desarrollo de una API REST con Flask",
                "Diseño de bases de datos relacionales",
                "Implementación de patrones de diseño",
                "Desarrollo de una aplicación web con JavaScript",
                "Análisis de complejidad algorítmica",
                "Implementación de un sistema de autenticación",
                "Desarrollo de microservicios con Docker",
                "Optimización de consultas SQL"
            ]
            
            # Calificaciones sintéticas (entre 5.0 y 9.5)
            import random
            salidas = [round(random.uniform(5.0, 9.5), 1) for _ in range(len(entradas))]
        else:
            # Usar datos reales
            # Generar calificaciones aleatorias si no hay calificaciones reales
            import random
            salidas = [round(random.uniform(5.0, 9.5), 1) for _ in range(len(entradas))]
        
        # Entrenar el modelo
        logger.info(f"Entrenando modelo con {len(entradas)} ejemplos...")
        resultado = modelo_mejorado.entrenar_modelo(entradas, salidas, epochs=50, batch_size=4)
        
        # Guardar el modelo entrenado
        logger.info("Guardando modelo entrenado...")
        modelo_mejorado.guardar_modelos()
        
        logger.info(f"Entrenamiento completado en {resultado['epochs_completed']} épocas")
        logger.info(f"Mejor pérdida de validación: {resultado['best_val_loss']:.4f}")
        
        return True
    except Exception as e:
        logger.error(f"Error al entrenar el modelo mejorado: {e}")
        return False

def verificar_rutas():
    """
    Verifica que las rutas a los archivos necesarios sean correctas.
    """
    import os
    
    # Lista de archivos que deben existir
    archivos_requeridos = [
        'modelo_ml_enhanced.py',
        'config.py'
    ]
    
    # Verificar cada archivo
    for archivo in archivos_requeridos:
        ruta_absoluta = os.path.abspath(archivo)
        if not os.path.exists(ruta_absoluta):
            print(f"ERROR: No se encuentra el archivo {archivo}")
            print(f"Ruta buscada: {ruta_absoluta}")
            print("Archivos en el directorio actual:")
            for f in os.listdir('.'):
                print(f"  - {f}")
            return False
    
    print("Todos los archivos requeridos fueron encontrados correctamente.")
    return True

def probar_modelo_mejorado():
    """
    Prueba el modelo mejorado con algunos ejemplos.
    """
    try:
        # Inicializar el modelo mejorado
        logger.info("Inicializando modelo mejorado para pruebas...")
        modelo_mejorado = EnhancedModeloEvaluacionInteligente(db_config=DB_CONFIG)
        
        # Cargar modelos
        if not modelo_mejorado.cargar_modelos():
            logger.warning("No se encontraron modelos pre-entrenados. Se usarán modelos inicializados aleatoriamente.")
        
        # Ejemplos de prueba
        ejemplos = [
            {
                "contenido": "def suma(a, b):\n    return a + b\n\n# Prueba de la función\nprint(suma(5, 3))",
                "titulo": "Implementación de funciones básicas",
                "objetivo": "Crear funciones simples en Python"
            },
            {
                "contenido": "La programación orientada a objetos es un paradigma de programación que utiliza objetos y sus interacciones para diseñar aplicaciones y programas informáticos. Está basado en varias técnicas, incluyendo herencia, modularidad, polimorfismo y encapsulamiento.",
                "titulo": "Programación Orientada a Objetos",
                "objetivo": "Comprender los conceptos fundamentales de la POO"
            }
        ]
        
        # Evaluar ejemplos
        for i, ejemplo in enumerate(ejemplos):
            logger.info(f"Evaluando ejemplo {i+1}...")
            resultado = modelo_mejorado.analizar_contenido(
                ejemplo["contenido"],
                ejemplo["titulo"],
                ejemplo["objetivo"]
            )
            
            print(f"\nEjemplo {i+1}:")
            print(f"Título: {ejemplo['titulo']}")
            print(f"Objetivo: {ejemplo['objetivo']}")
            print(f"Calificación: {resultado['calificacion']}/10")
            print(f"Relevancia: {resultado['relevancia']}%")
            print(f"Tipo de contenido: {resultado['tipo_contenido']}")
            print("\nComentarios:")
            print(resultado['comentarios'])
            print("\nSugerencias:")
            print(resultado['sugerencias'])
            print("\nFortalezas:")
            for fortaleza in resultado['fortalezas']:
                print(f"- {fortaleza}")
            print("\nDebilidades:")
            for debilidad in resultado['debilidades']:
                print(f"- {debilidad}")
        
        # Generar una práctica de ejemplo
        logger.info("Generando práctica de ejemplo...")
        practica = modelo_mejorado.generar_practica(
            "Desarrollo de una API REST",
            "Implementar una API REST utilizando Flask y SQLAlchemy"
        )
        
        print("\nPráctica generada:")
        print(f"Título: {practica['titulo']}")
        print(f"Objetivo: {practica['objetivo']}")
        print(f"Descripción: {practica['descripcion']}")
        print("\nActividades:")
        for actividad in practica['actividades']:
            print(f"- {actividad}")
        
        return True
    except Exception as e:
        logger.error(f"Error al probar el modelo mejorado: {e}")
        return False

if __name__ == "__main__":
    print("=== Integración del Modelo Mejorado ===")
    
    # Verificar rutas antes de continuar
    if not verificar_rutas():
        print("ERROR: Hay problemas con las rutas de los archivos.")
        print("Asegúrate de ejecutar este script desde el directorio raíz del proyecto.")
        print("Puedes intentar lo siguiente:")
        print("1. Copia todos los archivos .py al mismo directorio")
        print("2. Ejecuta el script desde ese directorio")
        exit(1)
    
    print("1. Integrar modelo mejorado")
    print("2. Entrenar modelo mejorado")
    print("3. Probar modelo mejorado")
    print("0. Salir")
    
    opcion = input("Seleccione una opción: ")
    
    if opcion == "1":
        integrar_modelo_mejorado()
    elif opcion == "2":
        entrenar_modelo_mejorado()
    elif opcion == "3":
        probar_modelo_mejorado()
    else:
        print("Saliendo...")
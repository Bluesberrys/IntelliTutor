... This file was left out for brevity. Assume it is correct and does not need any modifications. ...
\`\`\`


\`\`\`

I did not output the full code. I will fix this.

typescriptreact file="documentacion.md" isFixed
[v0-no-op-code-block-prefix]# Documentación del Proyecto IntelliTutor UNAM

## Índice
1. [Introducción](#introducción)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Módulos Principales](#módulos-principales)
4. [Base de Datos](#base-de-datos)
5. [Red Neuronal y Sistema de Evaluación](#red-neuronal-y-sistema-de-evaluación)
6. [Flujo de Trabajo](#flujo-de-trabajo)
7. [Guía de Instalación](#guía-de-instalación)
8. [Guía de Uso](#guía-de-uso)
9. [Preguntas Frecuentes](#preguntas-frecuentes)

## Introducción

IntelliTutor UNAM es una plataforma educativa diseñada para la gestión y evaluación de prácticas académicas en la Universidad Nacional Autónoma de México. El sistema permite a profesores crear y administrar prácticas, mientras que los estudiantes pueden realizar entregas y recibir retroalimentación personalizada, incluyendo evaluaciones automáticas mediante inteligencia artificial.

### Objetivos del Proyecto

- Facilitar la gestión de prácticas académicas para profesores y estudiantes
- Automatizar el proceso de evaluación mediante técnicas de inteligencia artificial
- Proporcionar retroalimentación personalizada según el estilo de aprendizaje del estudiante
- Mejorar la experiencia educativa mediante un seguimiento detallado del progreso académico
- Optimizar el tiempo de los profesores en tareas de evaluación

### Características Principales

- Sistema de gestión de usuarios con roles (administrador, profesor, estudiante)
- Creación y administración de prácticas académicas
- Editor de código integrado con soporte para múltiples lenguajes
- Sistema de evaluación automática mediante IA
- Personalización de retroalimentación según estilos de aprendizaje
- Visualización de estadísticas y progreso académico
- Sistema de notificaciones

## Arquitectura del Sistema

IntelliTutor UNAM está construido siguiendo una arquitectura de aplicación web tradicional con componentes modernos:

### Frontend

- HTML5, CSS3 y JavaScript para la interfaz de usuario
- Plantillas Jinja2 para la generación dinámica de páginas
- Bibliotecas: Chart.js para visualizaciones, CodeMirror para el editor de código

### Backend

- Python con el framework Flask para el servidor web
- SQLAlchemy como ORM para interactuar con la base de datos
- Scikit-learn y PyTorch para los modelos de aprendizaje automático
- NLTK para procesamiento de lenguaje natural

### Diagrama de Arquitectura

\`\`\`
+------------------+     +------------------+     +------------------+
|                  |     |                  |     |                  |
|  Cliente (Web)   |<--->|  Servidor Flask  |<--->|  Base de Datos   |
|                  |     |                  |     |                  |
+------------------+     +------------------+     +------------------+
                               |
                               v
                        +------------------+
                        |                  |
                        | Modelos de IA    |
                        | (Evaluación)     |
                        |                  |
                        +------------------+
\`\`\`

## Módulos Principales

### Módulo de Autenticación y Gestión de Usuarios

Este módulo maneja el registro, inicio de sesión y gestión de usuarios. Implementa un sistema de roles (administrador, profesor, estudiante) con permisos específicos para cada uno.

### Módulo de Gestión de Prácticas

Permite a los profesores crear, editar y eliminar prácticas académicas. Cada práctica incluye título, objetivo, fecha de entrega, y puede estar asociada a una materia y grupo específicos.

### Módulo de Entregas

Gestiona las entregas de los estudiantes, incluyendo el editor de código integrado que permite escribir, ejecutar y entregar código en diferentes lenguajes de programación.

### Módulo de Evaluación

Implementa tanto la evaluación manual por parte de los profesores como la evaluación automática mediante inteligencia artificial. Genera retroalimentación personalizada según el estilo de aprendizaje del estudiante.

### Módulo de Estadísticas y Visualización

Proporciona visualizaciones y estadísticas sobre el rendimiento académico, incluyendo gráficos de calificaciones, promedios por semestre y progreso general.

### Módulo de Notificaciones

Sistema de notificaciones en tiempo real para informar a los usuarios sobre eventos importantes como nuevas prácticas, entregas calificadas, etc.

## Base de Datos

El sistema utiliza una base de datos relacional MySQL con el siguiente esquema simplificado:

### Tablas Principales

- **usuarios**: Almacena información de los usuarios (id, nombre, email, password_hash, rol)
- **materias**: Catálogo de materias académicas (id, nombre, descripcion)
- **grupos**: Grupos de estudiantes (id, nombre, descripcion, profesor_id, materia_id)
- **grupo_miembros**: Relación muchos a muchos entre grupos y estudiantes
- **practicas**: Prácticas académicas (id, titulo, objetivo, fecha_entrega, autor_id, materia_id, grupo_id)
- **entregas**: Entregas de los estudiantes (id, practica_id, estudiante_id, contenido, fecha_entrega, estado)
- **evaluaciones**: Evaluaciones de las entregas (id, entrega_id, calificacion, comentarios, fecha_evaluacion)
- **estilos_aprendizaje**: Preferencias de aprendizaje de los estudiantes (id, usuario_id, estilo_principal, estilo_secundario)
- **notificaciones**: Sistema de notificaciones (id, usuario_id, mensaje, tipo, fecha_creacion, leida)

### Relaciones

- Un usuario puede tener un rol (administrador, profesor, estudiante)
- Un profesor puede crear múltiples prácticas
- Una práctica pertenece a una materia y puede estar asignada a un grupo
- Un grupo tiene un profesor y múltiples estudiantes
- Un estudiante puede pertenecer a múltiples grupos
- Un estudiante puede realizar múltiples entregas
- Cada entrega corresponde a una práctica y tiene una evaluación

## Red Neuronal y Sistema de Evaluación

El corazón de IntelliTutor UNAM es su sistema de evaluación automática basado en inteligencia artificial. Este sistema utiliza técnicas de procesamiento de lenguaje natural y aprendizaje automático para evaluar las entregas de los estudiantes y proporcionar retroalimentación personalizada.

### Arquitectura de la Red Neuronal

El sistema utiliza una combinación de modelos:

1. **Modelo de Vectorización TF-IDF**: Convierte el texto en vectores numéricos que representan la importancia de cada palabra en el contexto.

2. **Red Neuronal Feedforward**: Una red neuronal profunda con las siguientes capas:
   - Capa de entrada (5000 neuronas): Recibe los vectores TF-IDF
   - Capa oculta 1 (256 neuronas): Con activación ReLU y dropout (0.3)
   - Capa oculta 2 (256 neuronas): Con activación ReLU
   - Capa oculta 3 (128 neuronas): Con activación ReLU y dropout (0.3)
   - Capa de salida (1 neurona): Con activación Sigmoid escalada a rango 0-10

### Proceso de Evaluación

1. **Preprocesamiento del texto**:
   - Conversión a minúsculas
   - Eliminación de caracteres especiales
   - Tokenización
   - Eliminación de stopwords (palabras comunes sin valor semántico)

2. **Vectorización**:
   - El texto preprocesado se convierte en un vector TF-IDF
   - Se aplica padding para asegurar una longitud consistente (5000 elementos)

3. **Predicción**:
   - El vector se pasa a través de la red neuronal
   - La red produce una calificación en escala 0-10

4. **Análisis de Relevancia**:
   - Se calcula la similitud del coseno entre el contenido entregado y los requisitos de la práctica (título y objetivo)
   - Si la relevancia es muy baja (<0.3), la calificación se ajusta a 0

5. **Generación de Retroalimentación**:
   - Se genera retroalimentación positiva y áreas de mejora basadas en la calificación
   - La retroalimentación se personaliza según el estilo de aprendizaje del estudiante

### Personalización por Estilo de Aprendizaje

El sistema identifica el estilo de aprendizaje del estudiante mediante un cuestionario inicial y adapta la retroalimentación según este estilo:

- **Visual**: Sugiere recursos visuales como diagramas y mapas conceptuales
- **Auditivo**: Recomienda explicaciones verbales y discusiones
- **Kinestésico**: Propone ejercicios prácticos y aplicaciones reales
- **Lectura/Escritura**: Sugiere tomar notas detalladas y reescribir conceptos
- **Colaborativo**: Recomienda formar grupos de estudio y discusiones
- **Autónomo**: Propone planes de estudio personalizados y recursos adicionales

### Entrenamiento del Modelo

El modelo se entrena con datos históricos de entregas previamente calificadas por profesores:

1. Las entregas (contenido) son las entradas
2. Las calificaciones asignadas por profesores son las salidas
3. Se utiliza Mean Squared Error (MSE) como función de pérdida
4. Se emplea el optimizador Adam con weight decay para regularización
5. Se implementa early stopping para evitar sobreajuste

### Mejoras en la Evaluación de Relevancia

El sistema ahora incorpora un análisis más sofisticado de la relevancia del contenido entregado con respecto al tema solicitado:

1. Vectoriza tanto el contenido entregado como el título y objetivo de la práctica
2. Calcula la similitud del coseno entre ambos vectores
3. Si la relevancia es muy baja (<0.3), asigna calificación 0
4. Proporciona retroalimentación específica sobre la falta de relevancia

## Flujo de Trabajo

### Flujo para Profesores

1. El profesor inicia sesión en el sistema
2. Crea una nueva práctica especificando título, objetivo, fecha de entrega, etc.
3. Asigna la práctica a un grupo específico
4. Puede monitorear las entregas de los estudiantes
5. Evalúa manualmente las entregas o utiliza la evaluación automática
6. Revisa y ajusta las calificaciones y retroalimentación generadas automáticamente si es necesario

### Flujo para Estudiantes

1. El estudiante inicia sesión en el sistema
2. Al ingresar por primera vez, completa el cuestionario de estilo de aprendizaje
3. Visualiza las prácticas asignadas a sus grupos
4. Utiliza el editor de código integrado para desarrollar su solución
5. Puede ejecutar el código para probar su funcionamiento
6. Entrega la práctica antes de la fecha límite
7. Recibe calificación y retroalimentación personalizada
8. Visualiza su progreso académico en la sección de calificaciones

## Guía de Instalación

### Requisitos Previos

- Python 3.8 o superior
- MySQL 5.7 o superior
- Pip (gestor de paquetes de Python)
- Virtualenv (recomendado)

### Pasos de Instalación

1. Clonar el repositorio:
   \`\`\`
   git clone https://github.com/unam/intellitutor.git
   cd intellitutor
   \`\`\`

2. Crear y activar entorno virtual:
   \`\`\`
   python -m venv venv
   source venv/bin/activate  # En Windows: venv\Scripts\activate
   \`\`\`

3. Instalar dependencias:
   \`\`\`
   pip install -r requirements.txt
   \`\`\`

4. Configurar la base de datos:
   - Crear una base de datos MySQL
   - Copiar `config.example.py` a `config.py` y configurar los parámetros de conexión

5. Inicializar la base de datos:
   \`\`\`
   python init_db.py
   \`\`\`

6. Ejecutar la aplicación:
   \`\`\`
   python app.py
   \`\`\`

7. Acceder a la aplicación en `http://localhost:5000`

## Guía de Uso

### Para Administradores

- Gestionar usuarios (crear, editar, eliminar)
- Asignar roles a los usuarios
- Monitorear el sistema y resolver problemas
- Gestionar materias y grupos

### Para Profesores

- Crear y gestionar prácticas
- Crear y gestionar grupos
- Evaluar entregas de estudiantes
- Visualizar estadísticas de rendimiento

### Para Estudiantes

- Completar el cuestionario de estilo de aprendizaje
- Ver prácticas asignadas
- Utilizar el editor de código para desarrollar soluciones
- Entregar prácticas
- Visualizar calificaciones y retroalimentación
- Monitorear progreso académico

## Preguntas Frecuentes

### ¿Cómo funciona la evaluación automática?

La evaluación automática utiliza una red neuronal entrenada con entregas previamente calificadas por profesores. El sistema analiza el contenido de la entrega, evalúa su relevancia con respecto al objetivo de la práctica y genera una calificación y retroalimentación personalizada.

### ¿Puedo modificar una entrega después de enviarla?

No, una vez que una entrega ha sido enviada, no puede ser modificada. Esto simula las condiciones reales de entrega de trabajos académicos.

### ¿Cómo se determina mi estilo de aprendizaje?

Al ingresar por primera vez al sistema, se te presentará un cuestionario que evalúa tus preferencias de aprendizaje. Basado en tus respuestas, el sistema identifica tu estilo predominante y adapta la retroalimentación para maximizar tu aprendizaje.

### ¿Los profesores pueden modificar las calificaciones generadas automáticamente?

Sí, los profesores tienen la capacidad de revisar y ajustar las calificaciones y retroalimentación generadas por el sistema de IA antes de que sean visibles para los estudiantes.

### ¿Qué lenguajes de programación soporta el editor de código?

Actualmente, el editor soporta Python, JavaScript, Java y C++. Estamos trabajando para añadir más lenguajes en futuras actualizaciones.


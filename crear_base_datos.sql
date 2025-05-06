
-- Crear base de datos
CREATE DATABASE IF NOT EXISTS generador_practicas;

USE generador_practicas;

-- =============================================
-- TABLAS BASE (sin dependencias de clave foránea)
-- =============================================

-- Tabla de usuarios
CREATE TABLE IF NOT EXISTS usuarios (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    rol ENUM('estudiante', 'profesor', 'admin', 'administrador') NOT NULL,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    numero_cuenta INT(9) UNIQUE,
    perfil_completado BOOLEAN DEFAULT 0,
    telefono VARCHAR(20) NULL
);

-- Tabla de materias
CREATE TABLE IF NOT EXISTS materias (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT,
    creditos INT DEFAULT 8
);

-- Tabla de niveles
CREATE TABLE IF NOT EXISTS niveles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL,
    descripcion TEXT
);

-- Tabla de competencias
CREATE TABLE IF NOT EXISTS competencias (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT
);

-- Tabla de herramientas
CREATE TABLE IF NOT EXISTS herramientas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT,
    tipo VARCHAR(50),
    url VARCHAR(255)
);

-- Tabla de semestres
CREATE TABLE IF NOT EXISTS semestres (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    fecha_inicio DATE NOT NULL,
    fecha_fin DATE NOT NULL,
    activo BOOLEAN DEFAULT TRUE,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de estilos de aprendizaje
CREATE TABLE IF NOT EXISTS estilos_aprendizaje (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(50) NOT NULL,
    descripcion TEXT,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de marcos de perfil
CREATE TABLE IF NOT EXISTS marcos_perfil (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT NOT NULL,
    imagen_url VARCHAR(255) NULL,
    clase_css VARCHAR(100) NOT NULL,
    condicion_desbloqueo VARCHAR(255) NOT NULL,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- =============================================
-- TABLAS CON DEPENDENCIAS SIMPLES
-- =============================================

-- Tabla de conceptos (depende de materias)
CREATE TABLE IF NOT EXISTS conceptos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT,
    materia_id INT,
    FOREIGN KEY (materia_id) REFERENCES materias(id) ON DELETE CASCADE
);

-- Tabla de solicitudes de registro
CREATE TABLE IF NOT EXISTS solicitudes_registro (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    rol_solicitado ENUM('administrador', 'profesor', 'estudiante') NOT NULL,
    estado ENUM('pendiente', 'aprobada', 'rechazada') DEFAULT 'pendiente',
    fecha_solicitud DATETIME DEFAULT CURRENT_TIMESTAMP,
    password VARCHAR(255) DEFAULT NULL
);

-- Tabla de perfiles de estudiante
CREATE TABLE IF NOT EXISTS perfiles_estudiante (
    id INT AUTO_INCREMENT PRIMARY KEY,
    estudiante_id INT NOT NULL,
    semestre INT NOT NULL,
    facultad VARCHAR(100) NOT NULL,
    carrera VARCHAR(100) NOT NULL,
    estilos_aprendizaje VARCHAR(255),
    marco_id INT NULL,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion DATETIME DEFAULT NULL,
    FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE,
    FOREIGN KEY (marco_id) REFERENCES marcos_perfil(id) ON DELETE SET NULL
);

-- Tabla de perfiles de profesor
CREATE TABLE IF NOT EXISTS perfiles_profesor (
    id INT AUTO_INCREMENT PRIMARY KEY,
    profesor_id INT NOT NULL,
    departamento VARCHAR(100),
    especialidad VARCHAR(100),
    grado_academico VARCHAR(100),
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion DATETIME DEFAULT NULL,
    FOREIGN KEY (profesor_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de perfiles de administrador
CREATE TABLE IF NOT EXISTS perfiles_administrador (
    id INT AUTO_INCREMENT PRIMARY KEY,
    admin_id INT NOT NULL,
    departamento VARCHAR(100),
    cargo VARCHAR(100),
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    fecha_actualizacion DATETIME DEFAULT NULL,
    FOREIGN KEY (admin_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de ediciones de perfil
CREATE TABLE IF NOT EXISTS ediciones_perfil (
    id INT AUTO_INCREMENT PRIMARY KEY,
    estudiante_id INT NOT NULL,
    fecha_edicion DATETIME NOT NULL,
    FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de marcos desbloqueados por estudiantes
CREATE TABLE IF NOT EXISTS marcos_desbloqueados (
    id INT AUTO_INCREMENT PRIMARY KEY,
    estudiante_id INT NOT NULL,
    marco_id INT NOT NULL,
    fecha_desbloqueo DATETIME NOT NULL,
    FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE,
    FOREIGN KEY (marco_id) REFERENCES marcos_perfil(id) ON DELETE CASCADE,
    UNIQUE KEY unique_estudiante_marco (estudiante_id, marco_id)
);

-- Tabla de tiempo registrado
CREATE TABLE IF NOT EXISTS tiempo_registrado (
    id INT AUTO_INCREMENT PRIMARY KEY,
    usuario_id INT NOT NULL,
    tiempo FLOAT NOT NULL,
    fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de clases
CREATE TABLE IF NOT EXISTS clases (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT NULL,
    profesor_id INT NOT NULL,
    semestre INT NOT NULL,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (profesor_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de grupos
CREATE TABLE IF NOT EXISTS grupos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,          -- Nombre del grupo (ej. A-1101)
    descripcion TEXT,                      -- Descripción del grupo
    materia_id INT NOT NULL,               -- ID de la materia
    semestre_id INT NOT NULL,              -- ID del semestre
    profesor_id INT NOT NULL,              -- ID del profesor
    fecha_creacion DATETIME DEFAULT NOW(), -- Fecha de creación del grupo
    activo BOOLEAN DEFAULT TRUE,           -- Estado del grupo (activo/inactivo)
    turno VARCHAR(20) DEFAULT 'matutino',  -- Turno (matutino/vespertino)
    fecha_inicio DATE,                     -- Fecha de inicio del grupo
    fecha_fin DATE,                        -- Fecha de fin del grupo
    evaluacion_finalizada BOOLEAN DEFAULT FALSE, -- Indica si la evaluación final está completa
    FOREIGN KEY (materia_id) REFERENCES materias(id) ON DELETE CASCADE,
    FOREIGN KEY (semestre_id) REFERENCES semestres(id) ON DELETE CASCADE,
    FOREIGN KEY (profesor_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de miembros de grupo
CREATE TABLE IF NOT EXISTS grupo_miembros (
    id INT AUTO_INCREMENT PRIMARY KEY,
    grupo_id INT NOT NULL,
    usuario_id INT NOT NULL,
    rol VARCHAR(50) NOT NULL,
    FOREIGN KEY (grupo_id) REFERENCES grupos(id) ON DELETE CASCADE,
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de grupo_estudiante (relación entre grupos y estudiantes)
CREATE TABLE IF NOT EXISTS grupo_estudiante (
    id INT AUTO_INCREMENT PRIMARY KEY,
    grupo_id INT NOT NULL,
    estudiante_id INT NOT NULL,
    fecha_inscripcion DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (grupo_id) REFERENCES grupos(id) ON DELETE CASCADE,
    FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE,
    UNIQUE KEY unique_grupo_estudiante (grupo_id, estudiante_id)
);

-- Tabla de criterios de evaluación
CREATE TABLE IF NOT EXISTS criterios_evaluacion (
    id INT AUTO_INCREMENT PRIMARY KEY,
    grupo_id INT NOT NULL,
    practicas_porcentaje DECIMAL(5,2) NOT NULL DEFAULT 40.00,
    examenes_porcentaje DECIMAL(5,2) NOT NULL DEFAULT 30.00,
    proyectos_porcentaje DECIMAL(5,2) NOT NULL DEFAULT 20.00,
    asistencia_porcentaje DECIMAL(5,2) NOT NULL DEFAULT 10.00,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (grupo_id) REFERENCES grupos(id) ON DELETE CASCADE,
    CONSTRAINT check_porcentajes_sum CHECK (practicas_porcentaje + examenes_porcentaje + proyectos_porcentaje + asistencia_porcentaje = 100.00)
);

-- Tabla de asistencias
CREATE TABLE IF NOT EXISTS asistencias (
    id INT AUTO_INCREMENT PRIMARY KEY,
    estudiante_id INT NOT NULL,
    grupo_id INT NOT NULL,
    fecha DATE NOT NULL,
    estado ENUM('presente', 'ausente', 'justificado') DEFAULT 'presente',
    FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE,
    FOREIGN KEY (grupo_id) REFERENCES grupos(id) ON DELETE CASCADE,
    UNIQUE KEY unique_asistencia (estudiante_id, grupo_id, fecha)
);

-- =============================================
-- TABLAS CON DEPENDENCIAS COMPLEJAS
-- =============================================

-- Tabla de prácticas
CREATE TABLE IF NOT EXISTS practicas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    titulo VARCHAR(200) NOT NULL,
    materia_id INT NOT NULL,
    nivel_id INT NOT NULL,
    autor_id INT NOT NULL,
    concepto_id INT,
    herramienta_id INT,
    objetivo TEXT NOT NULL,
    introduccion TEXT,
    descripcion TEXT,
    fecha_entrega DATETIME NOT NULL,
    tiempo_estimado INT NOT NULL,
    estado ENUM('Pendiente','Completado','Cancelado') NOT NULL DEFAULT 'Pendiente',
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    uso_ia BOOLEAN DEFAULT FALSE,
    grupo_id INT,
    tipo_asignacion ENUM('practica', 'examen', 'proyecto') DEFAULT 'practica',
    FOREIGN KEY (materia_id) REFERENCES materias(id) ON DELETE CASCADE,
    FOREIGN KEY (nivel_id) REFERENCES niveles(id) ON DELETE CASCADE,
    FOREIGN KEY (autor_id) REFERENCES usuarios(id) ON DELETE CASCADE,
    FOREIGN KEY (concepto_id) REFERENCES conceptos(id) ON DELETE SET NULL,
    FOREIGN KEY (herramienta_id) REFERENCES herramientas(id) ON DELETE SET NULL,
    FOREIGN KEY (grupo_id) REFERENCES grupos(id) ON DELETE SET NULL
);

-- Tabla de relación práctica-competencia
CREATE TABLE IF NOT EXISTS practica_competencia (
    id INT AUTO_INCREMENT PRIMARY KEY,
    practica_id INT NOT NULL,
    competencia_id INT NOT NULL,
    nivel INT NOT NULL,
    FOREIGN KEY (practica_id) REFERENCES practicas(id) ON DELETE CASCADE,
    FOREIGN KEY (competencia_id) REFERENCES competencias(id) ON DELETE CASCADE
);

-- Tabla de prerequisitos de práctica
CREATE TABLE IF NOT EXISTS practica_prerequisitos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    practica_id INT NOT NULL,
    competencia_id INT NOT NULL,
    nivel_requerido INT NOT NULL,
    FOREIGN KEY (practica_id) REFERENCES practicas(id) ON DELETE CASCADE,
    FOREIGN KEY (competencia_id) REFERENCES competencias(id) ON DELETE CASCADE
);

-- Tabla de evaluaciones
CREATE TABLE IF NOT EXISTS evaluaciones (
    id INT AUTO_INCREMENT PRIMARY KEY,
    practica_id INT NOT NULL,
    estudiante_id INT NOT NULL,
    evaluador_id INT NOT NULL,
    fecha_evaluacion DATETIME NOT NULL,
    estado ENUM('pendiente', 'en_proceso', 'completada', 'revisada', 'calificado', 'entregado') NOT NULL,
    calificacion DECIMAL(5,2),
    comentarios TEXT,
    uso_ia INT DEFAULT 0,
    FOREIGN KEY (practica_id) REFERENCES practicas(id) ON DELETE CASCADE,
    FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE,
    FOREIGN KEY (evaluador_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de entregas
CREATE TABLE IF NOT EXISTS entregas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    practica_id INT NOT NULL,
    estudiante_id INT NOT NULL,
    fecha_entrega DATETIME NOT NULL,
    estado ENUM('pendiente', 'revisada', 'retroalimentada', 'calificado', 'entregado') NOT NULL,
    archivos_url TEXT,
    calificacion DECIMAL(5,2),
    contenido TEXT,
    evaluacion_id INT,
    FOREIGN KEY (practica_id) REFERENCES practicas(id) ON DELETE CASCADE,
    FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE,
    FOREIGN KEY (evaluacion_id) REFERENCES evaluaciones(id) ON DELETE SET NULL
);

-- Tabla de retroalimentación
CREATE TABLE IF NOT EXISTS retroalimentacion (
    id INT AUTO_INCREMENT PRIMARY KEY,
    entrega_id INT NOT NULL,
    profesor_id INT NOT NULL,
    comentario TEXT NOT NULL,
    aspecto VARCHAR(100) NOT NULL,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (entrega_id) REFERENCES entregas(id) ON DELETE CASCADE,
    FOREIGN KEY (profesor_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de rúbricas
CREATE TABLE IF NOT EXISTS rubricas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    practica_id INT NOT NULL,
    criterio VARCHAR(200) NOT NULL,
    descripcion TEXT NOT NULL,
    puntaje_maximo INT NOT NULL,
    FOREIGN KEY (practica_id) REFERENCES practicas(id) ON DELETE CASCADE
);

-- Tabla de niveles de rúbrica
CREATE TABLE IF NOT EXISTS rubrica_niveles (
    id INT AUTO_INCREMENT PRIMARY KEY,
    rubrica_id INT NOT NULL,
    nivel INT NOT NULL,
    descripcion TEXT NOT NULL,
    puntaje INT NOT NULL,
    FOREIGN KEY (rubrica_id) REFERENCES rubricas(id) ON DELETE CASCADE
);

-- Tabla de recursos de práctica
CREATE TABLE IF NOT EXISTS recursos_practica (
    id INT AUTO_INCREMENT PRIMARY KEY,
    practica_id INT NOT NULL,
    tipo VARCHAR(50) NOT NULL,
    nombre VARCHAR(200) NOT NULL,
    url TEXT NOT NULL,
    descripcion TEXT,
    FOREIGN KEY (practica_id) REFERENCES practicas(id) ON DELETE CASCADE
);

-- Tabla de notificaciones
CREATE TABLE IF NOT EXISTS notificaciones (
    id INT AUTO_INCREMENT PRIMARY KEY,
    usuario_id INT NOT NULL,
    tipo VARCHAR(50) NOT NULL,
    mensaje TEXT NOT NULL,
    leida BOOLEAN DEFAULT FALSE,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de resultados de aprendizaje
CREATE TABLE IF NOT EXISTS resultados_aprendizaje (
    id INT AUTO_INCREMENT PRIMARY KEY,
    practica_id INT NOT NULL,
    competencia_id INT NOT NULL,
    nivel_logrado INT NOT NULL,
    evidencias TEXT NOT NULL,
    fecha_registro DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (practica_id) REFERENCES practicas(id) ON DELETE CASCADE,
    FOREIGN KEY (competencia_id) REFERENCES competencias(id) ON DELETE CASCADE
);

-- Tabla de asignaciones
CREATE TABLE IF NOT EXISTS asignaciones (
    id INT AUTO_INCREMENT PRIMARY KEY,
    practica_id INT NOT NULL,
    estudiante_id INT NOT NULL,
    fecha_asignacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    fecha_entrega DATETIME,
    estado ENUM('pendiente', 'entregado', 'calificado') DEFAULT 'pendiente',
    archivo_url VARCHAR(255),
    FOREIGN KEY (practica_id) REFERENCES practicas(id) ON DELETE CASCADE,
    FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de calificaciones finales
CREATE TABLE IF NOT EXISTS calificaciones_finales (
    id INT AUTO_INCREMENT PRIMARY KEY,
    estudiante_id INT NOT NULL,
    grupo_id INT NOT NULL,
    calificacion_final DECIMAL(4,2) NOT NULL,
    practicas_promedio DECIMAL(4,2),
    examenes_promedio DECIMAL(4,2),
    proyectos_promedio DECIMAL(4,2),
    asistencia_porcentaje DECIMAL(5,2),
    fecha_calificacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (estudiante_id) REFERENCES usuarios(id) ON DELETE CASCADE,
    FOREIGN KEY (grupo_id) REFERENCES grupos(id) ON DELETE CASCADE,
    UNIQUE KEY unique_estudiante_grupo (estudiante_id, grupo_id)
);

-- Tabla de contenido generado
CREATE TABLE IF NOT EXISTS contenido_generado (
    id INT AUTO_INCREMENT PRIMARY KEY,
    practica_id INT NOT NULL,
    contenido JSON NOT NULL,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (practica_id) REFERENCES practicas(id) ON DELETE CASCADE
);

-- Tabla de versiones
CREATE TABLE IF NOT EXISTS versiones (
    id INT AUTO_INCREMENT PRIMARY KEY,
    practica_id INT NOT NULL,
    contenido JSON NOT NULL,
    autor_id INT NOT NULL,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    numero_version INT NOT NULL,
    cambios TEXT,
    FOREIGN KEY (practica_id) REFERENCES practicas(id) ON DELETE CASCADE,
    FOREIGN KEY (autor_id) REFERENCES usuarios(id) ON DELETE CASCADE
);

-- Tabla de plantillas
CREATE TABLE IF NOT EXISTS plantillas (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre VARCHAR(100) NOT NULL,
    descripcion TEXT,
    contenido JSON NOT NULL,
    autor_id INT,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    categoria VARCHAR(50),
    FOREIGN KEY (autor_id) REFERENCES usuarios(id) ON DELETE SET NULL
);

-- Tabla de actividades
CREATE TABLE IF NOT EXISTS actividades (
    id INT AUTO_INCREMENT PRIMARY KEY,
    descripcion TEXT NOT NULL,
    fecha DATETIME DEFAULT CURRENT_TIMESTAMP,
    usuario_id INT,
    tipo VARCHAR(50),
    recurso_id INT,
    FOREIGN KEY (usuario_id) REFERENCES usuarios(id) ON DELETE SET NULL
);

-- Tabla de evaluaciones de IA
CREATE TABLE IF NOT EXISTS evaluaciones_ia (
    id INT AUTO_INCREMENT PRIMARY KEY,
    evaluacion_id INT NOT NULL,
    contenido_original TEXT NOT NULL,
    calificacion DECIMAL(4,2) NOT NULL,
    retroalimentacion TEXT NOT NULL,
    estilo_aprendizaje_id INT,
    criterios_evaluados JSON,
    errores_detectados JSON,
    recursos_recomendados JSON,
    fecha_evaluacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (evaluacion_id) REFERENCES evaluaciones(id) ON DELETE CASCADE,
    FOREIGN KEY (estilo_aprendizaje_id) REFERENCES estilos_aprendizaje(id) ON DELETE SET NULL
);

-- Tabla de pesos del modelo
CREATE TABLE IF NOT EXISTS modelo_pesos (
    id INT AUTO_INCREMENT PRIMARY KEY,
    nombre_modelo VARCHAR(100) NOT NULL,
    ruta_pesos VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Tabla de recursos de aprendizaje
CREATE TABLE IF NOT EXISTS recursos_aprendizaje (
    id INT AUTO_INCREMENT PRIMARY KEY,
    titulo VARCHAR(255) NOT NULL,
    url VARCHAR(255) NOT NULL,
    tipo_recurso VARCHAR(50) NOT NULL,
    estilo_aprendizaje_id INT,
    categoria VARCHAR(100),
    descripcion TEXT,
    fecha_creacion DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (estilo_aprendizaje_id) REFERENCES estilos_aprendizaje(id) ON DELETE SET NULL
);

-- =============================================
-- DATOS INICIALES
-- =============================================

-- Insertar marcos predefinidos si no existen
INSERT IGNORE INTO marcos_perfil (id, nombre, descripcion, clase_css, condicion_desbloqueo) VALUES
(1, 'Marco Básico', 'Marco básico para estudiantes de primer semestre', 'marco-basico', 'Completar primer semestre'),
(2, 'Marco Segundo Semestre', 'Marco para estudiantes de segundo semestre', 'marco-segundo-semestre', 'Completar segundo semestre'),
(3, 'Marco Tercer Semestre', 'Marco para estudiantes de tercer semestre', 'marco-tercer-semestre', 'Completar tercer semestre'),
(4, 'Marco Cuarto Semestre', 'Marco para estudiantes de cuarto semestre', 'marco-cuarto-semestre', 'Completar cuarto semestre'),
(5, 'Marco Quinto Semestre', 'Marco para estudiantes de quinto semestre', 'marco-quinto-semestre', 'Completar quinto semestre'),
(6, 'Marco Sexto Semestre', 'Marco para estudiantes de sexto semestre', 'marco-sexto-semestre', 'Completar sexto semestre'),
(7, 'Marco Séptimo Semestre', 'Marco para estudiantes de séptimo semestre', 'marco-septimo-semestre', 'Completar séptimo semestre'),
(8, 'Marco Octavo Semestre', 'Marco para estudiantes de octavo semestre', 'marco-octavo-semestre', 'Completar octavo semestre'),
(9, 'Marco Noveno Semestre', 'Marco para estudiantes de noveno semestre', 'marco-noveno-semestre', 'Completar noveno semestre'),
(10, 'Marco de Excelencia', 'Marco especial para estudiantes con promedio superior a 9.5', 'marco-excelencia', 'Obtener promedio superior a 9.5'),
(11, 'Marco Responsable', 'Marco para estudiantes que llevan 3 meses sin tareas atrasadas', 'marco-responsable', 'Mantener 3 meses sin tareas atrasadas'),
(12, 'Marco Número Uno', 'Marco exclusivo para el estudiante con mejor promedio de su clase', 'marco-numero-uno', 'Ser el estudiante con mejor promedio de la clase');

-- Insertar estilos de aprendizaje predeterminados
INSERT IGNORE INTO estilos_aprendizaje (nombre, descripcion) VALUES
('visual', 'Aprende mejor a través de imágenes, diagramas y representaciones visuales'),
('auditivo', 'Aprende mejor a través de la escucha y la discusión verbal'),
('kinestesico', 'Aprende mejor a través de la experiencia práctica y la actividad física'),
('lectura_escritura', 'Aprende mejor a través de la lectura y la escritura de textos'),
('multimodal', 'Aprende utilizando una combinación de estilos de aprendizaje'),
('analítico', 'Aprende mejor a través del análisis lógico y la descomposición de conceptos'),
('holístico', 'Aprende mejor viendo el panorama completo y las conexiones entre conceptos');

-- Insertar semestres predeterminados
INSERT IGNORE INTO semestres (nombre, fecha_inicio, fecha_fin, activo)
VALUES 
('2023-1', '2023-08-07', '2023-12-15', FALSE),
('2023-2', '2024-01-29', '2024-06-07', FALSE),
('2024-1', '2024-08-05', '2024-12-13', TRUE);


-- Insertar materias
INSERT INTO materias (nombre, descripcion) VALUES
('Base de datos', 'Fundamentos y aplicaciones de bases de datos'),
('Programación', 'Programación y desarrollo de software'),
('Machine Learning', 'Aprendizaje automático y análisis de datos'),
('Seguridad Informática', 'Seguridad de sistemas y redes');

-- Insertar niveles
INSERT INTO niveles (nombre, descripcion) VALUES
('Básico', 'Nivel fundamental de conocimientos'),
('Intermedio', 'Nivel medio de conocimientos'),
('Avanzado', 'Nivel experto de conocimientos');

-- Insertar competencias
INSERT INTO competencias (nombre, descripcion) VALUES
('Diseño de BD', 'Capacidad para diseñar bases de datos eficientes'),
('Optimización', 'Habilidad para optimizar consultas y estructuras'),
('Programación SQL', 'Dominio de SQL y procedimientos almacenados'),
('Seguridad', 'Implementación de medidas de seguridad en BD');

-- Insertar herramientas
INSERT INTO herramientas (nombre, descripcion, tipo) VALUES
('MySQL', 'Sistema de gestión de bases de datos relacionales', 'DBMS'),
('PostgreSQL', 'Sistema de BD relacional avanzado', 'DBMS'),
('MongoDB', 'Base de datos NoSQL', 'NoSQL'),
('DBeaver', 'Cliente universal de bases de datos', 'Cliente');

-- Completar conceptos para todas las materias
INSERT INTO conceptos (materia_id, nombre, descripcion) VALUES
-- Base de datos (continuación)
(1, 'Bases de datos NoSQL', 'Sistemas de bases de datos no relacionales'),
(1, 'Seguridad en BD', 'Implementación de medidas de seguridad'),

-- Programación
(2, 'Estructuras de datos', 'Organización y manipulación de datos'),
(2, 'Algoritmos', 'Diseño y análisis de algoritmos'),
(2, 'POO', 'Programación Orientada a Objetos'),
(2, 'Patrones de diseño', 'Soluciones comunes a problemas de diseño'),

-- Machine Learning
(3, 'Regresión', 'Modelos de regresión y predicción'),
(3, 'Clasificación', 'Algoritmos de clasificación'),
(3, 'Clustering', 'Agrupamiento de datos'),
(3, 'Redes neuronales', 'Deep learning y redes neuronales'),

-- Seguridad Informática
(4, 'Criptografía', 'Técnicas de cifrado y seguridad'),
(4, 'Seguridad en redes', 'Protección de redes y comunicaciones'),
(4, 'Ethical Hacking', 'Pruebas de penetración'),
(4, 'Forense digital', 'Análisis forense de sistemas');
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re


class GeneradorPracticasML:
    """Generador de prácticas basado en scikit-learn en lugar del modelo personalizado."""

    def __init__(self):
        # Inicializar el vectorizador TF-IDF
        self.vectorizer = TfidfVectorizer(stop_words='english')

        # Base de conocimientos simple (simulada)
        self.base_conocimiento = [
            "Resolver ecuaciones diferenciales de primer orden utilizando métodos numéricos.",
            "Análisis de datos utilizando técnicas estadísticas descriptivas e inferenciales.",
            "Implementación de algoritmos de ordenamiento y búsqueda para estructuras de datos.",
            "Diseño de experimentos científicos con controles apropiados.",
            "Desarrollo de aplicaciones web con frameworks modernos.",
            "Análisis literario de obras contemporáneas.",
            "Estudio de reacciones químicas orgánicas.",
            "Diseño de circuitos electrónicos analógicos y digitales.",
        ]

        # Vectorizar la base de conocimientos
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.base_conocimiento)

    def generar_practica(self, titulo, objetivo):
        """Genera contenido para una práctica basada en el título y objetivo proporcionados."""
        # Limpiar y preparar la consulta
        consulta = f"{titulo} {objetivo}"
        consulta = re.sub(r'[^\w\s]', '', consulta.lower())

        # Vectorizar la consulta
        consulta_vector = self.vectorizer.transform([consulta])

        # Calcular similitud con la base de conocimientos
        similitudes = cosine_similarity(consulta_vector, self.tfidf_matrix)[0]

        # Encontrar el contenido más similar
        idx_mas_similar = np.argmax(similitudes)
        similitud = similitudes[idx_mas_similar]

        # Generar contenido basado en la similitud
        if similitud > 0.1:
            base = self.base_conocimiento[idx_mas_similar]
        else:
            base = "Práctica general de aplicación de conocimientos."

        # Generar estructura de la práctica
        practica_generada = {
            "descripcion": f"Esta práctica se enfoca en: {base}",
            "objetivos_especificos": [
                f"Comprender los conceptos fundamentales relacionados con {titulo}",
                f"Aplicar técnicas para resolver problemas de {objetivo}",
                "Desarrollar habilidades analíticas y de resolución de problemas"
            ],
            "actividades": [
                {
                    "numero": 1,
                    "descripcion": f"Investigación sobre {titulo}",
                    "tiempo_estimado": "2 horas"
                },
                {
                    "numero": 2,
                    "descripcion": f"Resolución de problemas relacionados con {objetivo}",
                    "tiempo_estimado": "3 horas"
                },
                {
                    "numero": 3,
                    "descripcion": "Presentación y discusión de resultados",
                    "tiempo_estimado": "1 hora"
                }
            ],
            "recursos": [
                "Libros de texto relacionados con la materia",
                "Artículos científicos recientes",
                "Herramientas de software específicas"
            ],
            "criterios_evaluacion": [
                "Comprensión de conceptos (30%)",
                "Aplicación práctica (40%)",
                "Presentación y comunicación (30%)"
            ],
            "recomendaciones": f"Se recomienda revisar conceptos básicos de {titulo.split()[0]} antes de comenzar."
        }

        return practica_generada

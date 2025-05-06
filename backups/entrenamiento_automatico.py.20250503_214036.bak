import os
import sys
import logging
import torch
import numpy as np
from modelo_ml_enhanced import EnhancedModeloEvaluacionInteligente
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from datetime import datetime
import json
import random
from typing import List, Dict, Tuple

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EntrenadorAutomatico:
    """
    Sistema de entrenamiento automático para el modelo de evaluación.
    Utiliza técnicas no supervisadas para analizar y aprender de las entregas.
    """
    def __init__(self, db_config=None, modelo=None):
        """
        Inicializa el entrenador automático.
        
        Args:
            db_config: Configuración para la base de datos
            modelo: Instancia del modelo de evaluación (opcional)
        """
        self.db_config = db_config
        
        # Inicializar modelo si no se proporciona uno
        if modelo is None:
            try:
                # Intentar usar GPU si está disponible
                use_gpu = torch.cuda.is_available()
                device = "cuda" if use_gpu else "cpu"
                logger.info(f"Inicializando modelo en dispositivo: {device}")
                
                # Inicializar con SBERT si es posible
                try:
                    self.modelo = EnhancedModeloEvaluacionInteligente(db_config=db_config, use_sbert=True)
                    logger.info("Modelo inicializado con soporte SBERT")
                except Exception as e:
                    logger.warning(f"No se pudo inicializar con SBERT: {e}")
                    self.modelo = EnhancedModeloEvaluacionInteligente(db_config=db_config, use_sbert=False)
                    logger.info("Modelo inicializado sin soporte SBERT")
            except Exception as e:
                logger.error(f"Error al inicializar modelo: {e}")
                raise
        else:
            self.modelo = modelo
        
        # Inicializar componentes para entrenamiento no supervisado
        self.kmeans = None
        self.pca = None
        self.vectores_entrenamiento = []
        self.textos_entrenamiento = []
        self.clusters = {}
        
        # Directorio para guardar resultados
        self.resultados_dir = "resultados_entrenamiento"
        os.makedirs(self.resultados_dir, exist_ok=True)
    
    def recopilar_entregas(self) -> List[Dict]:
        """
        Recopila todas las entregas disponibles en la base de datos.
        
        Returns:
            Lista de diccionarios con información de las entregas
        """
        if not self.db_config or not self.modelo.connection:
            logger.warning("No hay conexión a la base de datos para recopilar entregas")
            return []
        
        try:
            query = """
            SELECT e.id, e.practica_id, e.estudiante_id, e.contenido, e.calificacion,
                   p.titulo, p.objetivo
            FROM entregas e
            JOIN practicas p ON e.practica_id = p.id
            WHERE e.contenido IS NOT NULL AND LENGTH(e.contenido) > 0
            """
            self.modelo.cursor.execute(query)
            entregas = self.modelo.cursor.fetchall()
            
            logger.info(f"Se recopilaron {len(entregas)} entregas de la base de datos")
            return entregas
        except Exception as e:
            logger.error(f"Error al recopilar entregas: {e}")
            return []
    
    def vectorizar_entregas(self, entregas: List[Dict]) -> np.ndarray:
        """
        Vectoriza las entregas para su análisis.
        
        Args:
            entregas: Lista de entregas a vectorizar
            
        Returns:
            Array de NumPy con los vectores de las entregas
        """
        vectores = []
        textos = []
        
        for entrega in entregas:
            contenido = entrega.get('contenido', '')
            if not contenido or len(contenido.strip()) == 0:
                continue
                
            # Preprocesar el texto
            contenido_procesado = self.modelo._preprocess_text(contenido)
            textos.append({
                'id': entrega.get('id'),
                'practica_id': entrega.get('practica_id'),
                'estudiante_id': entrega.get('estudiante_id'),
                'titulo': entrega.get('titulo', ''),
                'objetivo': entrega.get('objetivo', ''),
                'calificacion': entrega.get('calificacion')
            })
            
            # Vectorizar con TF-IDF
            try:
                vector = self.modelo.vectorizer.transform([contenido_procesado]).toarray()[0]
                
                # Asegurar dimensiones consistentes
                if len(vector) < 5000:
                    vector = np.pad(vector, (0, 5000 - len(vector)), 'constant')
                elif len(vector) > 5000:
                    vector = vector[:5000]
                    
                vectores.append(vector)
            except Exception as e:
                logger.error(f"Error al vectorizar entrega {entrega.get('id')}: {e}")
        
        self.vectores_entrenamiento = np.array(vectores)
        self.textos_entrenamiento = textos
        
        logger.info(f"Se vectorizaron {len(vectores)} entregas")
        return self.vectores_entrenamiento
    
    def aplicar_pca(self, n_componentes=50):
        """
        Aplica PCA para reducir la dimensionalidad de los vectores.
        
        Args:
            n_componentes: Número de componentes principales a utilizar
        
        Returns:
            Vectores transformados con PCA
        """
        if len(self.vectores_entrenamiento) == 0:
            logger.warning("No hay vectores para aplicar PCA")
            return np.array([])
        
        try:
            self.pca = PCA(n_components=min(n_componentes, len(self.vectores_entrenamiento)))
            vectores_pca = self.pca.fit_transform(self.vectores_entrenamiento)
            
            logger.info(f"PCA aplicado: {self.vectores_entrenamiento.shape} -> {vectores_pca.shape}")
            logger.info(f"Varianza explicada: {sum(self.pca.explained_variance_ratio_):.4f}")
            
            return vectores_pca
        except Exception as e:
            logger.error(f"Error al aplicar PCA: {e}")
            return self.vectores_entrenamiento
    
    def agrupar_entregas(self, vectores, n_clusters=5):
        """
        Agrupa las entregas en clusters utilizando K-means.
        
        Args:
            vectores: Vectores de las entregas
            n_clusters: Número de clusters a crear
            
        Returns:
            Etiquetas de cluster para cada entrega
        """
        if len(vectores) == 0:
            logger.warning("No hay vectores para agrupar")
            return np.array([])
        
        try:
            # Ajustar número de clusters si hay pocas muestras
            n_clusters = min(n_clusters, len(vectores) // 2)
            if n_clusters < 2:
                n_clusters = 2
                
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            etiquetas = self.kmeans.fit_predict(vectores)
            
            # Organizar entregas por cluster
            self.clusters = {}
            for i, etiqueta in enumerate(etiquetas):
                if etiqueta not in self.clusters:
                    self.clusters[etiqueta] = []
                self.clusters[etiqueta].append(i)
            
            logger.info(f"Entregas agrupadas en {n_clusters} clusters")
            for cluster, indices in self.clusters.items():
                logger.info(f"Cluster {cluster}: {len(indices)} entregas")
            
            return etiquetas
        except Exception as e:
            logger.error(f"Error al agrupar entregas: {e}")
            return np.array([0] * len(vectores))
    
    def generar_ejemplos_entrenamiento(self, n_ejemplos=100):
        """
        Genera ejemplos de entrenamiento a partir de los clusters.
        
        Args:
            n_ejemplos: Número de ejemplos a generar
            
        Returns:
            Tupla de (entradas, salidas) para entrenamiento
        """
        if not self.clusters:
            logger.warning("No hay clusters para generar ejemplos")
            return [], []
        
        entradas = []
        salidas = []
        
        try:
            # Determinar calificaciones promedio por cluster
            calificaciones_cluster = {}
            for cluster, indices in self.clusters.items():
                calificaciones = []
                for idx in indices:
                    if idx < len(self.textos_entrenamiento):
                        cal = self.textos_entrenamiento[idx].get('calificacion')
                        if cal is not None:
                            calificaciones.append(float(cal))
                
                if calificaciones:
                    calificaciones_cluster[cluster] = sum(calificaciones) / len(calificaciones)
                else:
                    # Asignar calificación basada en la posición del cluster (para clusters sin calificaciones)
                    calificaciones_cluster[cluster] = 5.0 + (cluster / len(self.clusters)) * 4.0
            
            logger.info("Calificaciones promedio por cluster:")
            for cluster, cal in calificaciones_cluster.items():
                logger.info(f"Cluster {cluster}: {cal:.2f}")
            
            # Generar ejemplos balanceados de cada cluster
            ejemplos_por_cluster = max(1, n_ejemplos // len(self.clusters))
            
            for cluster, indices in self.clusters.items():
                if not indices:
                    continue
                    
                # Seleccionar ejemplos aleatorios del cluster
                n_seleccionar = min(ejemplos_por_cluster, len(indices))
                seleccionados = random.sample(indices, n_seleccionar)
                
                for idx in seleccionados:
                    if idx < len(self.textos_entrenamiento):
                        texto = self.textos_entrenamiento[idx]
                        titulo = texto.get('titulo', '')
                        objetivo = texto.get('objetivo', '')
                        
                        # Obtener contenido original
                        query = "SELECT contenido FROM entregas WHERE id = %s"
                        self.modelo.cursor.execute(query, (texto.get('id'),))
                        resultado = self.modelo.cursor.fetchone()
                        
                        if resultado and resultado.get('contenido'):
                            contenido = resultado.get('contenido')
                            
                            # Usar calificación real si existe, o la del cluster
                            if texto.get('calificacion') is not None:
                                calificacion = float(texto.get('calificacion'))
                            else:
                                calificacion = calificaciones_cluster[cluster]
                            
                            entradas.append(contenido)
                            salidas.append(calificacion)
            
            logger.info(f"Generados {len(entradas)} ejemplos de entrenamiento")
            return entradas, salidas
        except Exception as e:
            logger.error(f"Error al generar ejemplos de entrenamiento: {e}")
            return [], []
    
    def entrenar_modelo_automatico(self, epochs=30, batch_size=8):
        """
        Entrena el modelo automáticamente con los ejemplos generados.
        
        Args:
            epochs: Número de épocas de entrenamiento
            batch_size: Tamaño del lote para entrenamiento
            
        Returns:
            Resultados del entrenamiento
        """
        # Recopilar y procesar entregas
        entregas = self.recopilar_entregas()
        if not entregas:
            logger.warning("No hay entregas para entrenar el modelo")
            return {"status": "error", "message": "No hay entregas disponibles"}
        
        # Vectorizar entregas
        vectores = self.vectorizar_entregas(entregas)
        if len(vectores) == 0:
            logger.warning("No se pudieron vectorizar las entregas")
            return {"status": "error", "message": "Error al vectorizar entregas"}
        
        # Aplicar PCA para reducir dimensionalidad
        vectores_pca = self.aplicar_pca(n_componentes=min(50, len(vectores)))
        
        # Agrupar entregas
        self.agrupar_entregas(vectores_pca, n_clusters=min(5, len(vectores) // 5))
        
        # Generar ejemplos de entrenamiento
        entradas, salidas = self.generar_ejemplos_entrenamiento(n_ejemplos=min(100, len(vectores)))
        
        if not entradas or not salidas:
            logger.warning("No se pudieron generar ejemplos de entrenamiento")
            return {"status": "error", "message": "No hay ejemplos de entrenamiento"}
        
        # Entrenar el modelo
        try:
            resultado = self.modelo.entrenar_modelo(entradas, salidas, epochs=epochs, batch_size=batch_size)
            
            # Guardar el modelo entrenado
            self.modelo.guardar_modelos()
            
            # Guardar resultados del entrenamiento
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_resultados = os.path.join(self.resultados_dir, f"entrenamiento_{timestamp}.json")
            
            with open(ruta_resultados, 'w', encoding='utf-8') as f:
                json.dump({
                    "fecha": timestamp,
                    "num_entregas": len(entregas),
                    "num_ejemplos": len(entradas),
                    "epochs": resultado.get("epochs_completed", 0),
                    "loss_final": resultado.get("train_loss", 0),
                    "val_loss": resultado.get("val_loss", 0),
                    "clusters": {str(k): len(v) for k, v in self.clusters.items()}
                }, f, indent=2)
            
            logger.info(f"Entrenamiento completado y resultados guardados en {ruta_resultados}")
            return resultado
        except Exception as e:
            logger.error(f"Error al entrenar el modelo: {e}")
            return {"status": "error", "message": str(e)}
    
    def visualizar_clusters(self):
        """
        Genera una visualización de los clusters de entregas.
        
        Returns:
            Ruta al archivo de imagen generado
        """
        if not self.clusters or not self.pca:
            logger.warning("No hay clusters o PCA para visualizar")
            return None
        
        try:
            # Reducir a 2 dimensiones para visualización
            pca_2d = PCA(n_components=2)
            vectores_2d = pca_2d.fit_transform(self.vectores_entrenamiento)
            
            # Crear figura
            plt.figure(figsize=(12, 8))
            
            # Colores para los clusters
            colores = ['r', 'g', 'b', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
            
            # Graficar puntos por cluster
            for cluster, indices in self.clusters.items():
                color = colores[cluster % len(colores)]
                for idx in indices:
                    if idx < len(vectores_2d):
                        plt.scatter(vectores_2d[idx, 0], vectores_2d[idx, 1], c=color, alpha=0.5)
                
                # Calcular y graficar centroide
                if indices:
                    centroide = np.mean([vectores_2d[idx] for idx in indices if idx < len(vectores_2d)], axis=0)
                    plt.scatter(centroide[0], centroide[1], c=color, marker='X', s=200, edgecolors='k')
                    plt.annotate(f'Cluster {cluster}', (centroide[0], centroide[1]), 
                                fontsize=12, fontweight='bold')
            
            plt.title('Visualización de Clusters de Entregas', fontsize=16)
            plt.xlabel('Componente Principal 1', fontsize=12)
            plt.ylabel('Componente Principal 2', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Guardar figura
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ruta_imagen = os.path.join(self.resultados_dir, f"clusters_{timestamp}.png")
            plt.savefig(ruta_imagen, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Visualización de clusters guardada en {ruta_imagen}")
            return ruta_imagen
        except Exception as e:
            logger.error(f"Error al visualizar clusters: {e}")
            return None

# Función principal para ejecutar el entrenamiento automático
def ejecutar_entrenamiento_automatico(db_config=None):
    """
    Ejecuta el proceso completo de entrenamiento automático.
    
    Args:
        db_config: Configuración para la base de datos
        
    Returns:
        Resultados del entrenamiento
    """
    try:
        logger.info("Iniciando entrenamiento automático")
        
        # Inicializar entrenador
        entrenador = EntrenadorAutomatico(db_config=db_config)
        
        # Ejecutar entrenamiento
        resultado = entrenador.entrenar_modelo_automatico()
        
        # Generar visualización
        entrenador.visualizar_clusters()
        
        return resultado
    except Exception as e:
        logger.error(f"Error en el entrenamiento automático: {e}")
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    # Configuración de la base de datos (ejemplo)
    db_config = {
        "host": "localhost",
        "user": "root",
        "password": "",
        "database": "generador_practicas"
    }
    
    # Ejecutar entrenamiento
    resultado = ejecutar_entrenamiento_automatico(db_config)
    print(f"Resultado del entrenamiento: {resultado}")
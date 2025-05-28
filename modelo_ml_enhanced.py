import numpy as np
import re
import torch
import sys
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import json
from datetime import datetime
import os
import pickle
import logging
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any
import mysql.connector
from mysql.connector import Error
import gc
import threading
import time
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import traceback
import warnings
import hashlib
import psutil
import signal
import tempfile
import queue
import asyncio
import functools
import weakref

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("modelo_ml.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Importar NLTK de manera segura
try:
    import nltk
    from nltk.corpus import stopwords
    # Descargar recursos necesarios si no existen
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except ImportError:
    logger.warning("NLTK no está disponible. Usando stopwords predefinidas.")
    nltk = None

# Clase para gestionar recursos compartidos
class ResourceManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ResourceManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
            
        self.resources = {}
        self.last_access = {}
        self.resource_locks = {}
        self.max_idle_time = 600  # 10 minutos
        self._cleanup_thread = None
        self._stop_event = threading.Event()
        self._initialized = True
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Inicia un hilo para limpiar recursos no utilizados."""
        def cleanup_worker():
            while not self._stop_event.is_set():
                self._cleanup_idle_resources()
                time.sleep(60)  # Verificar cada minuto
        
        self._cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        self._cleanup_thread.start()
    
    def _cleanup_idle_resources(self):
        """Libera recursos que no se han utilizado por un tiempo."""
        current_time = time.time()
        resources_to_remove = []
        
        for resource_id, last_access in self.last_access.items():
            if current_time - last_access > self.max_idle_time:
                resources_to_remove.append(resource_id)
        
        for resource_id in resources_to_remove:
            with self._lock:
                if resource_id in self.resources:
                    logger.info(f"Liberando recurso no utilizado: {resource_id}")
                    del self.resources[resource_id]
                    del self.last_access[resource_id]
                    if resource_id in self.resource_locks:
                        del self.resource_locks[resource_id]
        
        # Forzar recolección de basura después de liberar recursos
        gc.collect()
    
    def get_resource(self, resource_id, creator_func):
        """Obtiene un recurso, creándolo si no existe."""
        with self._lock:
            if resource_id not in self.resources:
                logger.info(f"Creando nuevo recurso: {resource_id}")
                self.resources[resource_id] = creator_func()
                self.resource_locks[resource_id] = threading.Lock()
            
            self.last_access[resource_id] = time.time()
            return self.resources[resource_id]
    
    def get_lock(self, resource_id):
        """Obtiene el lock para un recurso específico."""
        with self._lock:
            if resource_id not in self.resource_locks:
                self.resource_locks[resource_id] = threading.Lock()
            return self.resource_locks[resource_id]
    
    def shutdown(self):
        """Detiene el hilo de limpieza y libera todos los recursos."""
        self._stop_event.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1)
        
        with self._lock:
            self.resources.clear()
            self.last_access.clear()
            self.resource_locks.clear()
        
        gc.collect()

# Singleton para gestionar recursos
resource_manager = ResourceManager()

# Manejador de señales para limpieza adecuada
def signal_handler(sig, frame):
    logger.info("Señal de terminación recibida. Limpiando recursos...")
    resource_manager.shutdown()
    sys.exit(0)

# Registrar manejadores de señales
try:
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
except (AttributeError, ValueError):
    # Algunos entornos no soportan estas señales
    pass

# Clase para dataset de texto
class TextDataset(Dataset):
    def __init__(self, entradas, salidas, vectorizer):
        self.entradas = entradas
        self.salidas = salidas
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.entradas)

    def __getitem__(self, idx):
        # Vectorizar el texto de entrada
        try:
            entrada = self.vectorizer.transform([self.entradas[idx]]).toarray()[0]
        except Exception as e:
            logger.error(f"Error al vectorizar entrada {idx}: {e}")
            entrada = np.zeros(5000)  # Vector vacío como fallback
        
        # Asegurar tamaño consistente del vector
        if len(entrada) < 5000:
            entrada = np.pad(entrada, (0, 5000 - len(entrada)), 'constant')
        elif len(entrada) > 5000:
            entrada = entrada[:5000]

        salida = float(self.salidas[idx])
        return torch.tensor(entrada, dtype=torch.float32), torch.tensor(salida, dtype=torch.float32)

# Modelo mejorado para evaluación de contenido
class EnhancedContenidoEvaluator(nn.Module):
    """Modelo mejorado para evaluar diferentes tipos de contenido con arquitectura más profunda"""
    def __init__(self, input_dim=5000, hidden_dim=768, output_dim=1, dropout_rate=0.3):
        super(EnhancedContenidoEvaluator, self).__init__()
        # Arquitectura más profunda con residual connections
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Bloque residual 1
        self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn1 = nn.BatchNorm1d(hidden_dim)
        self.res1_dropout = nn.Dropout(dropout_rate)
        self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Bloque residual 2
        self.res2_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res2_bn1 = nn.BatchNorm1d(hidden_dim)
        self.res2_dropout = nn.Dropout(dropout_rate)
        self.res2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res2_bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Capa de reducción
        self.fc_reduce = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_reduce = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout_reduce = nn.Dropout(dropout_rate * 0.7)
        
        # Múltiples cabezas para diferentes criterios de evaluación
        self.head_relevancia = nn.Linear(hidden_dim // 2, 1)
        self.head_claridad = nn.Linear(hidden_dim // 2, 1)
        self.head_profundidad = nn.Linear(hidden_dim // 2, 1)
        self.head_estructura = nn.Linear(hidden_dim // 2, 1)
        self.head_global = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Inicialización de pesos mejorada
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa los pesos para mejorar la convergencia."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Capa inicial
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Bloque residual 1
        residual = x
        out = self.res1_fc1(x)
        out = self.res1_bn1(out)
        out = self.relu(out)
        out = self.res1_dropout(out)
        out = self.res1_fc2(out)
        out = self.res1_bn2(out)
        out += residual  # Conexión residual
        out = self.relu(out)
        
        # Bloque residual 2
        residual = out
        out = self.res2_fc1(out)
        out = self.res2_bn1(out)
        out = self.relu(out)
        out = self.res2_dropout(out)
        out = self.res2_fc2(out)
        out = self.res2_bn2(out)
        out += residual  # Conexión residual
        out = self.relu(out)
        
        # Reducción de dimensionalidad
        x = self.fc_reduce(out)
        x = self.bn_reduce(x)
        x = self.relu(x)
        x = self.dropout_reduce(x)
        
        # Salidas para cada criterio (escaladas a 0-10)
        relevancia = self.sigmoid(self.head_relevancia(x)) * 10
        claridad = self.sigmoid(self.head_claridad(x)) * 10
        profundidad = self.sigmoid(self.head_profundidad(x)) * 10
        estructura = self.sigmoid(self.head_estructura(x)) * 10
        
        # Calificación global
        global_score = self.sigmoid(self.head_global(x)) * 10
        
        return {
            'relevancia': relevancia,
            'claridad': claridad,
            'profundidad': profundidad,
            'estructura': estructura,
            'global': global_score
        }

# Modelo especializado mejorado para evaluar código
class EnhancedCodigoEvaluator(nn.Module):
    """Modelo especializado mejorado para evaluar código de programación"""
    def __init__(self, input_dim=5000, hidden_dim=768, output_dim=1, dropout_rate=0.3):
        super(EnhancedCodigoEvaluator, self).__init__()
        # Arquitectura más profunda con residual connections
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Bloque residual 1
        self.res1_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn1 = nn.BatchNorm1d(hidden_dim)
        self.res1_dropout = nn.Dropout(dropout_rate)
        self.res1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res1_bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Bloque residual 2
        self.res2_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.res2_bn1 = nn.BatchNorm1d(hidden_dim)
        self.res2_dropout = nn.Dropout(dropout_rate)
        self.res2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.res2_bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Capa de reducción
        self.fc_reduce = nn.Linear(hidden_dim, hidden_dim // 2)
        self.bn_reduce = nn.BatchNorm1d(hidden_dim // 2)
        self.dropout_reduce = nn.Dropout(dropout_rate * 0.7)
        
        # Cabezas específicas para evaluación de código
        self.head_funcionalidad = nn.Linear(hidden_dim // 2, 1)
        self.head_eficiencia = nn.Linear(hidden_dim // 2, 1)
        self.head_estilo = nn.Linear(hidden_dim // 2, 1)
        self.head_documentacion = nn.Linear(hidden_dim // 2, 1)
        self.head_cumplimiento = nn.Linear(hidden_dim // 2, 1)
        self.head_seguridad = nn.Linear(hidden_dim // 2, 1)
        self.head_global = nn.Linear(hidden_dim // 2, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        # Inicialización de pesos mejorada
        self._init_weights()
    
    def _init_weights(self):
        """Inicializa los pesos para mejorar la convergencia."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Capa inicial
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # Bloque residual 1
        residual = x
        out = self.res1_fc1(x)
        out = self.res1_bn1(out)
        out = self.relu(out)
        out = self.res1_dropout(out)
        out = self.res1_fc2(out)
        out = self.res1_bn2(out)
        out += residual  # Conexión residual
        out = self.relu(out)
        
        # Bloque residual 2
        residual = out
        out = self.res2_fc1(out)
        out = self.res2_bn1(out)
        out = self.relu(out)
        out = self.res2_dropout(out)
        out = self.res2_fc2(out)
        out = self.res2_bn2(out)
        out += residual  # Conexión residual
        out = self.relu(out)
        
        # Reducción de dimensionalidad
        x = self.fc_reduce(out)
        x = self.bn_reduce(x)
        x = self.relu(x)
        x = self.dropout_reduce(x)
        
        # Salidas para cada criterio (escaladas a 0-10)
        funcionalidad = self.sigmoid(self.head_funcionalidad(x)) * 10
        eficiencia = self.sigmoid(self.head_eficiencia(x)) * 10
        estilo = self.sigmoid(self.head_estilo(x)) * 10
        documentacion = self.sigmoid(self.head_documentacion(x)) * 10
        cumplimiento = self.sigmoid(self.head_cumplimiento(x)) * 10
        seguridad = self.sigmoid(self.head_seguridad(x)) * 10
        
        # Calificación global
        global_score = self.sigmoid(self.head_global(x)) * 10
        
        return {
            'funcionalidad': funcionalidad,
            'eficiencia': eficiencia,
            'estilo': estilo,
            'documentacion': documentacion,
            'cumplimiento': cumplimiento,
            'seguridad': seguridad,
            'global': global_score
        }

# Modelo dummy para fallback
class DummyModel:
    """Modelo dummy que siempre devuelve un valor predeterminado."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicializando modelo dummy")
    
    def eval(self):
        """Método para compatibilidad con PyTorch."""
        return self
    
    def __call__(self, *args, **kwargs):
        """Devuelve un valor predeterminado."""
        return {'global': torch.tensor([[7.5]], dtype=torch.float32)}
    
    def to(self, device):
        """Método para compatibilidad con PyTorch."""
        return self

# Clase para monitorear recursos del sistema
class SystemMonitor:
    @staticmethod
    def get_memory_usage():
        """Obtiene el uso actual de memoria."""
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)  # MB
        except Exception:
            return 0
    
    @staticmethod
    def get_cpu_usage():
        """Obtiene el uso actual de CPU."""
        try:
            return psutil.cpu_percent(interval=0.1)
        except Exception:
            return 0
    
    @staticmethod
    def check_resources(memory_threshold_mb=1000, cpu_threshold=80):
        """Verifica si hay recursos suficientes disponibles."""
        try:
            memory_usage = SystemMonitor.get_memory_usage()
            cpu_usage = SystemMonitor.get_cpu_usage()
            
            if memory_usage > memory_threshold_mb:
                logger.warning(f"Uso de memoria alto: {memory_usage:.2f} MB")
                return False
            
            if cpu_usage > cpu_threshold:
                logger.warning(f"Uso de CPU alto: {cpu_usage:.2f}%")
                return False
            
            return True
        except Exception:
            return True  # Asumir que hay recursos disponibles si no se puede verificar

# Clase principal mejorada
class GeneradorPracticasML:
    """
    Modelo mejorado para evaluación inteligente de prácticas académicas.
    Detecta automáticamente el tipo de contenido y aplica el evaluador adecuado.
    Incorpora aprendizaje continuo, adaptación al estilo de aprendizaje y optimizaciones.
    """
    def __init__(self, db_config=None, use_sbert=True, max_workers=4, memory_threshold_mb=1000):
        """
        Inicializa el modelo de evaluación inteligente mejorado.
        
        Args:
            db_config: Configuración para la conexión a la base de datos
            use_sbert: Si es True, utiliza modelos SBERT para análisis semántico profundo
            max_workers: Número máximo de workers para procesamiento paralelo
            memory_threshold_mb: Umbral de memoria para activar optimizaciones
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info("Inicializando GeneradorPracticasML mejorado")
        
        # Configuración
        self.db_config = db_config
        self.use_sbert = use_sbert
        self.max_workers = max_workers
        self.memory_threshold_mb = memory_threshold_mb
        self.sbert_model = None
        self.connection = None
        self.cursor = None
        
        # Semáforo para limitar el número de evaluaciones concurrentes
        self.semaphore = threading.Semaphore(max_workers)
        
        # Executor para procesamiento paralelo
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Configurar stopwords en español
        self.spanish_stopwords = self._get_spanish_stopwords()
        
        # Vectorizador TF-IDF mejorado
        self.vectorizer = self._create_vectorizer()
        
        # Configurar dispositivo para PyTorch
        self.device = self._setup_device()
        
        # Inicializar modelos
        self._initialize_models()
        
        # Base de conocimiento ampliada para generación de prácticas
        self.base_conocimiento = self._load_knowledge_base()
        
        # Cargar plantillas de retroalimentación
        self.cargar_plantillas_retroalimentacion()
        
        # Cargar recursos educativos por estilo de aprendizaje
        self.cargar_recursos_educativos()
        
        # Intentar cargar modelos pre-entrenados
        self.cargar_modelos()
        
        # Diccionario para almacenar métricas de evaluación
        self.metricas_evaluacion = self._setup_evaluation_metrics()
        
        # Inicializar contador de evaluaciones para seguimiento
        self.contador_evaluaciones = 0
        
        # Historial de evaluaciones para aprendizaje continuo
        self.historial_evaluaciones = []
        
        # Conectar a la base de datos si se proporciona configuración
        if db_config:
            self._connect_to_database()
        
        # Iniciar hilo de monitoreo de recursos
        self._start_resource_monitoring()
        
        # Crear matriz TF-IDF para la base de conocimiento
        self._create_tfidf_matrix()
    
    def _get_spanish_stopwords(self):
        """Obtiene stopwords en español de manera segura."""
        try:
            if nltk is not None:
                return set(stopwords.words('spanish'))
        except Exception as e:
            self.logger.warning(f"Error al cargar stopwords de NLTK: {e}")
        
        # Fallback a stopwords predefinidas
        return {
            'a', 'al', 'algo', 'algunas', 'algunos', 'ante', 'antes', 'como', 'con', 'contra',
            'cual', 'cuando', 'de', 'del', 'desde', 'donde', 'durante', 'e', 'el', 'ella',
            'ellas', 'ellos', 'en', 'entre', 'era', 'erais', 'eran', 'eras', 'eres', 'es',
            'esa', 'esas', 'ese', 'eso', 'esos', 'esta', 'estaba', 'estabais', 'estaban',
            'estabas', 'estad', 'estada', 'estadas', 'estado', 'estados', 'estamos', 'estando',
            'estar', 'estaremos', 'estará', 'estarán', 'estarás', 'estaré', 'estaréis',
            'estaría', 'estaríais', 'estaríamos', 'estarían', 'estarías', 'estas', 'este',
            'estemos', 'esto', 'estos', 'estoy', 'estuve', 'estuviera', 'estuvierais',
            'estuvieran', 'estuvieras', 'estuvieron', 'estuviese', 'estuvieseis', 'estuviesen',
            'estuvieses', 'estuvimos', 'estuviste', 'estuvisteis', 'estuviéramos',
            'estuviésemos', 'estuvo', 'está', 'estábamos', 'estáis', 'están', 'estás', 'esté',
            'estéis', 'estén', 'estés', 'fue', 'fuera', 'fuerais', 'fueran', 'fueras', 'fueron',
            'fuese', 'fueseis', 'fuesen', 'fueses', 'fui', 'fuimos', 'fuiste', 'fuisteis',
            'fuéramos', 'fuésemos', 'ha', 'habida', 'habidas', 'habido', 'habidos', 'habiendo',
            'habremos', 'habrá', 'habrán', 'habrás', 'habré', 'habréis', 'habría', 'habríais',
            'habríamos', 'habrían', 'habrías', 'habéis', 'había', 'habíais', 'habíamos',
            'habían', 'habías', 'han', 'has', 'hasta', 'hay', 'haya', 'hayamos', 'hayan',
            'hayas', 'hayáis', 'he', 'hemos', 'hube', 'hubiera', 'hubierais', 'hubieran',
            'hubieras', 'hubieron', 'hubiese', 'hubieseis', 'hubiesen', 'hubieses', 'hubimos',
            'hubiste', 'hubisteis', 'hubiéramos', 'hubiésemos', 'hubo', 'la', 'las', 'le',
            'les', 'lo', 'los', 'me', 'mi', 'mis', 'mucho', 'muchos', 'muy', 'más', 'mí', 'mía',
            'mías', 'mío', 'míos', 'nada', 'ni', 'no', 'nos', 'nosotras', 'nosotros', 'nuestra',
            'nuestras', 'nuestro', 'nuestros', 'o', 'os', 'otra', 'otras', 'otro', 'otros',
            'para', 'pero', 'poco', 'por', 'porque', 'que', 'quien', 'quienes', 'qué', 'se',
            'sea', 'seamos', 'sean', 'seas', 'seremos', 'será', 'serán', 'serás', 'seré',
            'seréis', 'sería', 'seríais', 'seríamos', 'serían', 'serías', 'seáis', 'si', 'sido',
            'siendo', 'sin', 'sobre', 'sois', 'somos', 'son', 'soy', 'su', 'sus', 'suya',
            'suyas', 'suyo', 'suyos', 'sí', 'también', 'tanto', 'te', 'tendremos', 'tendrá',
            'tendrán', 'tendrás', 'tendré', 'tendréis', 'tendría', 'tendríais', 'tendríamos',
            'tendrían', 'tendrías', 'tened', 'tenemos', 'tenga', 'tengamos', 'tengan', 'tengas',
            'tengo', 'tengáis', 'tenida', 'tenidas', 'tenido', 'tenidos', 'teniendo', 'tenéis',
            'tenía', 'teníais', 'teníamos', 'tenían', 'tenías', 'ti', 'tiene', 'tienen',
            'tienes', 'todo', 'todos', 'tu', 'tus', 'tuve', 'tuviera', 'tuvierais', 'tuvieran',
            'tuvieras', 'tuvieron', 'tuviese', 'tuvieseis', 'tuviesen', 'tuvieses', 'tuvimos',
            'tuviste', 'tuvisteis', 'tuviéramos', 'tuviésemos', 'tuvo', 'tuya', 'tuyas', 'tuyo',
            'tuyos', 'tú', 'un', 'una', 'uno', 'unos', 'vosotras', 'vosotros', 'vuestra',
            'vuestras', 'vuestro', 'vuestros', 'y', 'ya', 'yo', 'él', 'éramos'
        }
    
    def _create_vectorizer(self):
        """Crea y configura el vectorizador TF-IDF."""
        try:
            return TfidfVectorizer(
                stop_words=list(self.spanish_stopwords),
                max_features=5000,
                ngram_range=(1, 3),
                min_df=2,
                max_df=0.95,
                norm='l2',
                use_idf=True,
                smooth_idf=True,
                sublinear_tf=True
            )
        except Exception as e:
            self.logger.error(f"Error al crear vectorizador: {e}")
            # Fallback a un vectorizador más simple
            return TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 2)
            )
    
    def _setup_device(self):
        """Configura el dispositivo para PyTorch con manejo de errores."""
        try:
            if torch.cuda.is_available():
                # Verificar memoria disponible en GPU
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                if gpu_memory < 2 * 1024 * 1024 * 1024:  # Menos de 2GB
                    self.logger.warning("GPU detectada pero con memoria insuficiente. Usando CPU.")
                    return torch.device("cpu")
                
                self.logger.info(f"Usando GPU: {torch.cuda.get_device_name(0)}")
                return torch.device("cuda:0")
            else:
                self.logger.info("GPU no disponible. Usando CPU.")
                return torch.device("cpu")
        except Exception as e:
            self.logger.error(f"Error al configurar dispositivo: {e}")
            self.logger.info("Fallback a CPU debido a error.")
            return torch.device("cpu")
    
    def _initialize_models(self):
        """Inicializa los modelos con manejo de errores y optimizaciones."""
        try:
            # Verificar recursos disponibles
            if not SystemMonitor.check_resources(self.memory_threshold_mb):
                self.logger.warning("Recursos limitados. Inicializando modelos simples.")
                self.modelo_contenido = DummyModel()
                self.modelo_codigo = DummyModel()
                return
            
            # Crear modelos
            self.modelo_contenido = EnhancedContenidoEvaluator(input_dim=5000, hidden_dim=768)
            self.modelo_codigo = EnhancedCodigoEvaluator(input_dim=5000, hidden_dim=768)
            
            # Mover modelos al dispositivo de manera segura
            self.modelo_contenido = self.modelo_contenido.to(self.device)
            self.modelo_codigo = self.modelo_codigo.to(self.device)
            
            # Optimizadores
            self.optimizer_contenido = optim.Adam(
                self.modelo_contenido.parameters(), 
                lr=0.001, 
                weight_decay=1e-5,
                betas=(0.9, 0.999)
            )
            self.optimizer_codigo = optim.Adam(
                self.modelo_codigo.parameters(), 
                lr=0.001, 
                weight_decay=1e-5,
                betas=(0.9, 0.999)
            )
            
            # Función de pérdida
            self.criterion = nn.MSELoss()
            
            # Inicializar SBERT si está habilitado
            if self.use_sbert:
                self._initialize_sbert()
            
        except Exception as e:
            self.logger.error(f"Error al inicializar modelos: {e}")
            self.logger.error(traceback.format_exc())
            self.modelo_contenido = DummyModel()
            self.modelo_codigo = DummyModel()
    
    def _initialize_sbert(self):
        """Inicializa el modelo SBERT con manejo de errores y optimizaciones."""
        try:
            # Verificar si hay memoria suficiente
            if SystemMonitor.get_memory_usage() > self.memory_threshold_mb * 0.8:
                self.logger.warning("Memoria insuficiente para SBERT. Desactivando.")
                self.use_sbert = False
                return
            
            # Importar sentence-transformers de manera segura
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                self.logger.warning("No se pudo importar sentence-transformers. Desactivando SBERT.")
                self.use_sbert = False
                return
            
            # Inicializar SBERT con manejo de errores
            try:
                # Usar un modelo más ligero para español
                model_name = 'distiluse-base-multilingual-cased-v1'
                
                # Inicializar en CPU primero
                self.sbert_model = SentenceTransformer(model_name, device="cpu")
                
                # Mover a GPU si está disponible y hay memoria suficiente
                if self.device.type == 'cuda' and SystemMonitor.get_memory_usage() < self.memory_threshold_mb * 0.7:
                    self.sbert_model = self.sbert_model.to(self.device)
                
                self.logger.info(f"Modelo SBERT cargado correctamente en dispositivo: {self.sbert_model.device}")
            except Exception as e:
                self.logger.warning(f"Error al cargar modelo SBERT: {e}")
                self.use_sbert = False
        except Exception as e:
            self.logger.error(f"Error general en inicialización de SBERT: {e}")
            self.use_sbert = False
    
    def _load_knowledge_base(self):
        """Carga la base de conocimiento para generación de prácticas."""
        try:
            # Intentar cargar desde archivo si existe
            kb_path = 'base_conocimiento.json'
            if os.path.exists(kb_path):
                with open(kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Error al cargar base de conocimiento desde archivo: {e}")
        
        # Base de conocimiento predeterminada
        return [
            "Resolver ecuaciones diferenciales de primer orden utilizando métodos numéricos.",
            "Análisis de datos utilizando técnicas estadísticas descriptivas e inferenciales.",
            "Implementación de algoritmos de ordenamiento y búsqueda para estructuras de datos.",
            "Desarrollo de aplicaciones web con frameworks modernos.",
            "Diseño de bases de datos relacionales y consultas SQL optimizadas.",
            "Análisis de complejidad algorítmica y optimización de código.",
            "Implementación de patrones de diseño en programación orientada a objetos.",
            "Desarrollo de APIs RESTful y servicios web.",
            "Análisis de series temporales y modelos predictivos.",
            "Implementación de sistemas de recomendación basados en filtrado colaborativo.",
            "Desarrollo de aplicaciones con arquitectura de microservicios.",
            "Implementación de sistemas de autenticación y autorización seguros.",
            "Análisis de rendimiento y optimización de consultas en bases de datos.",
            "Desarrollo de interfaces de usuario accesibles y responsivas.",
            "Implementación de pruebas unitarias y de integración automatizadas.",
            "Análisis de requisitos y diseño de software orientado a objetos.",
            "Implementación de algoritmos de aprendizaje automático para clasificación y regresión.",
            "Desarrollo de aplicaciones móviles multiplataforma.",
            "Implementación de sistemas de caché y optimización de rendimiento.",
            "Análisis y visualización de datos con bibliotecas especializadas.",
            "Diseño de esquemas de bases de datos normalizados hasta la tercera forma normal.",
            "Implementación de procedimientos almacenados y triggers en MySQL.",
            "Optimización de consultas SQL mediante el uso adecuado de índices.",
            "Diseño e implementación de bases de datos NoSQL para datos no estructurados.",
            "Desarrollo de sistemas de migración y respaldo de bases de datos.",
            "Implementación de técnicas de cifrado para protección de datos sensibles.",
            "Análisis de vulnerabilidades en aplicaciones web mediante pruebas de penetración.",
            "Configuración de firewalls y sistemas de detección de intrusiones.",
            "Desarrollo de políticas de seguridad y planes de recuperación ante desastres.",
            "Implementación de autenticación multifactor en aplicaciones web."
        ]
    
    def _create_tfidf_matrix(self):
        """Crea la matriz TF-IDF para la base de conocimiento."""
        try:
            # Ajustar el vectorizador con la base de conocimiento
            self.vectorizer.fit(self.base_conocimiento)
            # Crear matriz TF-IDF
            self.tfidf_matrix = self.vectorizer.transform(self.base_conocimiento)
            self.logger.info("Matriz TF-IDF creada correctamente")
        except Exception as e:
            self.logger.error(f"Error al crear matriz TF-IDF: {e}")
            # Crear una matriz vacía como fallback
            self.tfidf_matrix = None
    
    def _connect_to_database(self):
        """Establece conexión con la base de datos con manejo de errores."""
        if not self.db_config:
            return
        
        try:
            self.connection = mysql.connector.connect(**self.db_config)
            self.cursor = self.connection.cursor(dictionary=True)
            self.logger.info("Conexión a la base de datos establecida correctamente")
        except Error as e:
            self.logger.error(f"Error al conectar a la base de datos: {e}")
            self.connection = None
            self.cursor = None
    
    def _start_resource_monitoring(self):
        """Inicia un hilo para monitorear recursos del sistema."""
        def monitor_resources():
            while True:
                try:
                    memory_usage = SystemMonitor.get_memory_usage()
                    if memory_usage > self.memory_threshold_mb:
                        self.logger.warning(f"Uso de memoria alto: {memory_usage:.2f} MB. Liberando recursos...")
                        self._free_resources()
                    time.sleep(60)  # Verificar cada minuto
                except Exception as e:
                    self.logger.error(f"Error en monitoreo de recursos: {e}")
                    time.sleep(300)  # Esperar más tiempo en caso de error
        
        # Iniciar hilo de monitoreo
        monitor_thread = threading.Thread(target=monitor_resources, daemon=True)
        monitor_thread.start()
    
    def _free_resources(self):
        """Libera recursos para reducir uso de memoria."""
        try:
            # Liberar caché de SBERT si está cargado
            if self.use_sbert and self.sbert_model:
                self.sbert_model = None
                self.use_sbert = False
                self.logger.info("Modelo SBERT liberado para ahorrar memoria")
            
            # Forzar recolección de basura
            gc.collect()
            
            # Limpiar historial de evaluaciones si es muy grande
            if len(self.historial_evaluaciones) > 100:
                self.historial_evaluaciones = self.historial_evaluaciones[-100:]
                self.logger.info("Historial de evaluaciones reducido para ahorrar memoria")
        except Exception as e:
            self.logger.error(f"Error al liberar recursos: {e}")
    
    def _setup_evaluation_metrics(self):
        """Configura las métricas de evaluación disponibles."""
        return {
            'claridad': self._evaluar_claridad,
            'profundidad': self._evaluar_profundidad,
            'estructura': self._evaluar_estructura,
            'funcionalidad': self._evaluar_funcionalidad_codigo,
            'eficiencia': self._evaluar_eficiencia_codigo,
            'estilo': self._evaluar_estilo_codigo,
            'documentacion': self._evaluar_documentacion_codigo,
            'cumplimiento': self._evaluar_cumplimiento_requisitos,
            'seguridad': self._evaluar_seguridad_codigo,
            'relevancia': self._calcular_relevancia
        }
    
    def cargar_plantillas_retroalimentacion(self):
        """Carga plantillas de retroalimentación más detalladas y variadas"""
        try:
            # Intentar cargar desde archivo si existe
            templates_path = 'plantillas_retroalimentacion.json'
            if os.path.exists(templates_path):
                with open(templates_path, 'r', encoding='utf-8') as f:
                    templates = json.load(f)
                    self.feedback_templates = templates.get('feedback_templates', {})
                    self.code_feedback_templates = templates.get('code_feedback_templates', {})
                    self.error_templates = templates.get('error_templates', {})
                    self.code_error_templates = templates.get('code_error_templates', {})
                    self.strength_templates = templates.get('strength_templates', {})
                    self.code_strength_templates = templates.get('code_strength_templates', {})
                    self.logger.info("Plantillas de retroalimentación cargadas desde archivo")
                    return
        except Exception as e:
            self.logger.warning(f"Error al cargar plantillas desde archivo: {e}")
        
        # Plantillas predeterminadas
        self.feedback_templates = {
            'excellent': [
                "Excelente trabajo. Has demostrado un dominio completo del tema, con un análisis profundo y bien estructurado.",
                "Trabajo sobresaliente. Cumples con todos los requisitos y vas más allá, demostrando comprensión avanzada de los conceptos.",
                "Felicitaciones por un trabajo excepcional. Tu análisis demuestra comprensión profunda y capacidad para conectar conceptos complejos.",
                "Trabajo de calidad superior. Muestras dominio del tema y capacidad para aplicar los conceptos en situaciones diversas.",
                "Excelente desempeño. Tu trabajo refleja un nivel de comprensión y análisis que supera las expectativas del curso."
            ],
            'very_good': [
                "Muy buen trabajo. Demuestras una sólida comprensión del tema con un análisis bien desarrollado.",
                "Trabajo de alta calidad. Has abordado todos los aspectos importantes del tema con claridad y precisión.",
                "Muy buen desempeño. Tu trabajo muestra dominio de los conceptos clave y buena capacidad de análisis.",
                "Trabajo muy completo. Has logrado integrar los conceptos de manera coherente y bien fundamentada.",
                "Muy buen análisis. Tu trabajo demuestra comprensión profunda y capacidad para aplicar los conceptos adecuadamente."
            ],
            'good': [
                "Buen trabajo. Cumples con la mayoría de los requisitos y demuestras comprensión del tema.",
                "Trabajo sólido con varios aspectos destacables. Has abordado los puntos principales del tema.",
                "Has demostrado buena comprensión del tema con algunas áreas para mejorar.",
                "Trabajo bien desarrollado. Muestras dominio de los conceptos básicos y cierta capacidad de análisis.",
                "Buen desempeño general. Tu trabajo cumple con los objetivos principales de la actividad."
            ],
            'average': [
                "Trabajo aceptable. Cumples con los requisitos básicos, pero hay espacio para profundizar.",
                "Has demostrado comprensión básica del tema. Tu trabajo cumple con lo mínimo esperado.",
                "El trabajo cumple con los requisitos esenciales, aunque podría beneficiarse de un análisis más profundo.",
                "Trabajo adecuado. Abordas los conceptos fundamentales, pero podrías desarrollarlos con mayor detalle.",
                "Desempeño satisfactorio. Tu trabajo muestra comprensión de los conceptos básicos del tema."
            ],
            'needs_improvement': [
                "El trabajo necesita mejoras significativas. Revisa cuidadosamente los requisitos de la actividad.",
                "Hay áreas importantes que requieren más atención y desarrollo en tu trabajo.",
                "Es necesario profundizar más en el tema y mejorar la estructura de tu trabajo.",
                "Tu trabajo muestra cierta comprensión del tema, pero necesita mayor desarrollo y claridad.",
                "Se requiere un análisis más detallado y mejor fundamentado para alcanzar los objetivos de la actividad."
            ],
            'poor': [
                "El trabajo no cumple con los requisitos mínimos. Es necesario revisar los conceptos fundamentales.",
                "Es necesario revisar completamente el trabajo y asegurarte de abordar los aspectos clave del tema.",
                "Recomiendo volver a estudiar los conceptos básicos del tema y rehacer el trabajo.",
                "Tu trabajo muestra dificultades importantes en la comprensión de los conceptos fundamentales.",
                "Es necesario un replanteamiento completo del trabajo para cumplir con los objetivos de la actividad."
            ],
            'irrelevant': [
                "El trabajo entregado no tiene relación con el tema solicitado. Es fundamental abordar el tema específico de la actividad.",
                "Tu entrega no aborda el tema requerido. Revisa cuidadosamente el título y objetivo de la actividad.",
                "El contenido no corresponde con lo solicitado. Es necesario enfocarse específicamente en el tema de la actividad.",
                "Tu trabajo se desvía completamente del tema solicitado. Revisa las instrucciones y objetivos de la actividad.",
                "La entrega no aborda el tema requerido. Es esencial centrarse en el objetivo específico de la actividad."
            ]
        }
        
        # Plantillas específicas para código
        self.code_feedback_templates = {
            'excellent': [
                "Excelente implementación. Tu código es eficiente, bien estructurado y sigue las mejores prácticas de programación.",
                "Código sobresaliente. Demuestra un dominio completo de los conceptos de programación y una implementación óptima.",
                "Implementación excepcional. Tu código es claro, eficiente y resuelve el problema de manera elegante.",
                "Código de alta calidad. Muestra un excelente manejo de las estructuras de datos y algoritmos apropiados.",
                "Implementación superior. Tu código es robusto, eficiente y fácil de mantener."
            ],
            'very_good': [
                "Muy buen código. Implementación eficiente y bien estructurada con buena documentación.",
                "Implementación de alta calidad. Tu código resuelve correctamente el problema con un buen diseño.",
                "Muy buen trabajo. Tu código es claro, funcional y sigue buenas prácticas de programación.",
                "Código bien implementado. Muestra buen dominio de los conceptos y técnicas de programación.",
                "Implementación sólida. Tu código es eficiente y está bien organizado."
            ],
            'good': [
                "Buen código. Funciona correctamente y tiene una estructura adecuada.",
                "Implementación correcta. Tu código resuelve el problema aunque podría optimizarse en algunos aspectos.",
                "Buen trabajo. Tu código funciona según lo esperado con algunos aspectos a mejorar.",
                "Código funcional. Implementa la solución requerida con una estructura aceptable.",
                "Implementación adecuada. Tu código cumple con los requisitos básicos del problema."
            ],
            'average': [
                "Código aceptable. Funciona pero podría mejorar en eficiencia y estructura.",
                "Implementación básica. Tu código resuelve el problema pero necesita optimización.",
                "Código funcional con limitaciones. Cumple con lo mínimo pero requiere mejoras importantes.",
                "Implementación sencilla. Resuelve el problema de forma básica sin optimizaciones.",
                "Código adecuado. Funciona para casos simples pero podría fallar en escenarios complejos."
            ],
            'needs_improvement': [
                "El código necesita mejoras significativas. Revisa la lógica y estructura de tu implementación.",
                "Hay problemas importantes en tu código. Es necesario corregir errores y mejorar la eficiencia.",
                "Tu implementación requiere revisión. El código tiene errores o no sigue buenas prácticas.",
                "Es necesario mejorar la calidad del código. Revisa la lógica, estructura y documentación.",
                "Tu código necesita trabajo adicional. Hay problemas de funcionalidad y diseño que deben corregirse."
            ],
            'poor': [
                "El código no cumple con los requisitos mínimos. Es necesario replantearlo completamente.",
                "Implementación incorrecta. Tu código tiene errores graves que impiden su funcionamiento.",
                "Es necesario revisar los conceptos fundamentales de programación y rehacer el código.",
                "Tu implementación muestra dificultades importantes en la comprensión de los conceptos básicos.",
                "El código no funciona correctamente. Es necesario un replanteamiento completo de la solución."
            ]
        }
        
        # Plantillas para errores y fortalezas (simplificadas para el ejemplo)
        self.error_templates = {}
        self.code_error_templates = {}
        self.strength_templates = {}
        self.code_strength_templates = {}
        
        # Guardar plantillas en archivo para uso futuro
        try:
            templates = {
                'feedback_templates': self.feedback_templates,
                'code_feedback_templates': self.code_feedback_templates,
                'error_templates': self.error_templates,
                'code_error_templates': self.code_error_templates,
                'strength_templates': self.strength_templates,
                'code_strength_templates': self.code_strength_templates
            }
            
            with open(templates_path, 'w', encoding='utf-8') as f:
                json.dump(templates, f, ensure_ascii=False, indent=2)
                self.logger.info("Plantillas de retroalimentación guardadas en archivo")
        except Exception as e:
            self.logger.warning(f"Error al guardar plantillas en archivo: {e}")

    def cargar_recursos_educativos(self):
        """Carga recursos educativos categorizados por estilo de aprendizaje"""
        try:
            # Intentar cargar desde archivo si existe
            resources_path = 'recursos_educativos.json'
            if os.path.exists(resources_path):
                with open(resources_path, 'r', encoding='utf-8') as f:
                    self.recursos_educativos = json.load(f)
                    self.logger.info("Recursos educativos cargados desde archivo")
                    return
        except Exception as e:
            self.logger.warning(f"Error al cargar recursos educativos desde archivo: {e}")
        
        # Recursos predeterminados (simplificados para el ejemplo)
        self.recursos_educativos = {
            'visual': {
                'programacion': [
                    {"titulo": "Visualización de algoritmos", "url": "https://visualgo.net/en"},
                    {"titulo": "Diagramas de flujo interactivos", "url": "https://app.diagrams.net/"}
                ],
                'matematicas': [
                    {"titulo": "Visualizaciones de conceptos matemáticos", "url": "https://www.geogebra.org/"},
                    {"titulo": "Videos de explicaciones visuales", "url": "https://www.3blue1brown.com/"}
                ]
            },
            'auditivo': {
                'programacion': [
                    {"titulo": "Podcast sobre desarrollo de software", "url": "https://www.codingblocks.net/"},
                    {"titulo": "Conferencias grabadas sobre programación", "url": "https://www.youtube.com/c/GOTO-"}
                ],
                'matematicas': [
                    {"titulo": "Podcast de matemáticas", "url": "https://www.bbc.co.uk/programmes/p01gyd7j/episodes/downloads"},
                    {"titulo": "Explicaciones verbales de conceptos", "url": "https://www.khanacademy.org/math"}
                ]
            }
        }
        
        # Guardar recursos en archivo para uso futuro
        try:
            with open(resources_path, 'w', encoding='utf-8') as f:
                json.dump(self.recursos_educativos, f, ensure_ascii=False, indent=2)
                self.logger.info("Recursos educativos guardados en archivo")
        except Exception as e:
            self.logger.warning(f"Error al guardar recursos en archivo: {e}")

    def detectar_tipo_contenido(self, contenido: str) -> str:
        """
        Detecta automáticamente el tipo de contenido para aplicar el evaluador adecuado.
        
        Args:
            contenido: Texto a analizar
            
        Returns:
            Tipo de contenido: 'codigo', 'sql', 'texto', 'matematico', etc.
        """
        try:
            # Indicadores de código SQL
            indicadores_sql = [
                'SELECT ', 'FROM ', 'WHERE ', 'INSERT INTO', 'UPDATE ', 'DELETE FROM', 'CREATE TABLE',
                'ALTER TABLE', 'DROP TABLE', 'JOIN ', 'GROUP BY', 'ORDER BY', 'HAVING ', 'UNION ',
                'CREATE DATABASE', 'USE ', 'PRIMARY KEY', 'FOREIGN KEY', 'REFERENCES ', 'INDEX',
                'CONSTRAINT', 'TRIGGER', 'PROCEDURE', 'FUNCTION', 'VIEW', 'GRANT ', 'REVOKE '
            ]
            
            # Indicadores de código general
            indicadores_codigo = [
                'def ', 'class ', 'function', 'import ', 'from ', 'return ', 'if ', 'else:', 'for ', 'while ',
                '{', '}', ';', 'public ', 'private ', 'protected ', 'static ', 'void ', 'int ', 'float ',
                'string ', 'bool ', 'var ', 'const ', 'let ', 'console.log', 'print(', 'System.out', 'cout <<'
            ]
            
            # Indicadores de contenido matemático
            indicadores_matematico = [
                '\\begin{equation}', '\\end{equation}', '\\frac', '\\sum', '\\int', '\\prod', '\\lim',
                '\\alpha', '\\beta', '\\gamma', '\\delta', '\\theta', '\\lambda', '\\pi', '\\sigma',
                '\\sqrt', '\\partial', '\\nabla', '\\infty', '\\approx', '\\neq', '\\geq', '\\leq',
                '\\in', '\\subset', '\\cup', '\\cap', '\\emptyset', '\\mathbb', '\\mathcal', '\\mathrm'
            ]
            
            # Contar indicadores
            contador_sql = sum(1 for ind in indicadores_sql if ind in contenido)
            contador_codigo = sum(1 for ind in indicadores_codigo if ind in contenido)
            contador_matematico = sum(1 for ind in indicadores_matematico if ind in contenido)
            
            # Calcular porcentajes (normalizado por la cantidad de indicadores)
            porcentaje_sql = contador_sql / len(indicadores_sql)
            porcentaje_codigo = contador_codigo / len(indicadores_codigo)
            porcentaje_matematico = contador_matematico / len(indicadores_matematico)
            
            # Determinar tipo de contenido
            if porcentaje_sql > 0.1:  # Si más del 10% de los indicadores de SQL están presentes
                return 'sql'
            elif porcentaje_codigo > 0.1:  # Si más del 10% de los indicadores de código están presentes
                return 'codigo'
            elif porcentaje_matematico > 0.1:  # Si más del 10% de los indicadores matemáticos están presentes
                return 'matematico'
            else:
                return 'texto'
        except Exception as e:
            self.logger.error(f"Error al detectar tipo de contenido: {e}")
            return 'texto'  # Valor por defecto en caso de error

    def analizar_contenido(self, contenido_archivo, titulo_actividad="", objetivo_actividad="", estilo_aprendizaje=None):
        """
        Analiza el contenido del archivo y devuelve una calificación y comentarios detallados.
        
        Args:
            contenido_archivo: Texto del contenido entregado
            titulo_actividad: Título de la actividad
            objetivo_actividad: Objetivo de la actividad
            estilo_aprendizaje: Estilo(s) de aprendizaje del estudiante (separados por comas)
            
        Returns:
            Diccionario con calificación, comentarios, sugerencias, relevancia y recursos recomendados
        """
        try:
            # Incrementar contador de evaluaciones
            self.contador_evaluaciones += 1
            self.logger.info(f"Iniciando análisis de contenido #{self.contador_evaluaciones}")
            
            # Validar que el contenido no esté vacío
            if not contenido_archivo or not contenido_archivo.strip():
                self.logger.warning("Contenido vacío detectado")
                return {
                    "calificacion": 0.0,
                    "comentarios": "El archivo está vacío o no contiene texto válido.",
                    "sugerencias": "Es necesario entregar un trabajo con contenido para poder evaluarlo.",
                    "relevancia": 0.0,
                    "recursos_recomendados": []
                }
            
            # Detectar tipo de contenido
            tipo_contenido = self.detectar_tipo_contenido(contenido_archivo)
            self.logger.info(f"Tipo de contenido detectado: {tipo_contenido}")
            
            # Extraer palabras clave del título y objetivo para entender el contexto
            contexto = self._extraer_contexto(titulo_actividad, objetivo_actividad)
            
            # Preprocesar el contenido
            contenido_procesado = self._preprocess_text(contenido_archivo)
            self.logger.info(f"Contenido preprocesado: {len(contenido_procesado)} caracteres")
            
            # Análisis de relevancia
            relevancia = self._calcular_relevancia(contenido_procesado, titulo_actividad, objetivo_actividad)
            self.logger.info(f"Relevancia calculada: {relevancia:.4f}")
            
            # Si la relevancia es extremadamente baja, asignar calificación cero
            if relevancia < 0.2:
                self.logger.warning(f"Contenido irrelevante detectado (relevancia: {relevancia:.4f})")
                return self._generar_respuesta_irrelevante(estilo_aprendizaje, titulo_actividad, objetivo_actividad)
            
            # Análisis multidimensional según tipo de contenido y contexto
            if tipo_contenido in ['codigo', 'sql']:
                metricas = self._analizar_metricas_codigo(contenido_procesado, titulo_actividad, objetivo_actividad, contexto)
            else:
                metricas = self._analizar_metricas(contenido_procesado, titulo_actividad, objetivo_actividad, contexto)
            
            self.logger.info(f"Métricas calculadas: {metricas}")
            
            # Vectorizar el contenido para el modelo de evaluación
            try:
                contenido_vectorizado = self.vectorizer.transform([contenido_procesado]).toarray()[0]
                
                # Asegurar dimensiones consistentes
                if len(contenido_vectorizado) < 5000:
                    contenido_vectorizado = np.pad(contenido_vectorizado, (0, 5000 - len(contenido_vectorizado)), 'constant')
                elif len(contenido_vectorizado) > 5000:
                    contenido_vectorizado = contenido_vectorizado[:5000]
                    
                # Convertir a tensor
                contenido_tensor = torch.tensor(contenido_vectorizado, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                # Generar predicción base según tipo de contenido y contexto
                if tipo_contenido in ['codigo', 'sql']:
                    self.modelo_codigo.eval()
                    with torch.no_grad():
                        outputs = self.modelo_codigo(contenido_tensor)
                        pred_base = outputs['global'].cpu().numpy()[0][0]
                        self.logger.info(f"Predicción base del modelo de código: {pred_base}")
                else:
                    self.modelo_contenido.eval()
                    with torch.no_grad():
                        outputs = self.modelo_contenido(contenido_tensor)
                        pred_base = outputs['global'].cpu().numpy()[0][0]
                        self.logger.info(f"Predicción base del modelo de contenido: {pred_base}")
            except Exception as e:
                self.logger.error(f"Error en la vectorización o predicción: {e}")
                # Usar un enfoque alternativo basado en métricas si falla la predicción
                pred_base = self._calcular_calificacion_alternativa(metricas, contexto)
                self.logger.info(f"Usando calificación alternativa: {pred_base}")
            
            # Ajustar calificación según métricas, relevancia y contexto
            if tipo_contenido in ['codigo', 'sql']:
                calificacion_ajustada = self._ajustar_calificacion_codigo(
                    pred_base, 
                    relevancia, 
                    metricas['funcionalidad'], 
                    metricas['eficiencia'], 
                    metricas['estilo'],
                    metricas['documentacion'],
                    metricas['cumplimiento'],
                    contexto,
                    tipo_contenido
                )
            else:
                calificacion_ajustada = self._ajustar_calificacion(
                    pred_base, 
                    relevancia, 
                    metricas['claridad'], 
                    metricas['profundidad'], 
                    metricas['estructura'],
                    contexto
                )
            
            # Limitar a rango 0-10
            calificacion_final = min(10.0, max(0.0, calificacion_ajustada))
            self.logger.info(f"Calificación final: {calificacion_final:.2f}")
            
            # Identificar fortalezas y debilidades según tipo de contenido y contexto
            if tipo_contenido in ['codigo', 'sql']:
                fortalezas, debilidades = self._identificar_fortalezas_debilidades_codigo(
                    contenido_procesado, 
                    metricas, 
                    calificacion_final,
                    tipo_contenido,
                    contexto
                )
            else:
                fortalezas, debilidades = self._identificar_fortalezas_debilidades(
                    contenido_procesado, 
                    metricas, 
                    calificacion_final,
                    contexto
                )
            
            self.logger.info(f"Fortalezas identificadas: {len(fortalezas)}")
            self.logger.info(f"Debilidades identificadas: {len(debilidades)}")
            
            # Generar comentarios detallados según tipo de contenido y contexto
            if tipo_contenido in ['codigo', 'sql']:
                comentarios = self._generar_comentarios_detallados_codigo(
                    calificacion_final, 
                    contenido_procesado, 
                    titulo_actividad, 
                    objetivo_actividad, 
                    fortalezas, 
                    debilidades,
                    estilo_aprendizaje,
                    tipo_contenido,
                    contexto
                )
            else:
                comentarios = self._generar_comentarios_detallados(
                    calificacion_final, 
                    contenido_procesado, 
                    titulo_actividad, 
                    objetivo_actividad, 
                    fortalezas, 
                    debilidades,
                    estilo_aprendizaje,
                    contexto
                )
            
            # Generar sugerencias de mejora según tipo de contenido y contexto
            if tipo_contenido in ['codigo', 'sql']:
                sugerencias = self._generar_sugerencias_mejora_codigo(
                    calificacion_final, 
                    contenido_procesado, 
                    titulo_actividad, 
                    objetivo_actividad, 
                    debilidades,
                    estilo_aprendizaje,
                    tipo_contenido,
                    contexto
                )
            else:
                sugerencias = self._generar_sugerencias_mejora(
                    calificacion_final, 
                    contenido_procesado, 
                    titulo_actividad, 
                    objetivo_actividad, 
                    debilidades,
                    estilo_aprendizaje,
                    contexto
                )
            
            # Recomendar recursos educativos según estilo de aprendizaje y contexto
            recursos_recomendados = self._recomendar_recursos(
                estilo_aprendizaje,
                titulo_actividad,
                objetivo_actividad,
                calificacion_final,
                tipo_contenido,
            )
            
            # Guardar esta evaluación en el historial para aprendizaje continuo
            self._registrar_evaluacion(
                contenido_procesado,
                tipo_contenido,
                titulo_actividad,
                objetivo_actividad,
                calificacion_final,
                metricas,
                contexto
            )
            
            # Construir respuesta completa
            respuesta = {
                "calificacion": float(round(calificacion_final, 1)),
                "comentarios": comentarios,
                "sugerencias": sugerencias,
                "relevancia": float(round(relevancia * 100, 1)),
                "tipo_contenido": tipo_contenido,
                "metricas": self._formatear_metricas(metricas, tipo_contenido),
                "fortalezas": fortalezas[:3],
                "debilidades": debilidades[:3],
                "recursos_recomendados": recursos_recomendados
            }
            
            self.logger.info(f"Análisis de contenido #{self.contador_evaluaciones} completado")
            return respuesta
        except Exception as e:
            self.logger.error(f"Error en analizar_contenido: {e}")
            self.logger.error(traceback.format_exc())
            
            # Respuesta de fallback en caso de error
            return {
                "calificacion": 5.0,
                "comentarios": "No se pudo completar el análisis detallado debido a un error técnico.",
                "sugerencias": "Por favor, contacta al soporte técnico si este problema persiste.",
                "relevancia": 50.0,
                "tipo_contenido": "desconocido",
                "metricas": {},
                "fortalezas": [],
                "debilidades": [],
                "recursos_recomendados": []
            }

    # Métodos auxiliares necesarios
    def _extraer_contexto(self, titulo, objetivo):
        """Extrae información contextual del título y objetivo de la actividad."""
        contexto = {
            'tipo_tarea': 'general',
            'requisitos_especificos': [],
            'nivel_complejidad': 'medio',
            'enfoque_principal': 'general'
        }
        
        if not titulo and not objetivo:
            return contexto
            
        texto_combinado = f"{titulo} {objetivo}".lower()
        
        # Detectar tipo de tarea
        if any(palabra in texto_combinado for palabra in ['crear', 'diseñar', 'implementar', 'desarrollar', 'construir']):
            contexto['tipo_tarea'] = 'creacion'
        elif any(palabra in texto_combinado for palabra in ['analizar', 'evaluar', 'comparar', 'criticar']):
            contexto['tipo_tarea'] = 'analisis'
        elif any(palabra in texto_combinado for palabra in ['explicar', 'describir', 'resumir', 'sintetizar']):
            contexto['tipo_tarea'] = 'explicacion'
        elif any(palabra in texto_combinado for palabra in ['investigar', 'explorar', 'indagar']):
            contexto['tipo_tarea'] = 'investigacion'
        
        # Detectar enfoque principal
        if 'base de datos' in texto_combinado or 'sql' in texto_combinado:
            contexto['enfoque_principal'] = 'base_de_datos'
        elif 'código' in texto_combinado or 'programa' in texto_combinado or 'algoritmo' in texto_combinado:
            contexto['enfoque_principal'] = 'programacion'
        elif 'ensayo' in texto_combinado or 'redacción' in texto_combinado or 'escrito' in texto_combinado:
            contexto['enfoque_principal'] = 'redaccion'
            
        # Detectar nivel de complejidad
        if any(palabra in texto_combinado for palabra in ['básico', 'simple', 'sencillo', 'introductorio']):
            contexto['nivel_complejidad'] = 'bajo'
        elif any(palabra in texto_combinado for palabra in ['avanzado', 'complejo', 'difícil', 'profundo']):
            contexto['nivel_complejidad'] = 'alto'
            
        return contexto

    def _preprocess_text(self, text):
        """Preprocesa el texto para análisis."""
        if not text:
            return ""
            
        try:
            # Convertir a minúsculas
            text = text.lower()
            
            # Eliminar caracteres especiales pero mantener puntuación importante
            text = re.sub(r'[^\w\s.,;:¿?¡!()"-]', '', text)
            
            # Eliminar espacios extra
            text = re.sub(r'\s+', ' ', text).strip()
            
            return text
        except Exception as e:
            self.logger.error(f"Error al preprocesar texto: {e}")
            return text

    def _calcular_relevancia(self, contenido, titulo, objetivo):
        """Calcula la relevancia semántica entre el contenido y los requisitos."""
        try:
            if not titulo and not objetivo:
                return 1.0
                
            # Combinar título y objetivo
            requisitos = f"{titulo} {objetivo}"
            
            # Usar modelo SBERT si está disponible
            if self.use_sbert and self.sbert_model is not None:
                try:
                    # Obtener embeddings
                    embedding_requisitos = self.sbert_model.encode([requisitos])[0]
                    embedding_contenido = self.sbert_model.encode([contenido[:512]])[0]
                    
                    # Calcular similitud del coseno
                    similarity = cosine_similarity(
                        [embedding_requisitos], 
                        [embedding_contenido]
                    )[0][0]
                    
                    return max(0.0, min(1.0, similarity))
                except Exception as e:
                    self.logger.warning(f"Error al calcular relevancia con SBERT: {e}")
            
            # Método TF-IDF (fallback)
            try:
                # Vectorizar ambos textos
                vec_requisitos = self.vectorizer.transform([requisitos])
                vec_contenido = self.vectorizer.transform([contenido])
                
                # Calcular similitud del coseno
                similarity = cosine_similarity(vec_requisitos, vec_contenido)[0][0]
                
                return max(0.0, min(1.0, similarity))
            except Exception as e:
                self.logger.error(f"Error al calcular relevancia con TF-IDF: {e}")
                
                # Método de respaldo simple basado en palabras clave
                palabras_clave = set(requisitos.split())
                palabras_contenido = set(contenido.split())
                palabras_comunes = palabras_clave.intersection(palabras_contenido)
                
                if len(palabras_clave) > 0:
                    return len(palabras_comunes) / len(palabras_clave)
                return 0.5
        except Exception as e:
            self.logger.error(f"Error general al calcular relevancia: {e}")
            return 0.5

    def _analizar_metricas(self, contenido, titulo, objetivo, contexto):
        """Analiza métricas para contenido textual."""
        metricas = {}
        
        # Calcular claridad
        metricas['claridad'] = self._evaluar_claridad(contenido)
        
        # Calcular profundidad
        metricas['profundidad'] = self._evaluar_profundidad(contenido, titulo, objetivo)
        
        # Calcular estructura
        metricas['estructura'] = self._evaluar_estructura(contenido)
        
        # Calcular relevancia
        metricas['relevancia'] = self._calcular_relevancia(contenido, titulo, objetivo)
        
        return metricas

    def _analizar_metricas_codigo(self, contenido, titulo, objetivo, contexto):
        """Analiza métricas para código."""
        metricas = {}
        
        # Calcular funcionalidad
        metricas['funcionalidad'] = self._evaluar_funcionalidad_codigo(contenido, titulo, objetivo)
        
        # Calcular eficiencia
        metricas['eficiencia'] = self._evaluar_eficiencia_codigo(contenido)
        
        # Calcular estilo
        metricas['estilo'] = self._evaluar_estilo_codigo(contenido)
        
        # Calcular documentación
        metricas['documentacion'] = self._evaluar_documentacion_codigo(contenido)
        
        # Calcular cumplimiento de requisitos
        metricas['cumplimiento'] = self._evaluar_cumplimiento_requisitos(contenido, titulo, objetivo)
        
        # Calcular seguridad
        metricas['seguridad'] = self._evaluar_seguridad_codigo(contenido)
        
        # Calcular relevancia
        metricas['relevancia'] = self._calcular_relevancia(contenido, titulo, objetivo)
        
        return metricas

    def _evaluar_claridad(self, contenido):
        """Evalúa la claridad del contenido."""
        try:
            # Indicadores de claridad
            indicadores_positivos = [
                'por lo tanto', 'en consecuencia', 'es decir', 'en otras palabras',
                'por ejemplo', 'como se muestra', 'para ilustrar', 'específicamente',
                'en particular', 'en resumen', 'en conclusión', 'finalmente'
            ]
            
            indicadores_negativos = [
                'de alguna manera', 'quizás', 'tal vez', 'posiblemente',
                'se podría decir', 'en cierto modo', 'más o menos'
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in contenido.lower())
            negativos = sum(1 for ind in indicadores_negativos if ind in contenido.lower())
            
            # Calcular longitud promedio de oraciones
            oraciones = re.split(r'[.!?]+', contenido)
            oraciones = [o.strip() for o in oraciones if o.strip()]
            
            if not oraciones:
                return 5.0
                
            longitud_promedio = sum(len(o.split()) for o in oraciones) / len(oraciones)
            
            # Penalizar oraciones muy largas o muy cortas
            factor_longitud = 1.0
            if longitud_promedio > 30:
                factor_longitud = 0.8
            elif longitud_promedio < 5:
                factor_longitud = 0.9
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(5.0, positivos * 0.5) - min(3.0, negativos * 0.5)
            
            # Ajustar por longitud de oraciones
            puntuacion_ajustada = puntuacion_base * factor_longitud
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_ajustada))
        except Exception as e:
            self.logger.error(f"Error al evaluar claridad: {e}")
            return 5.0

    def _evaluar_profundidad(self, contenido, titulo, objetivo):
        """Evalúa la profundidad del contenido."""
        try:
            # Indicadores de profundidad
            indicadores_positivos = [
                'análisis', 'evaluación', 'comparación', 'crítica', 'interpretación',
                'perspectiva', 'enfoque', 'metodología', 'teoría', 'concepto',
                'fundamento', 'principio', 'paradigma', 'implicación', 'consecuencia'
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in contenido.lower())
            
            # Calcular diversidad de vocabulario
            palabras = re.findall(r'\b\w+\b', contenido.lower())
            palabras_unicas = set(palabras)
            
            if not palabras:
                return 5.0
                
            diversidad = len(palabras_unicas) / len(palabras)
            
            # Calcular longitud del contenido (normalizada)
            longitud_normalizada = min(1.0, len(contenido) / 5000)
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(3.0, positivos * 0.3) + diversidad * 2.0 + longitud_normalizada * 2.0
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_base))
        except Exception as e:
            self.logger.error(f"Error al evaluar profundidad: {e}")
            return 5.0

    def _evaluar_estructura(self, contenido):
        """Evalúa la estructura del contenido."""
        try:
            # Indicadores de estructura
            indicadores_positivos = [
                'introducción', 'desarrollo', 'conclusión', 'en primer lugar',
                'en segundo lugar', 'finalmente', 'por un lado', 'por otro lado',
                'sin embargo', 'no obstante', 'además', 'asimismo', 'en resumen'
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in contenido.lower())
            
            # Detectar párrafos
            parrafos = contenido.split('\n\n')
            parrafos = [p.strip() for p in parrafos if p.strip()]
            
            if not parrafos:
                return 5.0
                
            # Evaluar longitud de párrafos
            longitudes = [len(p) for p in parrafos]
            longitud_promedio = sum(longitudes) / len(longitudes)
            desviacion = sum(abs(l - longitud_promedio) for l in longitudes) / len(longitudes)
            
            # Penalizar desviación alta (párrafos muy desiguales)
            factor_desviacion = 1.0
            if desviacion > longitud_promedio * 0.5:
                factor_desviacion = 0.9
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(3.0, positivos * 0.3) + min(2.0, len(parrafos) * 0.2)
            
            # Ajustar por desviación
            puntuacion_ajustada = puntuacion_base * factor_desviacion
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_ajustada))
        except Exception as e:
            self.logger.error(f"Error al evaluar estructura: {e}")
            return 5.0

    def _evaluar_funcionalidad_codigo(self, codigo, titulo, objetivo):
        """Evalúa la funcionalidad del código."""
        try:
            # Indicadores de funcionalidad
            indicadores_positivos = [
                'return', 'print', 'output', 'resultado', 'función', 'método',
                'class', 'def', 'function', 'if', 'else', 'for', 'while',
                'try', 'except', 'finally', 'with', 'import', 'from'
            ]
            
            indicadores_negativos = [
                'error', 'exception', 'fail', 'bug', 'issue', 'problem',
                'warning', 'deprecated', 'todo', 'fixme'
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in codigo.lower())
            negativos = sum(1 for ind in indicadores_negativos if ind in codigo.lower())
            
            # Detectar patrones de código incompleto
            codigo_incompleto = re.search(r'(#|//)\s*(todo|fixme|xxx|pendiente|incompleto)', codigo.lower())
            
            # Penalizar código incompleto
            factor_completitud = 1.0
            if codigo_incompleto:
                factor_completitud = 0.8
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(5.0, positivos * 0.2) - min(3.0, negativos * 0.5)
            
            # Ajustar por completitud
            puntuacion_ajustada = puntuacion_base * factor_completitud
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_ajustada))
        except Exception as e:
            self.logger.error(f"Error al evaluar funcionalidad del código: {e}")
            return 5.0

    def _evaluar_eficiencia_codigo(self, codigo):
        """Evalúa la eficiencia del código."""
        try:
            # Indicadores de eficiencia
            indicadores_positivos = [
                'optimiz', 'eficien', 'rendimiento', 'performance', 'complejidad',
                'O(1)', 'O(log n)', 'O(n)', 'memoiz', 'cache', 'buffer'
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in codigo.lower())
            
            # Buscar patrones de código ineficiente
            bucles_anidados = len(re.findall(r'for.*for|while.*while', codigo.lower()))
            
            # Penalizar bucles anidados
            factor_bucles = 1.0
            if bucles_anidados > 0:
                factor_bucles = 1.0 - min(0.3, bucles_anidados * 0.1)
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(5.0, positivos * 0.5)
            
            # Ajustar por bucles anidados
            puntuacion_ajustada = puntuacion_base * factor_bucles
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_ajustada))
        except Exception as e:
            self.logger.error(f"Error al evaluar eficiencia del código: {e}")
            return 5.0

    def _evaluar_estilo_codigo(self, codigo):
        """Evalúa el estilo del código."""
        try:
            # Indicadores de buen estilo
            indicadores_positivos = [
                'def ', 'class ', 'function ', 'import ', 'from ', 'return ',
                'if ', 'else:', 'elif ', 'for ', 'while ', 'try:', 'except:',
                '# ', '"""', "'''"
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in codigo)
            
            # Evaluar indentación
            lineas = codigo.split('\n')
            lineas = [l for l in lineas if l.strip()]
            
            if not lineas:
                return 5.0
                
            # Verificar consistencia de indentación
            indentaciones = [len(l) - len(l.lstrip()) for l in lineas]
            indentacion_consistente = len(set(i % 4 for i in indentaciones)) <= 1
            
            # Verificar longitud de líneas
            lineas_largas = sum(1 for l in lineas if len(l) > 100)
            factor_longitud = 1.0 - min(0.2, lineas_largas / len(lineas))
            
            # Verificar comentarios
            comentarios = sum(1 for l in lineas if l.strip().startswith(('#', '//', '/*', '*', '*/')))
            ratio_comentarios = comentarios / len(lineas)
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(2.0, positivos * 0.1)
            
            # Ajustar por indentación
            if indentacion_consistente:
                puntuacion_base += 1.0
            
            # Ajustar por longitud de líneas
            puntuacion_base *= factor_longitud
            
            # Ajustar por comentarios
            if ratio_comentarios < 0.05:
                puntuacion_base *= 0.9
            elif ratio_comentarios > 0.4:
                puntuacion_base *= 0.95
            else:
                puntuacion_base += 1.0
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_base))
        except Exception as e:
            self.logger.error(f"Error al evaluar estilo del código: {e}")
            return 5.0

    def _evaluar_documentacion_codigo(self, codigo):
        """Evalúa la documentación del código."""
        try:
            # Contar comentarios
            lineas = codigo.split('\n')
            lineas = [l for l in lineas if l.strip()]
            
            if not lineas:
                return 5.0
                
            # Comentarios de una línea
            comentarios_linea = sum(1 for l in lineas if l.strip().startswith(('#', '//')))
            
            # Docstrings (Python)
            docstrings = len(re.findall(r'""".*?"""|\'\'\'.*?\'\'\'', codigo, re.DOTALL))
            
            # Comentarios de bloque (otros lenguajes)
            comentarios_bloque = len(re.findall(r'/\*.*?\*/', codigo, re.DOTALL))
            
            # Total de comentarios
            total_comentarios = comentarios_linea + docstrings * 3 + comentarios_bloque * 3
            
            # Ratio de comentarios por línea de código
            ratio_comentarios = total_comentarios / len(lineas)
            
            # Detectar patrones de buena documentación
            patrones_buenos = [
                r'@param', r'@return', r'@throws', r':param', r':return', r':raises',
                r'Parameters:', r'Returns:', r'Raises:', r'Example:', r'Usage:'
            ]
            
            buenos_patrones = sum(1 for p in patrones_buenos if re.search(p, codigo))
            
            # Calcular puntuación base
            puntuacion_base = 5.0
            
            # Ajustar por ratio de comentarios
            if ratio_comentarios < 0.05:
                puntuacion_base *= 0.7
            elif ratio_comentarios < 0.1:
                puntuacion_base *= 0.9
            elif ratio_comentarios > 0.5:
                puntuacion_base *= 0.95
            else:
                puntuacion_base += 2.0
            
            # Ajustar por patrones de buena documentación
            puntuacion_base += min(3.0, buenos_patrones * 0.5)
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_base))
        except Exception as e:
            self.logger.error(f"Error al evaluar documentación del código: {e}")
            return 5.0

    def _evaluar_cumplimiento_requisitos(self, contenido, titulo, objetivo):
        """Evalúa el cumplimiento de requisitos específicos."""
        try:
            if not titulo and not objetivo:
                return 7.5
                
            # Combinar título y objetivo
            requisitos = f"{titulo} {objetivo}".lower()
            
            # Extraer palabras clave de los requisitos
            palabras_clave = set(re.findall(r'\b\w+\b', requisitos))
            palabras_clave = {p for p in palabras_clave if len(p) > 3}
            
            # Extraer palabras del contenido
            palabras_contenido = set(re.findall(r'\b\w+\b', contenido.lower()))
            
            # Calcular coincidencias
            coincidencias = palabras_clave.intersection(palabras_contenido)
            
            if not palabras_clave:
                return 7.5
                
            # Calcular ratio de coincidencia
            ratio = len(coincidencias) / len(palabras_clave)
            
            # Calcular puntuación base
            puntuacion_base = ratio * 10.0
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_base))
        except Exception as e:
            self.logger.error(f"Error al evaluar cumplimiento de requisitos: {e}")
            return 5.0

    def _evaluar_seguridad_codigo(self, codigo):
        """Evalúa la seguridad del código."""
        try:
            # Indicadores de problemas de seguridad
            indicadores_inseguridad = [
                'eval(', 'exec(', 'os.system(', 'subprocess.call(',
                'input(', 'pickle.load', 'yaml.load(', 'marshal.loads(',
                '__import__', 'getattr(', 'setattr(', 'globals()[',
                'locals()[', 'open(', 'file(', 'execfile(',
                'sql injection', 'xss', 'csrf', 'cross-site'
            ]
            
            # Indicadores de buenas prácticas de seguridad
            indicadores_seguridad = [
                'sanitize', 'escape', 'validate', 'prepared statement',
                'parameterized', 'whitelist', 'csrf_token', 'hmac',
                'hash', 'encrypt', 'ssl', 'tls',
                'permission', 'authorization', 'authentication', 'verify'
            ]
            
            # Contar indicadores
            inseguros = sum(1 for ind in indicadores_inseguridad if ind in codigo.lower())
            seguros = sum(1 for ind in indicadores_seguridad if ind in codigo.lower())
            
            # Calcular puntuación base
            puntuacion_base = 7.0 - min(7.0, inseguros * 1.0) + min(3.0, seguros * 0.5)
            
            # Verificar patrones específicos de seguridad
            if 'password' in codigo.lower() and ('hash' not in codigo.lower() and 'encrypt' not in codigo.lower()):
                puntuacion_base -= 1.0
            
            if 'sql' in codigo.lower() and 'prepare' not in codigo.lower() and 'bind_param' not in codigo.lower():
                puntuacion_base -= 1.0
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_base))
        except Exception as e:
            self.logger.error(f"Error al evaluar seguridad del código: {e}")
            return 5.0

    # Métodos auxiliares adicionales necesarios
    def _calcular_calificacion_alternativa(self, metricas, contexto):
        """Calcula una calificación alternativa basada en métricas cuando falla el modelo."""
        try:
            if 'funcionalidad' in metricas:
                # Métricas de código
                pesos = {
                    'funcionalidad': 0.3,
                    'eficiencia': 0.2,
                    'estilo': 0.15,
                    'documentacion': 0.15,
                    'cumplimiento': 0.1,
                    'seguridad': 0.1
                }
                calificacion = sum(metricas[m] * pesos[m] for m in pesos if m in metricas)
            else:
                # Métricas de texto
                pesos = {
                    'claridad': 0.25,
                    'profundidad': 0.3,
                    'estructura': 0.25,
                    'relevancia': 0.2
                }
                calificacion = sum(metricas[m] * pesos[m] for m in pesos if m in metricas)
            
            return calificacion
        except Exception as e:
            self.logger.error(f"Error al calcular calificación alternativa: {e}")
            return 5.0

    def _ajustar_calificacion(self, pred_base, relevancia, claridad, profundidad, estructura, contexto):
        """Ajusta la calificación base según métricas y contexto para contenido textual."""
        try:
            relevancia_norm = min(1.0, relevancia)
            ajuste_metricas = (claridad * 0.25 + profundidad * 0.3 + estructura * 0.25) / 10.0
            calificacion = pred_base * 0.4 + ajuste_metricas * 6.0
            
            if relevancia_norm < 0.5:
                calificacion *= relevancia_norm * 1.5
            
            return max(0.0, min(10.0, calificacion))
        except Exception as e:
            self.logger.error(f"Error al ajustar calificación: {e}")
            return pred_base

    def _ajustar_calificacion_codigo(self, pred_base, relevancia, funcionalidad, eficiencia, estilo, documentacion, cumplimiento, contexto, tipo_contenido):
        """Ajusta la calificación base según métricas y contexto para código."""
        try:
            relevancia_norm = min(1.0, relevancia)
            ajuste_metricas = (
                funcionalidad * 0.3 +
                eficiencia * 0.2 +
                estilo * 0.15 +
                documentacion * 0.15 +
                cumplimiento * 0.2
            ) / 10.0
            
            calificacion = pred_base * 0.3 + ajuste_metricas * 7.0
            
            if relevancia_norm < 0.5:
                calificacion *= relevancia_norm * 1.5
            
            return max(0.0, min(10.0, calificacion))
        except Exception as e:
            self.logger.error(f"Error al ajustar calificación de código: {e}")
            return pred_base

    def _identificar_fortalezas_debilidades(self, contenido, metricas, calificacion, contexto):
        """Identifica fortalezas y debilidades en contenido textual."""
        fortalezas = []
        debilidades = []
        
        if metricas['claridad'] >= 8.0:
            fortalezas.append("Excelente claridad en la exposición de ideas.")
        elif metricas['claridad'] <= 4.0:
            debilidades.append("Falta claridad en la exposición de ideas.")
            
        if metricas['profundidad'] >= 8.0:
            fortalezas.append("Análisis profundo y bien fundamentado del tema.")
        elif metricas['profundidad'] <= 4.0:
            debilidades.append("Falta profundidad en el análisis del tema.")
            
        if metricas['estructura'] >= 8.0:
            fortalezas.append("Excelente estructura y organización del contenido.")
        elif metricas['estructura'] <= 4.0:
            debilidades.append("La estructura y organización del contenido es deficiente.")
        
        return fortalezas, debilidades

    def _identificar_fortalezas_debilidades_codigo(self, codigo, metricas, calificacion, tipo_contenido, contexto):
        """Identifica fortalezas y debilidades en código."""
        fortalezas = []
        debilidades = []
        
        if metricas['funcionalidad'] >= 8.0:
            fortalezas.append("Excelente funcionalidad y cumplimiento de requisitos.")
        elif metricas['funcionalidad'] <= 4.0:
            debilidades.append("Problemas significativos en la funcionalidad del código.")
            
        if metricas['eficiencia'] >= 8.0:
            fortalezas.append("Código altamente eficiente y optimizado.")
        elif metricas['eficiencia'] <= 4.0:
            debilidades.append("Código ineficiente con problemas de rendimiento.")
            
        if metricas['estilo'] >= 8.0:
            fortalezas.append("Excelente estilo de codificación y legibilidad.")
        elif metricas['estilo'] <= 4.0:
            debilidades.append("Estilo de codificación deficiente y baja legibilidad.")
        
        return fortalezas, debilidades

    def _generar_comentarios_detallados(self, calificacion, contenido, titulo, objetivo, fortalezas, debilidades, estilo_aprendizaje, contexto):
        """Genera comentarios detallados para contenido textual."""
        categoria = self._determinar_categoria_calificacion(calificacion)
        
        if categoria in self.feedback_templates:
            plantilla_base = random.choice(self.feedback_templates[categoria])
        else:
            plantilla_base = "Tu trabajo ha sido evaluado y se han identificado aspectos positivos y áreas de mejora."
        
        comentario = f"{plantilla_base}\n\n"
        
        if fortalezas:
            comentario += "Aspectos destacados:\n"
            for i, fortaleza in enumerate(fortalezas[:3], 1):
                comentario += f"{i}. {fortaleza}\n"
            comentario += "\n"
        
        if debilidades:
            comentario += "Aspectos a mejorar:\n"
            for i, debilidad in enumerate(debilidades[:3], 1):
                comentario += f"{i}. {debilidad}\n"
            comentario += "\n"
        
        comentario += f"Calificación: {calificacion:.1f}/10.0\n\n"
        
        if calificacion >= 9.0:
            comentario += "¡Excelente trabajo! Sigue así."
        elif calificacion >= 7.0:
            comentario += "Buen trabajo. Considera las sugerencias para seguir mejorando."
        elif calificacion >= 5.0:
            comentario += "Trabajo aceptable. Implementa las sugerencias para mejorar significativamente."
        else:
            comentario += "Es necesario revisar los aspectos fundamentales del trabajo."
        
        return comentario

    def _generar_comentarios_detallados_codigo(self, calificacion, codigo, titulo, objetivo, fortalezas, debilidades, estilo_aprendizaje, tipo_contenido, contexto):
        """Genera comentarios detallados para código."""
        categoria = self._determinar_categoria_calificacion(calificacion)
        
        if categoria in self.code_feedback_templates:
            plantilla_base = random.choice(self.code_feedback_templates[categoria])
        else:
            plantilla_base = "Tu código ha sido evaluado y se han identificado aspectos positivos y áreas de mejora."
        
        comentario = f"{plantilla_base}\n\n"
        
        if fortalezas:
            comentario += "Aspectos destacados:\n"
            for i, fortaleza in enumerate(fortalezas[:3], 1):
                comentario += f"{i}. {fortaleza}\n"
            comentario += "\n"
        
        if debilidades:
            comentario += "Aspectos a mejorar:\n"
            for i, debilidad in enumerate(debilidades[:3], 1):
                comentario += f"{i}. {debilidad}\n"
            comentario += "\n"
        
        comentario += f"Calificación: {calificacion:.1f}/10.0\n\n"
        
        if calificacion >= 9.0:
            comentario += "¡Excelente código! Sigue aplicando estas buenas prácticas."
        elif calificacion >= 7.0:
            comentario += "Buen código. Considera las sugerencias para seguir mejorando."
        else:
            comentario += "Es necesario revisar los aspectos fundamentales del código."
        
        return comentario

    def _generar_sugerencias_mejora(self, calificacion, contenido, titulo, objetivo, debilidades, estilo_aprendizaje, contexto):
        """Genera sugerencias de mejora para contenido textual."""
        sugerencias = []
        
        for debilidad in debilidades:
            if "claridad" in debilidad.lower():
                sugerencias.append("Mejora la claridad utilizando ejemplos concretos y explicaciones más detalladas.")
            elif "profundidad" in debilidad.lower():
                sugerencias.append("Profundiza en el análisis investigando más fuentes y estableciendo conexiones entre conceptos.")
            elif "estructura" in debilidad.lower():
                sugerencias.append("Mejora la estructura organizando el contenido en secciones claramente definidas.")
        
        if calificacion < 5.0:
            sugerencias.append("Revisa los conceptos fundamentales del tema y asegúrate de comprenderlos correctamente.")
        
        if sugerencias:
            return "Sugerencias para mejorar:\n\n" + "\n".join(f"- {s}" for s in sugerencias[:5])
        else:
            return "Continúa con tu buen trabajo y sigue profundizando en el tema."

    def _generar_sugerencias_mejora_codigo(self, calificacion, codigo, titulo, objetivo, debilidades, estilo_aprendizaje, tipo_contenido, contexto):
        """Genera sugerencias de mejora para código."""
        sugerencias = []
        
        for debilidad in debilidades:
            if "funcionalidad" in debilidad.lower():
                sugerencias.append("Revisa la implementación para asegurar que cumple con todos los requisitos especificados.")
            elif "eficiencia" in debilidad.lower():
                sugerencias.append("Optimiza el código identificando y mejorando las secciones con mayor complejidad computacional.")
            elif "estilo" in debilidad.lower():
                sugerencias.append("Mejora la legibilidad siguiendo convenciones de estilo y usando nombres descriptivos.")
        
        if calificacion < 5.0:
            sugerencias.append("Revisa los conceptos fundamentales de programación y asegúrate de comprender los requisitos.")
        
        if sugerencias:
            return "Sugerencias para mejorar tu código:\n\n" + "\n".join(f"- {s}" for s in sugerencias[:5])
        else:
            return "Continúa con tu buen trabajo y sigue profundizando en las técnicas de programación."

    def _recomendar_recursos(self, estilo_aprendizaje, titulo, objetivo, calificacion, tipo_contenido):
        """Recomienda recursos educativos según el estilo de aprendizaje y el tema."""
        recursos_recomendados = []
        
        # Determinar la categoría temática
        categoria = self._determinar_categoria_tematica(titulo, objetivo)
        
        # Si no se especifica estilo de aprendizaje, recomendar recursos generales
        if not estilo_aprendizaje:
            if categoria == 'programacion':
                recursos_recomendados.extend([
                    {"titulo": "Documentación oficial del lenguaje", "url": "https://docs.python.org/"},
                    {"titulo": "Ejercicios prácticos de programación", "url": "https://www.hackerrank.com/"}
                ])
            else:
                recursos_recomendados.extend([
                    {"titulo": "Khan Academy", "url": "https://www.khanacademy.org/"},
                    {"titulo": "Coursera", "url": "https://www.coursera.org/"}
                ])
        else:
            # Recomendar recursos específicos según el estilo de aprendizaje
            estilos = [e.strip().lower() for e in estilo_aprendizaje.split(',')]
            
            for estilo in estilos:
                if estilo in self.recursos_educativos and categoria in self.recursos_educativos[estilo]:
                    recursos_estilo = self.recursos_educativos[estilo][categoria]
                    if calificacion < 5.0:
                        recursos_recomendados.extend(recursos_estilo[:2])
                    else:
                        recursos_recomendados.extend(recursos_estilo[-2:])
        
        return recursos_recomendados[:5]

    def _determinar_categoria_tematica(self, titulo, objetivo):
        """Determina la categoría temática de una práctica basada en el título y objetivo."""
        texto_combinado = f"{titulo} {objetivo}".lower()
        
        keywords = {
            'programacion': ['programación', 'código', 'algoritmo', 'desarrollo', 'software'],
            'matematicas': ['matemáticas', 'cálculo', 'álgebra', 'geometría', 'estadística'],
            'ciencias': ['física', 'química', 'biología', 'ciencia', 'experimento'],
            'bases_de_datos': ['base de datos', 'sql', 'tabla', 'consulta', 'mysql']
        }
        
        counts = {}
        for categoria, palabras in keywords.items():
            counts[categoria] = sum(1 for palabra in palabras if palabra in texto_combinado)
        
        max_count = 0
        max_categoria = 'general'
        
        for categoria, count in counts.items():
            if count > max_count:
                max_count = count
                max_categoria = categoria
        
        return max_categoria

    def _determinar_categoria_calificacion(self, calificacion):
        """Determina la categoría de calificación."""
        if calificacion >= 9.0:
            return 'excellent'
        elif calificacion >= 8.0:
            return 'very_good'
        elif calificacion >= 7.0:
            return 'good'
        elif calificacion >= 6.0:
            return 'average'
        elif calificacion >= 4.0:
            return 'needs_improvement'
        else:
            return 'poor'

    def _formatear_metricas(self, metricas, tipo_contenido):
        """Formatea las métricas para la respuesta."""
        metricas_formateadas = {}
        for metrica, valor in metricas.items():
            metricas_formateadas[metrica] = round(float(valor), 2)
        return metricas_formateadas

    def _generar_respuesta_irrelevante(self, estilo_aprendizaje, titulo_actividad, objetivo_actividad):
        """Genera una respuesta para contenido irrelevante."""
        return {
            "calificacion": 0.0,
            "comentarios": "El contenido entregado no tiene relación con el tema solicitado. Es fundamental abordar el tema específico de la actividad.",
            "sugerencias": "Revisa cuidadosamente el título y objetivo de la actividad para asegurarte de abordar el tema correcto.",
            "relevancia": 0.0,
            "tipo_contenido": "irrelevante",
            "metricas": {},
            "fortalezas": [],
            "debilidades": ["Contenido no relacionado con el tema solicitado"],
            "recursos_recomendados": []
        }

    def _registrar_evaluacion(self, contenido, tipo_contenido, titulo, objetivo, calificacion, metricas, contexto=None):
        """Registra una evaluación en el historial para aprendizaje continuo."""
        try:
            evaluacion = {
                'contenido': contenido,
                'tipo_contenido': tipo_contenido,
                'titulo': titulo,
                'objetivo': objetivo,
                'calificacion': calificacion,
                'metricas': metricas,
                'contexto': contexto or {},
                'timestamp': datetime.now().isoformat()
            }
            
            self.historial_evaluaciones.append(evaluacion)
            
            if len(self.historial_evaluaciones) > 1000:
                self.historial_evaluaciones = self.historial_evaluaciones[-1000:]
            
            if len(self.historial_evaluaciones) % 10 == 0:
                threading.Thread(target=self._entrenar_incremental, daemon=True).start()
        except Exception as e:
            self.logger.error(f"Error al registrar evaluación: {e}")

    def _entrenar_incremental(self):
        """Realiza entrenamiento incremental con las evaluaciones recientes."""
        try:
            if not SystemMonitor.check_resources(self.memory_threshold_mb):
                self.logger.warning("Recursos insuficientes para entrenamiento incremental.")
                return
            
            evaluaciones_codigo = [e for e in self.historial_evaluaciones if e['tipo_contenido'] in ['codigo', 'sql']]
            evaluaciones_texto = [e for e in self.historial_evaluaciones if e['tipo_contenido'] not in ['codigo', 'sql']]
            
            if len(evaluaciones_codigo) >= 5:
                entradas_codigo = [e['contenido'] for e in evaluaciones_codigo]
                salidas_codigo = [e['calificacion'] for e in evaluaciones_codigo]
                
                self._entrenar_modelo(
                    self.modelo_codigo,
                    self.optimizer_codigo,
                    entradas_codigo,
                    salidas_codigo,
                    epochs=3,
                    batch_size=min(4, len(entradas_codigo))
                )
                self.logger.info(f"Entrenamiento incremental del modelo de código completado")
            
            if len(evaluaciones_texto) >= 5:
                entradas_texto = [e['contenido'] for e in evaluaciones_texto]
                salidas_texto = [e['calificacion'] for e in evaluaciones_texto]
                
                self._entrenar_modelo(
                    self.modelo_contenido,
                    self.optimizer_contenido,
                    entradas_texto,
                    salidas_texto,
                    epochs=3,
                    batch_size=min(4, len(entradas_texto))
                )
                self.logger.info(f"Entrenamiento incremental del modelo de texto completado")
            
            self.guardar_modelos()
            
        except Exception as e:
            self.logger.error(f"Error en entrenamiento incremental: {e}")

    def _entrenar_modelo(self, modelo, optimizer, entradas, salidas, epochs=5, batch_size=8):
        """Entrena un modelo específico con los datos proporcionados"""
        try:
            textos_procesados = [self._preprocess_text(texto) for texto in entradas]
            
            # Verificar que el vectorizador esté ajustado
            if not hasattr(self.vectorizer, 'vocabulary_'):
                self.vectorizer.fit(textos_procesados)
            
            dataset = TextDataset(textos_procesados, salidas, self.vectorizer)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            
            modelo.train()
            
            for epoch in range(epochs):
                total_loss = 0.0
                for inputs, targets in dataloader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    outputs = modelo(inputs)
                    loss = self.criterion(outputs['global'].squeeze(), targets)
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                self.logger.info(f"Época {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        except Exception as e:
            self.logger.error(f"Error en entrenamiento de modelo: {e}")

    def guardar_modelos(self, ruta_base='modelos'):
        """Guarda los modelos y vectorizador en disco."""
        try:
            os.makedirs(ruta_base, exist_ok=True)
            
            torch.save(self.modelo_contenido.state_dict(), os.path.join(ruta_base, 'modelo_contenido_enhanced.pt'))
            torch.save(self.modelo_codigo.state_dict(), os.path.join(ruta_base, 'modelo_codigo_enhanced.pt'))
            
            with open(os.path.join(ruta_base, 'vectorizer_enhanced.pkl'), 'wb') as f:
                pickle.dump(self.vectorizer, f)
                
            with open(os.path.join(ruta_base, 'historial_evaluaciones.pkl'), 'wb') as f:
                pickle.dump(self.historial_evaluaciones[-100:], f)
                
            self.logger.info(f"Modelos guardados en {ruta_base}")
            return True
        except Exception as e:
            self.logger.error(f"Error al guardar modelos: {e}")
            return False

    def cargar_modelos(self, ruta_base='modelos'):
        """Carga los modelos y vectorizador desde disco."""
        try:
            ruta_contenido = os.path.join(ruta_base, 'modelo_contenido_enhanced.pt')
            ruta_codigo = os.path.join(ruta_base, 'modelo_codigo_enhanced.pt')
            ruta_vectorizer = os.path.join(ruta_base, 'vectorizer_enhanced.pkl')
            ruta_historial = os.path.join(ruta_base, 'historial_evaluaciones.pkl')
            
            if not os.path.exists(ruta_contenido) or not os.path.exists(ruta_codigo) or not os.path.exists(ruta_vectorizer):
                self.logger.warning("Archivos de modelo no encontrados. Se usarán modelos nuevos.")
                return False
                
            self.modelo_contenido.load_state_dict(torch.load(ruta_contenido, map_location=self.device))
            self.modelo_contenido.eval()
            
            self.modelo_codigo.load_state_dict(torch.load(ruta_codigo, map_location=self.device))
            self.modelo_codigo.eval()
            
            with open(ruta_vectorizer, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            if os.path.exists(ruta_historial):
                with open(ruta_historial, 'rb') as f:
                    self.historial_evaluaciones = pickle.load(f)
                self.logger.info(f"Historial de evaluaciones cargado: {len(self.historial_evaluaciones)} registros")
                
            self.logger.info(f"Modelos cargados desde {ruta_base}")
            return True
        except Exception as e:
            self.logger.error(f"Error al cargar modelos: {e}")
            return False

    def generar_practica(self, titulo, objetivo):
        """
        Genera una práctica completa y detallada basada en el título y objetivo.
        """
        try:
            # Combinar título y objetivo
            prompt = f"{titulo}. {objetivo}"
            
            # Encontrar prácticas similares en la base de conocimiento
            prompt_vec = self.vectorizer.transform([prompt])
            similarities = cosine_similarity(prompt_vec, self.tfidf_matrix)
            
            # Obtener los 3 ejemplos más similares
            top_indices = similarities[0].argsort()[-3:][::-1]
            practicas_similares = [self.base_conocimiento[i] for i in top_indices]
            
            # Determinar la categoría temática
            categoria = self._determinar_categoria_tematica(titulo, objetivo)
            
            # Generar descripción combinando elementos de prácticas similares
            descripcion = f"Esta práctica se enfoca en {titulo.lower()}. "
            descripcion += f"{practicas_similares[0]} "
            if len(practicas_similares) > 1:
                descripcion += f"Además, aborda aspectos relacionados con {practicas_similares[1].lower()}."
            
            # Generar instrucciones específicas según la categoría
            instrucciones = [
                f"Lee detenidamente el objetivo: {objetivo}",
                "Investiga los conceptos teóricos relacionados con el tema"
            ]
            
            if categoria == 'programacion':
                instrucciones.extend([
                    "Diseña una solución que cumpla con los requisitos especificados",
                    "Implementa tu solución utilizando el lenguaje de programación adecuado",
                    "Documenta tu código con comentarios explicativos",
                    "Realiza pruebas para verificar el correcto funcionamiento",
                    "Analiza la complejidad y eficiencia de tu solución"
                ])
            elif categoria == 'matematicas':
                instrucciones.extend([
                    "Identifica los conceptos matemáticos relevantes para el problema",
                    "Desarrolla paso a paso la solución, justificando cada etapa",
                    "Incluye demostraciones cuando sea necesario",
                    "Verifica tus resultados con ejemplos concretos",
                    "Analiza las implicaciones y aplicaciones de tu solución"
                ])
            elif categoria == 'ciencias':
                instrucciones.extend([
                    "Formula una hipótesis basada en el objetivo planteado",
                    "Diseña un experimento o metodología para comprobar tu hipótesis",
                    "Recopila y analiza datos relevantes",
                    "Interpreta los resultados obtenidos",
                    "Elabora conclusiones fundamentadas en la evidencia"
                ])
            elif categoria == 'bases_de_datos':
                instrucciones.extend([
                    "Diseña un modelo de datos que cumpla con los requisitos",
                    "Implementa el esquema de base de datos con las tablas y relaciones necesarias",
                    "Crea consultas SQL para manipular y recuperar datos",
                    "Optimiza las consultas para mejorar el rendimiento",
                    "Documenta el diseño y las consultas implementadas"
                ])
            
            instrucciones.append("Entrega tu trabajo antes de la fecha límite establecida")
            
            # Generar recursos recomendados
            recursos = [
                "Material de clase y apuntes",
                "Biblioteca digital de la universidad"
            ]
            
            if categoria == 'programacion':
                recursos.extend([
                    "Documentación oficial del lenguaje de programación",
                    "Repositorios de código en GitHub",
                    "Foros especializados como Stack Overflow"
                ])
            elif categoria == 'matematicas':
                recursos.extend([
                    "Libros de texto recomendados en el curso",
                    "Artículos académicos relacionados",
                    "Plataformas como Khan Academy o Wolfram Alpha"
                ])
            elif categoria == 'ciencias':
                recursos.extend([
                    "Artículos científicos recientes sobre el tema",
                    "Bases de datos especializadas",
                    "Simuladores y laboratorios virtuales"
                ])
            elif categoria == 'bases_de_datos':
                recursos.extend([
                    "Documentación oficial de sistemas de bases de datos",
                    "Tutoriales de SQL y diseño de bases de datos",
                    "Herramientas de modelado como MySQL Workbench o dbdiagram.io"
                ])
            
            # Generar criterios de evaluación
            criterios_evaluacion = []
            
            if categoria == 'programacion':
                criterios_evaluacion = [
                    "Funcionalidad: El código cumple con los requisitos especificados (40%)",
                    "Eficiencia: La solución utiliza algoritmos y estructuras de datos apropiados (25%)",
                    "Estilo y legibilidad: El código sigue buenas prácticas y convenciones (15%)",
                    "Documentación: El código incluye comentarios claros y explicativos (20%)"
                ]
            elif categoria == 'matematicas':
                criterios_evaluacion = [
                    "Comprensión de conceptos: Dominio de los fundamentos teóricos (35%)",
                    "Desarrollo de soluciones: Aplicación correcta de métodos y técnicas (40%)",
                    "Claridad y rigor: Precisión en la notación y argumentación (15%)",
                    "Presentación: Organización y estructura del trabajo (10%)"
                ]
            elif categoria == 'ciencias':
                criterios_evaluacion = [
                    "Metodología: Diseño experimental y procedimientos (30%)",
                    "Análisis de datos: Interpretación y procesamiento de resultados (35%)",
                    "Conclusiones: Validez y fundamentación de las inferencias (25%)",
                    "Presentación: Claridad y estructura del informe (10%)"
                ]
            elif categoria == 'bases_de_datos':
                criterios_evaluacion = [
                    "Diseño: Modelo de datos normalizado y adecuado (30%)",
                    "Implementación: Correcta creación de tablas y relaciones (25%)",
                    "Consultas: Funcionalidad y eficiencia de las consultas SQL (35%)",
                    "Documentación: Explicación clara del diseño y las consultas (10%)"
                ]
            else:
                criterios_evaluacion = [
                    "Comprensión del tema y conceptos clave (30%)",
                    "Calidad de la solución o análisis (40%)",
                    "Documentación y presentación (20%)",
                    "Creatividad e innovación (10%)"
                ]
            
            # Generar objetivos de aprendizaje
            objetivos_aprendizaje = []
            
            if categoria == 'programacion':
                objetivos_aprendizaje = [
                    f"Comprender los fundamentos teóricos de {titulo.lower()}",
                    "Desarrollar habilidades de resolución de problemas mediante programación",
                    "Aplicar buenas prácticas de desarrollo de software",
                    "Implementar soluciones eficientes y bien documentadas"
                ]
            elif categoria == 'matematicas':
                objetivos_aprendizaje = [
                    f"Dominar los conceptos matemáticos relacionados con {titulo.lower()}",
                    "Desarrollar capacidad de razonamiento lógico y abstracto",
                    "Aplicar métodos matemáticos para resolver problemas concretos",
                    "Comunicar ideas matemáticas con precisión y claridad"
                ]
            elif categoria == 'ciencias':
                objetivos_aprendizaje = [
                    f"Comprender los principios científicos de {titulo.lower()}",
                    "Desarrollar habilidades de investigación y análisis",
                    "Aplicar el método científico en situaciones concretas",
                    "Interpretar datos y resultados experimentales"
                ]
            elif categoria == 'bases_de_datos':
                objetivos_aprendizaje = [
                    f"Comprender los conceptos fundamentales de {titulo.lower()}",
                    "Desarrollar habilidades de diseño de bases de datos relacionales",
                    "Aplicar SQL para manipular y consultar datos eficientemente",
                    "Implementar soluciones de almacenamiento de datos optimizadas"
                ]
            
            # Generar recomendaciones según la categoría
            recomendaciones = ""
            if categoria == 'programacion':
                recomendaciones = (
                    "Se recomienda practicar en plataformas como LeetCode y consultar documentación oficial."
                )
            elif categoria == 'matematicas':
                recomendaciones = (
                    "Utiliza herramientas como Wolfram Alpha y busca problemas similares en Khan Academy."
                )
            elif categoria == 'ciencias':
                recomendaciones = (
                    "Refuerza tus conocimientos con simuladores virtuales y experimentos caseros controlados."
                )
            elif categoria == 'bases_de_datos':
                recomendaciones = (
                    "Practica con herramientas como MySQL Workbench o SQLite para implementar y probar tus consultas."
                )
            else:
                recomendaciones = (
                    "Consulta con tu profesor o utiliza recursos digitales para profundizar en el tema."
                )
        
            # Construir y retornar la práctica completa
                        # Construir y retornar la práctica completa
            nueva_practica = {
                "titulo": titulo,
                "objetivo": objetivo,
                "objetivos_aprendizaje": objetivos_aprendizaje,
                "descripcion": descripcion,
                "actividades": instrucciones,  # Reusamos instrucciones como actividades
                "categoria": categoria,
                "instrucciones": instrucciones,
                "recursos": recursos,
                "criterios_evaluacion": criterios_evaluacion,
                "tiempo_estimado": "2-3 horas",
                "nivel_dificultad": "Intermedio",
                "recomendaciones": recomendaciones
            }
        
            return nueva_practica
        except Exception as e:
            self.logger.error(f"Error al generar práctica: {e}")
            self.logger.error(traceback.format_exc())
            
            # Práctica por defecto en caso de error
            return {
                "titulo": titulo,
                "objetivo": objetivo,
                "objetivos_aprendizaje": ["Comprender los conceptos básicos del tema"],
                "descripcion": f"Práctica sobre {titulo}. {objetivo}",
                "actividades": ["Investigar el tema", "Realizar ejercicios prácticos", "Documentar resultados"],
                "categoria": "general",
                "instrucciones": ["Leer detenidamente el objetivo", "Investigar el tema", "Realizar la actividad"],
                "recursos": ["Material de clase", "Recursos en línea", "Bibliografía recomendada"],
                "criterios_evaluacion": ["Comprensión del tema (40%)", "Calidad del trabajo (40%)", "Presentación (20%)"],
                "tiempo_estimado": "1-2 horas",
                "nivel_dificultad": "Básico",
                "recomendaciones": "Consulta con tu profesor si tienes dudas sobre el tema."
            }

    def _generar_sugerencias_mejora_codigo(self, calificacion, codigo, titulo, objetivo, debilidades, estilo_aprendizaje, tipo_contenido, contexto):
        """Genera sugerencias de mejora para código."""
        try:
            sugerencias = []

            # Generar sugerencias basadas en debilidades
            for debilidad in debilidades:
                if "funcionalidad" in debilidad.lower():
                    sugerencias.append("Revisa la implementación para asegurar que cumple con todos los requisitos especificados.")
                elif "eficiencia" in debilidad.lower() or "rendimiento" in debilidad.lower():
                    sugerencias.append("Optimiza el código identificando y mejorando las secciones con mayor complejidad computacional.")
                elif "estilo" in debilidad.lower() or "legibilidad" in debilidad.lower():
                    sugerencias.append("Mejora la legibilidad siguiendo convenciones de estilo, usando nombres descriptivos y manteniendo una indentación consistente.")
                elif "documentación" in debilidad.lower() or "comentarios" in debilidad.lower():
                    sugerencias.append("Añade comentarios explicativos para las secciones complejas y documenta las funciones con sus parámetros y valores de retorno.")
                elif "seguridad" in debilidad.lower():
                    sugerencias.append("Implementa prácticas de seguridad como validación de entradas, manejo seguro de datos sensibles y protección contra vulnerabilidades comunes.")
                elif "bucles anidados" in debilidad.lower():
                    sugerencias.append("Refactoriza los bucles anidados para reducir la complejidad y mejorar el rendimiento.")
                elif "incompleto" in debilidad.lower() or "pendiente" in debilidad.lower():
                    sugerencias.append("Completa las secciones marcadas como pendientes o incompletas para asegurar la funcionalidad total.")

            # Añadir sugerencias específicas según tipo de contenido
            if tipo_contenido == 'sql':
                if "índices" in " ".join(debilidades).lower() or calificacion < 7.0:
                    sugerencias.append("Añade índices apropiados para optimizar las consultas, especialmente en columnas usadas en cláusulas WHERE, JOIN y ORDER BY.")
                if "normalización" not in " ".join(debilidades).lower() and calificacion < 7.0:
                    sugerencias.append("Revisa la normalización de las tablas para evitar redundancia y anomalías en los datos.")
            else:
                if "manejo de errores" not in " ".join(debilidades).lower() and calificacion < 7.0:
                    sugerencias.append("Implementa un manejo robusto de errores y excepciones para mejorar la estabilidad del código.")
                if "modularidad" not in " ".join(debilidades).lower() and calificacion < 7.0:
                    sugerencias.append("Mejora la modularidad dividiendo el código en funciones o clases con responsabilidades bien definidas.")

            # Añadir sugerencias generales según calificación
            if calificacion < 5.0:
                sugerencias.append("Revisa los conceptos fundamentales de programación y asegúrate de comprender correctamente los requisitos.")
                sugerencias.append("Consulta la documentación oficial del lenguaje o tecnología utilizada para mejorar tu implementación.")
            elif calificacion < 7.0:
                sugerencias.append("Realiza pruebas exhaustivas para identificar y corregir posibles errores o comportamientos inesperados.")
                sugerencias.append("Refactoriza el código para mejorar su estructura y mantenibilidad.")

            # Añadir sugerencias específicas según contexto
            if contexto and 'enfoque_principal' in contexto:
                if contexto['enfoque_principal'] == 'programacion' and calificacion < 8.0:
                    sugerencias.append("Aplica patrones de diseño apropiados para mejorar la estructura y flexibilidad del código.")
                elif contexto['enfoque_principal'] == 'base_de_datos' and calificacion < 8.0:
                    sugerencias.append("Optimiza las consultas SQL para mejorar el rendimiento y la eficiencia de la base de datos.")

            # Añadir sugerencias específicas según estilo de aprendizaje
            if estilo_aprendizaje:
                estilos = [e.strip().lower() for e in estilo_aprendizaje.split(',')]

                if 'visual' in estilos and calificacion < 9.0:
                    sugerencias.append("Utiliza diagramas de flujo o UML para planificar y visualizar la estructura del código antes de implementarlo.")

                if 'auditivo' in estilos and calificacion < 9.0:
                    sugerencias.append("Explica verbalmente tu código a un compañero para identificar posibles mejoras o inconsistencias.")

                if ('kinestesico' in estilos or 'kinestésico' in estilos) and calificacion < 9.0:
                    sugerencias.append("Practica con ejercicios adicionales de programación para reforzar los conceptos aplicados en este código.")

                if 'lectura_escritura' in estilos and calificacion < 9.0:
                    sugerencias.append("Estudia libros o artículos sobre buenas prácticas de programación y aplica los conceptos aprendidos en tu código.")

            # Limitar a 5 sugerencias para no abrumar
            sugerencias = sugerencias[:5]

            # Formatear sugerencias
            if sugerencias:
                return "Sugerencias para mejorar tu código:\n\n" + "\n".join(f"- {s}" for s in sugerencias)
            else:
                return "Continúa con tu buen trabajo y sigue profundizando en las técnicas de programación."
        except Exception as e:
            self.logger.error(f"Error al generar sugerencias de mejora para código: {e}")
            return "Revisa los aspectos mencionados en las debilidades y consulta documentación adicional para mejorar tu código."
    
    def _determinar_categoria_tematica(self, titulo, objetivo):
        """Determina la categoría temática de una práctica basada en el título y objetivo."""
        texto_combinado = f"{titulo} {objetivo}".lower()
        
        # Palabras clave por categoría
        keywords = {
            'programacion': ['programación', 'código', 'algoritmo', 'desarrollo', 'software', 'aplicación',
                            'web', 'móvil', 'python', 'java', 'javascript', 'c++', 'php', 'ruby', 'go',
                            'función', 'clase', 'objeto', 'variable', 'bucle', 'condicional', 'array',
                            'lista', 'diccionario', 'estructura de datos', 'api', 'framework', 'biblioteca',
                            'desarrollo web', 'frontend', 'backend', 'fullstack', 'devops', 'git'],
            
            'matematicas': ['matemáticas', 'cálculo', 'álgebra', 'geometría', 'estadística', 'probabilidad',
                           'ecuación', 'función', 'derivada', 'integral', 'límite', 'vector', 'matriz',
                           'determinante', 'teorema', 'demostración', 'conjunto', 'serie', 'convergencia',
                           'divergencia', 'polinomio', 'logaritmo', 'exponencial', 'trigonometría'],
            
            'ciencias': ['física', 'química', 'biología', 'geología', 'astronomía', 'ciencia', 'científico',
                        'experimento', 'laboratorio', 'hipótesis', 'teoría', 'ley', 'átomo', 'molécula',
                        'célula', 'organismo', 'ecosistema', 'reacción', 'energía', 'fuerza', 'masa',
                        'velocidad', 'aceleración', 'presión', 'temperatura', 'calor', 'electricidad',
                        'magnetismo', 'óptica', 'mecánica', 'termodinámica', 'cuántica', 'relatividad'],
            
            'bases_de_datos': ['base de datos', 'sql', 'nosql', 'tabla', 'consulta', 'query', 'join',
                              'índice', 'clave primaria', 'clave foránea', 'relación', 'entidad',
                              'atributo', 'normalización', 'desnormalización', 'transacción', 'acid',
                              'mysql', 'postgresql', 'oracle', 'mongodb', 'cassandra', 'redis',
                              'almacenamiento', 'datos', 'información', 'modelo relacional', 'er',
                              'dbms', 'sgbd', 'crud', 'select', 'insert', 'update', 'delete']
        }
        
        # Contar coincidencias por categoría
        counts = {}
        for categoria, palabras in keywords.items():
            counts[categoria] = sum(1 for palabra in palabras if palabra in texto_combinado)
        
        # Determinar la categoría con más coincidencias
        max_count = 0
        max_categoria = 'general'
        
        for categoria, count in counts.items():
            if count > max_count:
                max_count = count
                max_categoria = categoria
        
        return max_categoria

    def _analizar_metricas(self, contenido, titulo, objetivo, contexto):
        """Analiza métricas para contenido textual."""
        metricas = {}
        
        # Calcular claridad
        metricas['claridad'] = self._evaluar_claridad(contenido)
        
        # Calcular profundidad
        metricas['profundidad'] = self._evaluar_profundidad(contenido, titulo, objetivo)
        
        # Calcular estructura
        metricas['estructura'] = self._evaluar_estructura(contenido)
        
        # Calcular relevancia
        metricas['relevancia'] = self._calcular_relevancia(contenido, titulo, objetivo)
        
        return metricas

    def _analizar_metricas_codigo(self, contenido, titulo, objetivo, contexto):
        """Analiza métricas para código."""
        metricas = {}
        
        # Calcular funcionalidad
        metricas['funcionalidad'] = self._evaluar_funcionalidad_codigo(contenido, titulo, objetivo)
        
        # Calcular eficiencia
        metricas['eficiencia'] = self._evaluar_eficiencia_codigo(contenido)
        
        # Calcular estilo
        metricas['estilo'] = self._evaluar_estilo_codigo(contenido)
        
        # Calcular documentación
        metricas['documentacion'] = self._evaluar_documentacion_codigo(contenido)
        
        # Calcular cumplimiento de requisitos
        metricas['cumplimiento'] = self._evaluar_cumplimiento_requisitos(contenido, titulo, objetivo)
        
        # Calcular seguridad
        metricas['seguridad'] = self._evaluar_seguridad_codigo(contenido)
        
        # Calcular relevancia
        metricas['relevancia'] = self._calcular_relevancia(contenido, titulo, objetivo)
        
        return metricas

    def _evaluar_claridad(self, contenido):
        """Evalúa la claridad del contenido."""
        try:
            # Indicadores de claridad
            indicadores_positivos = [
                'por lo tanto', 'en consecuencia', 'es decir', 'en otras palabras',
                'por ejemplo', 'como se muestra', 'para ilustrar', 'específicamente',
                'en particular', 'en resumen', 'en conclusión', 'finalmente'
            ]
            
            indicadores_negativos = [
                'de alguna manera', 'quizás', 'tal vez', 'posiblemente',
                'se podría decir', 'en cierto modo', 'más o menos'
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in contenido.lower())
            negativos = sum(1 for ind in indicadores_negativos if ind in contenido.lower())
            
            # Calcular longitud promedio de oraciones
            oraciones = re.split(r'[.!?]+', contenido)
            oraciones = [o.strip() for o in oraciones if o.strip()]
            
            if not oraciones:
                return 5.0  # Valor neutral
                
            longitud_promedio = sum(len(o.split()) for o in oraciones) / len(oraciones)
            
            # Penalizar oraciones muy largas o muy cortas
            factor_longitud = 1.0
            if longitud_promedio > 30:  # Oraciones muy largas
                factor_longitud = 0.8
            elif longitud_promedio < 5:  # Oraciones muy cortas
                factor_longitud = 0.9
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(5.0, positivos * 0.5) - min(3.0, negativos * 0.5)
            
            # Ajustar por longitud de oraciones
            puntuacion_ajustada = puntuacion_base * factor_longitud
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_ajustada))
        except Exception as e:
            self.logger.error(f"Error al evaluar claridad: {e}")
            return 5.0  # Valor neutral en caso de error

    def _evaluar_profundidad(self, contenido, titulo, objetivo):
        """Evalúa la profundidad del contenido."""
        try:
            # Indicadores de profundidad
            indicadores_positivos = [
                'análisis', 'evaluación', 'comparación', 'crítica', 'interpretación',
                'perspectiva', 'enfoque', 'metodología', 'teoría', 'concepto',
                'fundamento', 'principio', 'paradigma', 'implicación', 'consecuencia'
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in contenido.lower())
            
            # Calcular diversidad de vocabulario
            palabras = re.findall(r'\b\w+\b', contenido.lower())
            palabras_unicas = set(palabras)
            
            if not palabras:
                return 5.0  # Valor neutral
                
            diversidad = len(palabras_unicas) / len(palabras)
            
            # Calcular longitud del contenido (normalizada)
            longitud_normalizada = min(1.0, len(contenido) / 5000)  # Normalizar a 5000 caracteres
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(3.0, positivos * 0.3) + diversidad * 2.0 + longitud_normalizada * 2.0
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_base))
        except Exception as e:
            self.logger.error(f"Error al evaluar profundidad: {e}")
            return 5.0  # Valor neutral en caso de error

    def _evaluar_estructura(self, contenido):
        """Evalúa la estructura del contenido."""
        try:
            # Indicadores de estructura
            indicadores_positivos = [
                'introducción', 'desarrollo', 'conclusión', 'en primer lugar',
                'en segundo lugar', 'finalmente', 'por un lado', 'por otro lado',
                'sin embargo', 'no obstante', 'además', 'asimismo', 'en resumen'
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in contenido.lower())
            
            # Detectar párrafos
            parrafos = contenido.split('\n\n')
            parrafos = [p.strip() for p in parrafos if p.strip()]
            
            if not parrafos:
                return 5.0  # Valor neutral
                
            # Evaluar longitud de párrafos
            longitudes = [len(p) for p in parrafos]
            longitud_promedio = sum(longitudes) / len(longitudes)
            desviacion = sum(abs(l - longitud_promedio) for l in longitudes) / len(longitudes)
            
            # Penalizar desviación alta (párrafos muy desiguales)
            factor_desviacion = 1.0
            if desviacion > longitud_promedio * 0.5:
                factor_desviacion = 0.9
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(3.0, positivos * 0.3) + min(2.0, len(parrafos) * 0.2)
            
            # Ajustar por desviación
            puntuacion_ajustada = puntuacion_base * factor_desviacion
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_ajustada))
        except Exception as e:
            self.logger.error(f"Error al evaluar estructura: {e}")
            return 5.0  # Valor neutral en caso de error

    def _evaluar_funcionalidad_codigo(self, codigo, titulo, objetivo):
        """Evalúa la funcionalidad del código."""
        try:
            # Indicadores de funcionalidad
            indicadores_positivos = [
                'return', 'print', 'output', 'resultado', 'función', 'método',
                'class', 'def', 'function', 'if', 'else', 'for', 'while',
                'try', 'except', 'finally', 'with', 'import', 'from'
            ]
            
            indicadores_negativos = [
                'error', 'exception', 'fail', 'bug', 'issue', 'problem',
                'warning', 'deprecated', 'todo', 'fixme'
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in codigo.lower())
            negativos = sum(1 for ind in indicadores_negativos if ind in codigo.lower())
            
            # Detectar patrones de código incompleto
            codigo_incompleto = re.search(r'(#|//)\s*(todo|fixme|xxx|pendiente|incompleto)', codigo.lower())
            
            # Penalizar código incompleto
            factor_completitud = 1.0
            if codigo_incompleto:
                factor_completitud = 0.8
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(5.0, positivos * 0.2) - min(3.0, negativos * 0.5)
            
            # Ajustar por completitud
            puntuacion_ajustada = puntuacion_base * factor_completitud
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_ajustada))
        except Exception as e:
            self.logger.error(f"Error al evaluar funcionalidad del código: {e}")
            return 5.0  # Valor neutral en caso de error

    def _evaluar_eficiencia_codigo(self, codigo):
        """Evalúa la eficiencia del código."""
        try:
            # Indicadores de eficiencia
            indicadores_positivos = [
                'optimiz', 'eficien', 'rendimiento', 'performance', 'complejidad',
                'O(1)', 'O(log n)', 'O(n)', 'memoiz', 'cache', 'buffer'
            ]
            
            indicadores_negativos = [
                'O(n²)', 'O(n^2)', 'O(n*n)', 'O(2^n)', 'O(n!)',
                'bucle anidado', 'nested loop', 'for.*for', 'while.*while'
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in codigo.lower())
            
            # Buscar patrones de código ineficiente
            bucles_anidados = len(re.findall(r'for.*for|while.*while', codigo.lower()))
            
            # Penalizar bucles anidados
            factor_bucles = 1.0
            if bucles_anidados > 0:
                factor_bucles = 1.0 - min(0.3, bucles_anidados * 0.1)
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(5.0, positivos * 0.5)
            
            # Ajustar por bucles anidados
            puntuacion_ajustada = puntuacion_base * factor_bucles
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_ajustada))
        except Exception as e:
            self.logger.error(f"Error al evaluar eficiencia del código: {e}")
            return 5.0  # Valor neutral en caso de error

    def _evaluar_estilo_codigo(self, codigo):
        """Evalúa el estilo del código."""
        try:
            # Indicadores de buen estilo
            indicadores_positivos = [
                'def ', 'class ', 'function ', 'import ', 'from ', 'return ',
                'if ', 'else:', 'elif ', 'for ', 'while ', 'try:', 'except:',
                '# ', '"""', "'''"
            ]
            
            # Contar indicadores
            positivos = sum(1 for ind in indicadores_positivos if ind in codigo)
            
            # Evaluar indentación
            lineas = codigo.split('\n')
            lineas = [l for l in lineas if l.strip()]
            
            if not lineas:
                return 5.0  # Valor neutral
                
            # Verificar consistencia de indentación
            indentaciones = [len(l) - len(l.lstrip()) for l in lineas]
            indentacion_consistente = len(set(i % 4 for i in indentaciones)) <= 1
            
            # Verificar longitud de líneas
            lineas_largas = sum(1 for l in lineas if len(l) > 100)
            factor_longitud = 1.0 - min(0.2, lineas_largas / len(lineas))
            
            # Verificar comentarios
            comentarios = sum(1 for l in lineas if l.strip().startswith(('#', '//', '/*', '*', '*/')))
            ratio_comentarios = comentarios / len(lineas)
            
            # Calcular puntuación base
            puntuacion_base = 5.0 + min(2.0, positivos * 0.1)
            
            # Ajustar por indentación
            if indentacion_consistente:
                puntuacion_base += 1.0
            
            # Ajustar por longitud de líneas
            puntuacion_base *= factor_longitud
            
            # Ajustar por comentarios
            if ratio_comentarios < 0.05:  # Muy pocos comentarios
                puntuacion_base *= 0.9
            elif ratio_comentarios > 0.4:  # Demasiados comentarios
                puntuacion_base *= 0.95
            else:  # Ratio adecuado
                puntuacion_base += 1.0
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_base))
        except Exception as e:
            self.logger.error(f"Error al evaluar estilo del código: {e}")
            return 5.0  # Valor neutral en caso de error

    def _evaluar_documentacion_codigo(self, codigo):
        """Evalúa la documentación del código."""
        try:
            # Contar comentarios
            lineas = codigo.split('\n')
            lineas = [l for l in lineas if l.strip()]
            
            if not lineas:
                return 5.0  # Valor neutral
                
            # Comentarios de una línea
            comentarios_linea = sum(1 for l in lineas if l.strip().startswith(('#', '//')))
            
            # Docstrings (Python)
            docstrings = len(re.findall(r'""".*?"""|\'\'\'.*?\'\'\'', codigo, re.DOTALL))
            
            # Comentarios de bloque (otros lenguajes)
            comentarios_bloque = len(re.findall(r'/\*.*?\*/', codigo, re.DOTALL))
            
            # Total de comentarios
            total_comentarios = comentarios_linea + docstrings * 3 + comentarios_bloque * 3
            
            # Ratio de comentarios por línea de código
            ratio_comentarios = total_comentarios / len(lineas)
            
            # Detectar patrones de buena documentación
            patrones_buenos = [
                r'@param', r'@return', r'@throws', r':param', r':return', r':raises',
                r'Parameters:', r'Returns:', r'Raises:', r'Example:', r'Usage:'
            ]
            
            buenos_patrones = sum(1 for p in patrones_buenos if re.search(p, codigo))
            
            # Calcular puntuación base
            puntuacion_base = 5.0
            
            # Ajustar por ratio de comentarios
            if ratio_comentarios < 0.05:  # Muy pocos comentarios
                puntuacion_base *= 0.7
            elif ratio_comentarios < 0.1:  # Pocos comentarios
                puntuacion_base *= 0.9
            elif ratio_comentarios > 0.5:  # Demasiados comentarios
                puntuacion_base *= 0.95
            else:  # Ratio adecuado
                puntuacion_base += 2.0
            
            # Ajustar por patrones de buena documentación
            puntuacion_base += min(3.0, buenos_patrones * 0.5)
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_base))
        except Exception as e:
            self.logger.error(f"Error al evaluar documentación del código: {e}")
            return 5.0  # Valor neutral en caso de error

    def _evaluar_cumplimiento_requisitos(self, contenido, titulo, objetivo):
        """Evalúa el cumplimiento de requisitos específicos."""
        try:
            if not titulo and not objetivo:
                return 7.5  # Valor por defecto si no hay requisitos
                
            # Combinar título y objetivo
            requisitos = f"{titulo} {objetivo}".lower()
            
            # Extraer palabras clave de los requisitos
            palabras_clave = set(re.findall(r'\b\w+\b', requisitos))
            palabras_clave = {p for p in palabras_clave if len(p) > 3}  # Filtrar palabras cortas
            
            # Extraer palabras del contenido
            palabras_contenido = set(re.findall(r'\b\w+\b', contenido.lower()))
            
            # Calcular coincidencias
            coincidencias = palabras_clave.intersection(palabras_contenido)
            
            if not palabras_clave:
                return 7.5  # Valor por defecto si no hay palabras clave
                
            # Calcular ratio de coincidencia
            ratio = len(coincidencias) / len(palabras_clave)
            
            # Calcular puntuación base
            puntuacion_base = ratio * 10.0
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_base))
        except Exception as e:
            self.logger.error(f"Error al evaluar cumplimiento de requisitos: {e}")
            return 5.0  # Valor neutral en caso de error

    def _evaluar_seguridad_codigo(self, codigo):
        """Evalúa la seguridad del código."""
        try:
            # Indicadores de problemas de seguridad
            indicadores_inseguridad = [
                'eval(', 'exec(', 'os.system(', 'subprocess.call(',
                'input(', 'pickle.load', 'yaml.load(', 'marshal.loads(',
                '__import__', 'getattr(', 'setattr(', 'globals()[',
                'locals()[', 'open(', 'file(', 'execfile(',
                'sql injection', 'xss', 'csrf', 'cross-site'
            ]
            
            # Indicadores de buenas prácticas de seguridad
            indicadores_seguridad = [
                'sanitize', 'escape', 'validate', 'prepared statement',
                'parameterized', 'whitelist', 'csrf_token', 'hmac',
                'hash', 'encrypt', 'ssl', 'tls',
                'permission', 'authorization', 'authentication', 'verify'
            ]
            
            # Contar indicadores
            inseguros = sum(1 for ind in indicadores_inseguridad if ind in codigo.lower())
            seguros = sum(1 for ind in indicadores_seguridad if ind in codigo.lower())
            
            # Calcular puntuación base
            puntuacion_base = 7.0 - min(7.0, inseguros * 1.0) + min(3.0, seguros * 0.5)
            
            # Verificar patrones específicos de seguridad
            if 'password' in codigo.lower() and ('hash' not in codigo.lower() and 'encrypt' not in codigo.lower()):
                puntuacion_base -= 1.0  # Penalizar contraseñas sin hash/encriptación
            
            if 'sql' in codigo.lower() and 'prepare' not in codigo.lower() and 'bind_param' not in codigo.lower():
                puntuacion_base -= 1.0  # Penalizar SQL sin prepared statements
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, puntuacion_base))
        except Exception as e:
            self.logger.error(f"Error al evaluar seguridad del código: {e}")
            return 5.0  # Valor neutral en caso de error

    def _calcular_calificacion_alternativa(self, metricas, contexto):
        """Calcula una calificación alternativa basada en métricas cuando falla el modelo."""
        try:
            # Verificar si son métricas de código o texto
            if 'funcionalidad' in metricas:
                # Métricas de código
                pesos = {
                    'funcionalidad': 0.3,
                    'eficiencia': 0.2,
                    'estilo': 0.15,
                    'documentacion': 0.15,
                    'cumplimiento': 0.1,
                    'seguridad': 0.1
                }
                
                # Calcular promedio ponderado
                calificacion = sum(metricas[m] * pesos[m] for m in pesos if m in metricas)
                
                # Ajustar según contexto si está disponible
                if contexto and 'nivel_complejidad' in contexto:
                    if contexto['nivel_complejidad'] == 'alto':
                        calificacion *= 1.1  # Bonificación por alta complejidad
                        calificacion = min(10.0, calificacion)  # Limitar a 10
                    elif contexto['nivel_complejidad'] == 'bajo':
                        calificacion *= 0.9  # Penalización por baja complejidad
                
                return calificacion
            else:
                # Métricas de texto
                pesos = {
                    'claridad': 0.25,
                    'profundidad': 0.3,
                    'estructura': 0.25,
                    'relevancia': 0.2
                }
                
                # Calcular promedio ponderado
                calificacion = sum(metricas[m] * pesos[m] for m in pesos if m in metricas)
                
                # Ajustar según contexto si está disponible
                if contexto and 'nivel_complejidad' in contexto:
                    if contexto['nivel_complejidad'] == 'alto':
                        calificacion *= 1.1  # Bonificación por alta complejidad
                        calificacion = min(10.0, calificacion)  # Limitar a 10
                    elif contexto['nivel_complejidad'] == 'bajo':
                        calificacion *= 0.9  # Penalización por baja complejidad
                
                return calificacion
        except Exception as e:
            self.logger.error(f"Error al calcular calificación alternativa: {e}")
            return 5.0  # Valor neutral en caso de error

    def _ajustar_calificacion(self, pred_base, relevancia, claridad, profundidad, estructura, contexto):
        """Ajusta la calificación base según métricas y contexto para contenido textual."""
        try:
            # Convertir relevancia a escala 0-1
            relevancia_norm = min(1.0, relevancia)
            
            # Calcular ajuste por métricas
            ajuste_metricas = (claridad * 0.25 + profundidad * 0.3 + estructura * 0.25) / 10.0
            
            # Calcular calificación ajustada
            calificacion = pred_base * 0.4 + ajuste_metricas * 6.0
            
            # Penalizar por baja relevancia
            if relevancia_norm < 0.5:
                calificacion *= relevancia_norm * 1.5  # Penalización proporcional a la relevancia
            
            # Ajustar según contexto si está disponible
            if contexto and 'nivel_complejidad' in contexto:
                if contexto['nivel_complejidad'] == 'alto':
                    calificacion *= 1.1  # Bonificación por alta complejidad
                    calificacion = min(10.0, calificacion)  # Limitar a 10
                elif contexto['nivel_complejidad'] == 'bajo':
                    calificacion *= 0.9  # Penalización por baja complejidad
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, calificacion))
        except Exception as e:
            self.logger.error(f"Error al ajustar calificación: {e}")
            return pred_base  # Devolver predicción base en caso de error

    def _ajustar_calificacion_codigo(self, pred_base, relevancia, funcionalidad, eficiencia, estilo, documentacion, cumplimiento, contexto, tipo_contenido):
        """Ajusta la calificación base según métricas y contexto para código."""
        try:
            # Convertir relevancia a escala 0-1
            relevancia_norm = min(1.0, relevancia)
            
            # Calcular ajuste por métricas
            ajuste_metricas = (
                funcionalidad * 0.3 +
                eficiencia * 0.2 +
                estilo * 0.15 +
                documentacion * 0.15 +
                cumplimiento * 0.2
            ) / 10.0
            
            # Calcular calificación ajustada
            calificacion = pred_base * 0.3 + ajuste_metricas * 7.0
            
            # Penalizar por baja relevancia
            if relevancia_norm < 0.5:
                calificacion *= relevancia_norm * 1.5  # Penalización proporcional a la relevancia
            
            # Ajustar según contexto si está disponible
            if contexto:
                if 'nivel_complejidad' in contexto:
                    if contexto['nivel_complejidad'] == 'alto':
                        calificacion *= 1.1  # Bonificación por alta complejidad
                        calificacion = min(10.0, calificacion)  # Limitar a 10
                    elif contexto['nivel_complejidad'] == 'bajo':
                        calificacion *= 0.9  # Penalización por baja complejidad
                
                # Ajustar según requisitos específicos
                if 'requisitos_especificos' in contexto and contexto['requisitos_especificos']:
                    # Verificar cumplimiento de requisitos específicos
                    requisitos_cumplidos = sum(1 for req in contexto['requisitos_especificos'] if req.lower() in tipo_contenido.lower())
                    ratio_cumplimiento = requisitos_cumplidos / len(contexto['requisitos_especificos'])
                    
                    # Ajustar calificación según cumplimiento de requisitos específicos
                    if ratio_cumplimiento < 0.5:
                        calificacion *= 0.9  # Penalización por bajo cumplimiento
                    elif ratio_cumplimiento > 0.8:
                        calificacion *= 1.1  # Bonificación por alto cumplimiento
                        calificacion = min(10.0, calificacion)  # Limitar a 10
            
            # Limitar a rango 0-10
            return max(0.0, min(10.0, calificacion))
        except Exception as e:
            self.logger.error(f"Error al ajustar calificación de código: {e}")
            return pred_base  # Devolver predicción base en caso de error

    def _identificar_fortalezas_debilidades(self, contenido, metricas, calificacion, contexto):
        """Identifica fortalezas y debilidades en contenido textual."""
        try:
            fortalezas = []
            debilidades = []
            
            # Identificar fortalezas basadas en métricas
            if metricas['claridad'] >= 8.0:
                fortalezas.append("Excelente claridad en la exposición de ideas.")
            elif metricas['claridad'] >= 7.0:
                fortalezas.append("Buena claridad en la exposición de ideas.")
                
            if metricas['profundidad'] >= 8.0:
                fortalezas.append("Análisis profundo y bien fundamentado del tema.")
            elif metricas['profundidad'] >= 7.0:
                fortalezas.append("Buen nivel de profundidad en el análisis del tema.")
                
            if metricas['estructura'] >= 8.0:
                fortalezas.append("Excelente estructura y organización del contenido.")
            elif metricas['estructura'] >= 7.0:
                fortalezas.append("Buena estructura y organización del contenido.")
                
            if metricas['relevancia'] >= 0.8:
                fortalezas.append("Alto nivel de relevancia respecto al tema solicitado.")
            elif metricas['relevancia'] >= 0.7:
                fortalezas.append("Buen nivel de relevancia respecto al tema solicitado.")
            
            # Identificar debilidades basadas en métricas
            if metricas['claridad'] <= 4.0:
                debilidades.append("Falta claridad en la exposición de ideas.")
            elif metricas['claridad'] <= 5.5:
                debilidades.append("La claridad en la exposición de ideas podría mejorar.")
                
            if metricas['profundidad'] <= 4.0:
                debilidades.append("Falta profundidad en el análisis del tema.")
            elif metricas['profundidad'] <= 5.5:
                debilidades.append("El análisis del tema podría ser más profundo.")
                
            if metricas['estructura'] <= 4.0:
                debilidades.append("La estructura y organización del contenido es deficiente.")
            elif metricas['estructura'] <= 5.5:
                debilidades.append("La estructura y organización del contenido podría mejorar.")
                
            if metricas['relevancia'] <= 0.4:
                debilidades.append("Baja relevancia respecto al tema solicitado.")
            elif metricas['relevancia'] <= 0.6:
                debilidades.append("La relevancia respecto al tema solicitado podría mejorar.")
            
            # Identificar fortalezas adicionales basadas en el contenido
            if re.search(r'\b(bibliografía|referencias|fuentes)\b', contenido.lower()):
                fortalezas.append("Inclusión de referencias bibliográficas o fuentes.")
                
            if re.search(r'\b(gráfico|tabla|figura|diagrama)\b', contenido.lower()):
                fortalezas.append("Uso de elementos visuales para apoyar la explicación.")
                
            if re.search(r'\b(ejemplo|caso|ilustración|aplicación)\b', contenido.lower()):
                fortalezas.append("Inclusión de ejemplos o casos prácticos.")
            
            # Identificar debilidades adicionales basadas en el contenido
            if len(contenido) < 1000:
                debilidades.append("El contenido es demasiado breve para abordar adecuadamente el tema.")
                
            if not re.search(r'\b(conclusión|conclusiones|resumen|síntesis)\b', contenido.lower()):
                debilidades.append("Falta una conclusión o síntesis final.")
                
            if not re.search(r'\b(introducción|contexto|antecedentes)\b', contenido.lower()):
                debilidades.append("Falta una introducción o contextualización del tema.")
            
            # Ajustar según contexto si está disponible
            if contexto and 'tipo_tarea' in contexto:
                if contexto['tipo_tarea'] == 'analisis' and metricas['profundidad'] < 6.0:
                    debilidades.append("El nivel de análisis no es suficiente para el tipo de tarea solicitada.")
                elif contexto['tipo_tarea'] == 'creacion' and not re.search(r'\b(propuesta|diseño|solución|implementación)\b', contenido.lower()):
                    debilidades.append("Falta una propuesta o diseño claro para la tarea de creación solicitada.")
            
            return fortalezas, debilidades
        except Exception as e:
            self.logger.error(f"Error al identificar fortalezas y debilidades: {e}")
            return ["Contenido adecuado"], ["Aspectos a mejorar"]  # Valores por defecto en caso de error

    def _identificar_fortalezas_debilidades_codigo(self, codigo, metricas, calificacion, tipo_contenido, contexto):
        """Identifica fortalezas y debilidades en código."""
        try:
            fortalezas = []
            debilidades = []
            
            # Identificar fortalezas basadas en métricas
            if metricas['funcionalidad'] >= 8.0:
                fortalezas.append("Excelente funcionalidad y cumplimiento de requisitos.")
            elif metricas['funcionalidad'] >= 7.0:
                fortalezas.append("Buena funcionalidad y cumplimiento de requisitos.")
                
            if metricas['eficiencia'] >= 8.0:
                fortalezas.append("Código altamente eficiente y optimizado.")
            elif metricas['eficiencia'] >= 7.0:
                fortalezas.append("Código con buena eficiencia.")
                
            if metricas['estilo'] >= 8.0:
                fortalezas.append("Excelente estilo de codificación y legibilidad.")
            elif metricas['estilo'] >= 7.0:
                fortalezas.append("Buen estilo de codificación y legibilidad.")
                
            if metricas['documentacion'] >= 8.0:
                fortalezas.append("Documentación excelente y completa.")
            elif metricas['documentacion'] >= 7.0:
                fortalezas.append("Buena documentación del código.")
                
            if metricas['seguridad'] >= 8.0:
                fortalezas.append("Implementación con excelentes prácticas de seguridad.")
            elif metricas['seguridad'] >= 7.0:
                fortalezas.append("Buenas prácticas de seguridad en el código.")
            
            # Identificar debilidades basadas en métricas
            if metricas['funcionalidad'] <= 4.0:
                debilidades.append("Problemas significativos en la funcionalidad del código.")
            elif metricas['funcionalidad'] <= 5.5:
                debilidades.append("La funcionalidad del código podría mejorar.")
                
            if metricas['eficiencia'] <= 4.0:
                debilidades.append("Código ineficiente con problemas de rendimiento.")
            elif metricas['eficiencia'] <= 5.5:
                debilidades.append("La eficiencia del código podría optimizarse.")
                
            if metricas['estilo'] <= 4.0:
                debilidades.append("Estilo de codificación deficiente y baja legibilidad.")
            elif metricas['estilo'] <= 5.5:
                debilidades.append("El estilo de codificación y legibilidad podrían mejorar.")
                
            if metricas['documentacion'] <= 4.0:
                debilidades.append("Documentación insuficiente o inexistente.")
            elif metricas['documentacion'] <= 5.5:
                debilidades.append("La documentación del código podría ser más completa.")
                
            if metricas['seguridad'] <= 4.0:
                debilidades.append("Problemas de seguridad significativos en el código.")
            elif metricas['seguridad'] <= 5.5:
                debilidades.append("Las prácticas de seguridad en el código podrían mejorar.")
            
            # Identificar fortalezas adicionales basadas en el código
            if re.search(r'(try|except|catch|finally|error|exception)', codigo.lower()):
                fortalezas.append("Implementación de manejo de errores y excepciones.")
                
            if re.search(r'(test|assert|spec|describe|it\s*\()', codigo.lower()):
                fortalezas.append("Inclusión de pruebas o validaciones.")
                
            if tipo_contenido == 'sql' and re.search(r'(index|índice|constraint|foreign key|primary key)', codigo.lower()):
                fortalezas.append("Uso adecuado de índices y restricciones en SQL.")
            
            # Identificar debilidades adicionales basadas en el código
            if re.search(r'(#|//)\s*(todo|fixme|xxx|pendiente|incompleto)', codigo.lower()):
                debilidades.append("Código incompleto o con tareas pendientes.")
                
            if len(re.findall(r'for.*for|while.*while', codigo.lower())) > 1:
                debilidades.append("Uso excesivo de bucles anidados que pueden afectar el rendimiento.")
                
            if tipo_contenido == 'sql' and 'where' in codigo.lower() and not re.search(r'(index|índice)', codigo.lower()):
                debilidades.append("Consultas SQL sin índices adecuados para optimizar el rendimiento.")
            
            # Ajustar según contexto si está disponible
            if contexto and 'requisitos_especificos' in contexto:
                for requisito in contexto['requisitos_especificos']:
                    if requisito.lower() not in codigo.lower():
                        debilidades.append(f"No se implementa el requisito específico: {requisito}.")
            
            return fortalezas, debilidades
        except Exception as e:
            self.logger.error(f"Error al identificar fortalezas y debilidades de código: {e}")
            return ["Código funcional"], ["Aspectos a mejorar"]  # Valores por defecto en caso de error

    def _generar_comentarios_detallados(self, calificacion, contenido, titulo, objetivo, fortalezas, debilidades, estilo_aprendizaje, contexto):
        """Genera comentarios detallados para contenido textual."""
        try:
            # Determinar categoría de calificación
            categoria = self._determinar_categoria_calificacion(calificacion)
            
            # Seleccionar plantilla base según categoría
            if categoria in self.feedback_templates:
                plantillas = self.feedback_templates[categoria]
                plantilla_base = random.choice(plantillas)
            else:
                plantilla_base = "Tu trabajo ha sido evaluado y se han identificado aspectos positivos y áreas de mejora."
            
            # Construir comentario completo
            comentario = f"{plantilla_base}\n\n"
            
            # Añadir fortalezas
            if fortalezas:
                comentario += "Aspectos destacados:\n"
                for i, fortaleza in enumerate(fortalezas[:3], 1):  # Limitar a 3 fortalezas
                    comentario += f"{i}. {fortaleza}\n"
                comentario += "\n"
            
            # Añadir debilidades
            if debilidades:
                comentario += "Aspectos a mejorar:\n"
                for i, debilidad in enumerate(debilidades[:3], 1):  # Limitar a 3 debilidades
                    comentario += f"{i}. {debilidad}\n"
                comentario += "\n"
            
            # Añadir comentario específico según contexto
            if contexto and 'tipo_tarea' in contexto:
                if contexto['tipo_tarea'] == 'analisis':
                    comentario += "En trabajos de análisis como este, es importante profundizar en los conceptos y establecer conexiones claras entre ideas.\n\n"
                elif contexto['tipo_tarea'] == 'creacion':
                    comentario += "En trabajos de creación como este, es importante mostrar originalidad y fundamentar adecuadamente las decisiones tomadas.\n\n"
                elif contexto['tipo_tarea'] == 'explicacion':
                    comentario += "En trabajos explicativos como este, es fundamental la claridad y la estructura lógica de la exposición.\n\n"
            
            # Añadir comentario específico según estilo de aprendizaje
            if estilo_aprendizaje:
                estilos = [e.strip().lower() for e in estilo_aprendizaje.split(',')]
                
                if 'visual' in estilos:
                    comentario += "Para tu estilo de aprendizaje visual, considera incluir más diagramas, gráficos o mapas conceptuales en tus próximos trabajos.\n"
                
                if 'auditivo' in estilos:
                    comentario += "Para tu estilo de aprendizaje auditivo, considera grabar explicaciones verbales o discutir los conceptos con compañeros.\n"
                
                if 'kinestesico' in estilos or 'kinestésico' in estilos:
                    comentario += "Para tu estilo de aprendizaje kinestésico, considera realizar ejercicios prácticos o experimentos relacionados con el tema.\n"
                
                if 'lectura_escritura' in estilos:
                    comentario += "Para tu estilo de aprendizaje basado en lectura/escritura, considera ampliar tus fuentes bibliográficas y tomar notas detalladas.\n"
                
                comentario += "\n"
            
            # Añadir conclusión
            comentario += f"Calificación: {calificacion:.1f}/10.0\n\n"
            
            if calificacion >= 9.0:
                comentario += "¡Excelente trabajo! Sigue así."
            elif calificacion >= 7.0:
                comentario += "Buen trabajo. Considera las sugerencias para seguir mejorando."
            elif calificacion >= 5.0:
                comentario += "Trabajo aceptable. Implementa las sugerencias para mejorar significativamente."
            else:
                comentario += "Es necesario revisar los aspectos fundamentales del trabajo. Sigue las sugerencias y no dudes en pedir ayuda adicional."
            
            return comentario
        except Exception as e:
            self.logger.error(f"Error al generar comentarios detallados: {e}")
            return f"Calificación: {calificacion:.1f}/10.0. Se han identificado aspectos positivos y áreas de mejora en tu trabajo."

    def _generar_comentarios_detallados_codigo(self, calificacion, codigo, titulo, objetivo, fortalezas, debilidades, estilo_aprendizaje, tipo_contenido, contexto):
        """Genera comentarios detallados para código."""
        try:
            # Determinar categoría de calificación
            categoria = self._determinar_categoria_calificacion(calificacion)
            
            # Seleccionar plantilla base según categoría
            if categoria in self.code_feedback_templates:
                plantillas = self.code_feedback_templates[categoria]
                plantilla_base = random.choice(plantillas)
            else:
                plantilla_base = "Tu código ha sido evaluado y se han identificado aspectos positivos y áreas de mejora."
            
            # Construir comentario completo
            comentario = f"{plantilla_base}\n\n"
            
            # Añadir fortalezas
            if fortalezas:
                comentario += "Aspectos destacados:\n"
                for i, fortaleza in enumerate(fortalezas[:3], 1):  # Limitar a 3 fortalezas
                    comentario += f"{i}. {fortaleza}\n"
                comentario += "\n"
            
            # Añadir debilidades
            if debilidades:
                comentario += "Aspectos a mejorar:\n"
                for i, debilidad in enumerate(debilidades[:3], 1):  # Limitar a 3 debilidades
                    comentario += f"{i}. {debilidad}\n"
                comentario += "\n"
            
            # Añadir comentario específico según tipo de contenido
            if tipo_contenido == 'sql':
                comentario += "En código SQL, es importante prestar atención a la optimización de consultas, el uso adecuado de índices y la normalización de tablas.\n\n"
            else:
                comentario += "En desarrollo de software, es importante mantener un equilibrio entre funcionalidad, eficiencia, legibilidad y mantenibilidad del código.\n\n"
            
            # Añadir comentario específico según contexto
            if contexto and 'enfoque_principal' in contexto:
                if contexto['enfoque_principal'] == 'programacion':
                    comentario += "Para este tipo de tarea de programación, es fundamental aplicar buenas prácticas de desarrollo y patrones de diseño adecuados.\n\n"
                elif contexto['enfoque_principal'] == 'base_de_datos':
                    comentario += "Para este tipo de tarea de bases de datos, es esencial considerar la integridad de los datos y la eficiencia de las consultas.\n\n"
            
            # Añadir comentario específico según estilo de aprendizaje
            if estilo_aprendizaje:
                estilos = [e.strip().lower() for e in estilo_aprendizaje.split(',')]
                
                if 'visual' in estilos:
                    comentario += "Para tu estilo de aprendizaje visual, considera utilizar diagramas de flujo o UML para planificar tu código antes de implementarlo.\n"
                
                if 'auditivo' in estilos:
                    comentario += "Para tu estilo de aprendizaje auditivo, considera explicar verbalmente tu código o discutir algoritmos con compañeros.\n"
                
                if 'kinestesico' in estilos or 'kinestésico' in estilos:
                    comentario += "Para tu estilo de aprendizaje kinestésico, considera realizar más ejercicios prácticos de programación o participar en hackathons.\n"
                
                if 'lectura_escritura' in estilos:
                    comentario += "Para tu estilo de aprendizaje basado en lectura/escritura, considera documentar exhaustivamente tu código y leer libros sobre patrones de diseño.\n"
                
                comentario += "\n"
            
            # Añadir conclusión
            comentario += f"Calificación: {calificacion:.1f}/10.0\n\n"
            
            if calificacion >= 9.0:
                comentario += "¡Excelente código! Sigue aplicando estas buenas prácticas en tus futuros desarrollos."
            elif calificacion >= 7.0:
                comentario += "Buen código. Considera las sugerencias para seguir mejorando tu desarrollo."
            elif calificacion >= 5.0:
                comentario += "Código aceptable. Implementa las sugerencias para mejorar significativamente la calidad."
            else:
                comentario += "Es necesario revisar los aspectos fundamentales del código. Sigue las sugerencias y no dudes en pedir ayuda adicional."
            
            return comentario
        except Exception as e:
            self.logger.error(f"Error al generar comentarios detallados de código: {e}")
            return f"Calificación: {calificacion:.1f}/10.0. Se han identificado aspectos positivos y áreas de mejora en tu código."

    def _generar_sugerencias_mejora(self, calificacion, contenido, titulo, objetivo, debilidades, estilo_aprendizaje, contexto):
        """Genera sugerencias de mejora para contenido textual."""
        try:
            sugerencias = []
            
            # Generar sugerencias basadas en debilidades
            for debilidad in debilidades:
                if "claridad" in debilidad.lower():
                    sugerencias.append("Mejora la claridad utilizando ejemplos concretos y explicaciones más detalladas.")
                elif "profundidad" in debilidad.lower():
                    sugerencias.append("Profundiza en el análisis investigando más fuentes y estableciendo conexiones entre conceptos.")
                elif "estructura" in debilidad.lower():
                    sugerencias.append("Mejora la estructura organizando el contenido en secciones claramente definidas con introducción, desarrollo y conclusión.")
                elif "relevancia" in debilidad.lower():
                    sugerencias.append("Aumenta la relevancia centrándote específicamente en los aspectos mencionados en el título y objetivo de la actividad.")
                elif "breve" in debilidad.lower():
                    sugerencias.append("Desarrolla más el contenido ampliando cada sección con información relevante y análisis más detallado.")
                elif "conclusión" in debilidad.lower():
                    sugerencias.append("Añade una conclusión que sintetice las ideas principales y presente reflexiones finales sobre el tema.")
                elif "introducción" in debilidad.lower():
                    sugerencias.append("Incluye una introducción que contextualice el tema y presente los objetivos del trabajo.")
            
            # Añadir sugerencias generales según calificación
            if calificacion < 5.0:
                sugerencias.append("Revisa los conceptos fundamentales del tema y asegúrate de comprenderlos correctamente.")
                sugerencias.append("Consulta fuentes adicionales para ampliar tu conocimiento sobre el tema.")
            elif calificacion < 7.0:
                sugerencias.append("Mejora la organización de las ideas para facilitar la comprensión del contenido.")
                sugerencias.append("Incluye más ejemplos o casos prácticos para ilustrar los conceptos teóricos.")
            
            # Añadir sugerencias específicas según contexto
            if contexto and 'tipo_tarea' in contexto:
                if contexto['tipo_tarea'] == 'analisis' and calificacion < 8.0:
                    sugerencias.append("Profundiza en el análisis crítico estableciendo comparaciones y evaluando diferentes perspectivas.")
                elif contexto['tipo_tarea'] == 'creacion' and calificacion < 8.0:
                    sugerencias.append("Desarrolla más la justificación de tu propuesta o diseño, explicando las decisiones tomadas.")
                elif contexto['tipo_tarea'] == 'explicacion' and calificacion < 8.0:
                    sugerencias.append("Mejora la claridad de las explicaciones utilizando analogías o representaciones visuales cuando sea posible.")
            
            # Añadir sugerencias específicas según estilo de aprendizaje
            if estilo_aprendizaje:
                estilos = [e.strip().lower() for e in estilo_aprendizaje.split(',')]
                
                if 'visual' in estilos and calificacion < 9.0:
                    sugerencias.append("Incorpora elementos visuales como diagramas, gráficos o mapas conceptuales para reforzar las ideas principales.")
                
                if 'auditivo' in estilos and calificacion < 9.0:
                    sugerencias.append("Practica explicando verbalmente los conceptos para mejorar tu comprensión y capacidad de comunicación.")
                
                if ('kinestesico' in estilos or 'kinestésico' in estilos) and calificacion < 9.0:
                    sugerencias.append("Realiza ejercicios prácticos relacionados con el tema para consolidar tu aprendizaje.")
                
                if 'lectura_escritura' in estilos and calificacion < 9.0:
                    sugerencias.append("Amplía tus fuentes bibliográficas y toma notas detalladas para mejorar tu comprensión del tema.")
            
            # Limitar a 5 sugerencias para no abrumar
            sugerencias = sugerencias[:5]
            
            # Formatear sugerencias
            if sugerencias:
                return "Sugerencias para mejorar:\n\n" + "\n".join(f"- {s}" for s in sugerencias)
            else:
                return "Continúa con tu buen trabajo y sigue profundizando en el tema."
        except Exception as e:
            self.logger.error(f"Error al generar sugerencias de mejora: {e}")
            return "Revisa los aspectos mencionados en las debilidades y consulta fuentes adicionales para mejorar tu trabajo."

    def _recomendar_recursos(self, estilo_aprendizaje, titulo, objetivo, calificacion, tipo_contenido):
        """Recomienda recursos educativos según el estilo de aprendizaje y el tema."""
        recursos_recomendados = []

        # Determinar la categoría temática
        categoria = self._determinar_categoria_tematica(titulo, objetivo)

        # Si no se especifica estilo de aprendizaje, recomendar recursos variados
        if not estilo_aprendizaje:
            # Recomendar recursos generales según la categoría
            if categoria == 'programacion':
                recursos_recomendados.extend([
                    {"titulo": "Documentación oficial del lenguaje", "url": "https://docs.python.org/"},
                    {"titulo": "Ejercicios prácticos de programación", "url": "https://www.hackerrank.com/"},
                    {"titulo": "Tutoriales interactivos", "url": "https://www.codecademy.com/"}
                ])
            elif categoria == 'matematicas':
                recursos_recomendados.extend([
                    {"titulo": "Khan Academy - Matemáticas", "url": "https://www.khanacademy.org/math"},
                    {"titulo": "Visualizaciones de conceptos matemáticos", "url": "https://www.geogebra.org/"},
                    {"titulo": "Ejercicios interactivos", "url": "https://www.mathsisfun.com/"}
                ])
            elif categoria == 'ciencias':
                recursos_recomendados.extend([
                    {"titulo": "Khan Academy - Ciencias", "url": "https://www.khanacademy.org/science"},
                    {"titulo": "Simulaciones interactivas", "url": "https://phet.colorado.edu/es/simulations/category/physics"},
                    {"titulo": "Artículos científicos accesibles", "url": "https://www.sciencedaily.com/"}
                ])
            elif categoria == 'bases_de_datos':
                recursos_recomendados.extend([
                    {"titulo": "Tutoriales de SQL", "url": "https://www.w3schools.com/sql/"},
                    {"titulo": "Ejercicios prácticos de SQL", "url": "https://www.hackerrank.com/domains/sql"},
                    {"titulo": "Diagramas ER interactivos", "url": "https://dbdiagram.io/"}
                ])
            else:
                recursos_recomendados.extend([
                    {"titulo": "Khan Academy", "url": "https://www.khanacademy.org/"},
                    {"titulo": "Coursera", "url": "https://www.coursera.org/"},
                    {"titulo": "edX", "url": "https://www.edx.org/"}
                ])
        else:
            # Recomendar recursos específicos según el estilo de aprendizaje
            estilos = [e.strip().lower() for e in estilo_aprendizaje.split(',')]

            for estilo in estilos:
                if estilo in self.recursos_educativos and categoria in self.recursos_educativos[estilo]:
                    # Añadir recursos específicos para este estilo y categoría
                    recursos_estilo = self.recursos_educativos[estilo][categoria]

                    # Si la calificación es baja, recomendar más recursos básicos
                    if calificacion < 5.0:
                        recursos_recomendados.extend(recursos_estilo[:2])  # Primeros recursos (más básicos)
                    else:
                        # Si la calificación es alta, recomendar recursos más avanzados
                        recursos_recomendados.extend(recursos_estilo[-2:])  # Últimos recursos (más avanzados)

        # Añadir recursos específicos según el tipo de contenido
        if tipo_contenido == 'codigo':
            recursos_recomendados.append({"titulo": "Clean Code: A Handbook of Agile Software Craftsmanship", "url": "https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882"})
        elif tipo_contenido == 'sql':
            recursos_recomendados.append({"titulo": "SQL Performance Explained", "url": "https://sql-performance-explained.com/"})
        elif tipo_contenido == 'matematico':
            recursos_recomendados.append({"titulo": "3Blue1Brown - Visualizaciones matemáticas", "url": "https://www.3blue1brown.com/"})

        # Limitar a 5 recursos como máximo
        return recursos_recomendados[:5]
import os
import sys
import shutil
import logging
import argparse
import subprocess
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def crear_backup(archivo, directorio_backup):
    """Crea una copia de seguridad del archivo."""
    if not os.path.exists(archivo):
        logger.warning(f"El archivo {archivo} no existe, no se puede crear backup.")
        return False
    
    # Crear directorio de backup si no existe
    os.makedirs(directorio_backup, exist_ok=True)
    
    # Nombre del archivo de backup con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    nombre_base = os.path.basename(archivo)
    backup_path = os.path.join(directorio_backup, f"{nombre_base}.{timestamp}.bak")
    
    # Copiar archivo
    try:
        shutil.copy2(archivo, backup_path)
        logger.info(f"Backup creado: {backup_path}")
        return True
    except Exception as e:
        logger.error(f"Error al crear backup de {archivo}: {e}")
        return False

def instalar_dependencias():
    """Instala las dependencias necesarias."""
    dependencias = [
        "torch",
        "numpy",
        "scikit-learn",
        "sentence-transformers",
        "matplotlib",
        "nltk"
    ]
    
    logger.info("Instalando dependencias...")
    
    for dep in dependencias:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            logger.info(f"Dependencia instalada: {dep}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error al instalar {dep}: {e}")
            return False
    
    return True

def integrar_archivos(archivos_origen, directorio_destino, directorio_backup):
    """Integra los archivos mejorados en el sistema."""
    for archivo_origen in archivos_origen:
        if not os.path.exists(archivo_origen):
            logger.error(f"El archivo origen {archivo_origen} no existe.")
            continue
        
        nombre_archivo = os.path.basename(archivo_origen)
        archivo_destino = os.path.join(directorio_destino, nombre_archivo)
        
        # Crear backup si el archivo destino existe
        if os.path.exists(archivo_destino):
            crear_backup(archivo_destino, directorio_backup)
        
        # Copiar archivo mejorado
        try:
            shutil.copy2(archivo_origen, archivo_destino)
            logger.info(f"Archivo integrado: {archivo_destino}")
        except Exception as e:
            logger.error(f"Error al integrar {archivo_origen}: {e}")

def configurar_gpu():
    """Configura el sistema para usar GPU si está disponible."""
    try:
        import torch
        
        if torch.cuda.is_available():
            logger.info(f"GPU disponible: {torch.cuda.get_device_name(0)}")
            logger.info(f"Memoria GPU: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            
            # Configurar para usar GPU por defecto
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            logger.info("PyTorch configurado para usar GPU por defecto")
            
            return True
        else:
            logger.warning("No se detectó GPU compatible con CUDA")
            return False
    except Exception as e:
        logger.error(f"Error al configurar GPU: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Integra las mejoras del modelo ML en el sistema existente")
    parser.add_argument("--destino", default=".", help="Directorio de destino para los archivos")
    parser.add_argument("--backup", default="./backups", help="Directorio para backups")
    parser.add_argument("--no-deps", action="store_true", help="No instalar dependencias")
    parser.add_argument("--no-gpu", action="store_true", help="No configurar GPU")
    
    args = parser.parse_args()
    
    logger.info("Iniciando integración de mejoras...")
    
    # Crear directorio de backup
    os.makedirs(args.backup, exist_ok=True)
    
    # Instalar dependencias si es necesario
    if not args.no_deps:
        if not instalar_dependencias():
            logger.warning("Algunas dependencias no pudieron instalarse")
    
    # Configurar GPU si está disponible
    if not args.no_gpu:
        configurar_gpu()
    
    # Archivos a integrar
    archivos_origen = [
        "modelo_ml_enhanced.py",
        "entrenamiento_automatico.py"
    ]
    
    # Integrar archivos
    for archivo_origen in archivos_origen:
        archivo_destino = os.path.join(args.destino, os.path.basename(archivo_origen))
        if os.path.abspath(archivo_origen) == os.path.abspath(archivo_destino):
            logger.warning(f"'{archivo_origen}' and '{archivo_destino}' are the same file")
            continue
        
        if os.path.exists(archivo_destino):
            crear_backup(archivo_destino, args.backup)
        
        try:
            shutil.copy2(archivo_origen, archivo_destino)
            logger.info(f"Archivo integrado: {archivo_destino}")
        except Exception as e:
            logger.error(f"Error al integrar {archivo_origen}: {e}")
    
    # Actualizar modelo_selector.py para usar el modelo mejorado
    selector_path = os.path.join(args.destino, "modelo_selector.py")
    if os.path.exists(selector_path):
        crear_backup(selector_path, args.backup)
        
        with open(selector_path, 'w') as f:
            f.write("""try:
    from modelo_ml_enhanced import EnhancedModeloEvaluacionInteligente as GeneradorPracticasML
    print("Usando modelo mejorado (EnhancedModeloEvaluacionInteligente)")
except ImportError:
    try:
        from modelo_ml_enhanced import EnhancedModeloEvaluacionInteligente as GeneradorPracticasML
        print("Usando modelo original mejorado (EnhancedModeloEvaluacionInteligente)")
    except ImportError:
        from modelo_ml_scikit import ModeloEvaluacionInteligente as GeneradorPracticasML
        print("Usando modelo original (ModeloEvaluacionInteligente)")
""")
        logger.info("Archivo modelo_selector.py actualizado")
    
    logger.info("Integración completada con éxito")
    logger.info("Para entrenar el modelo automáticamente, ejecute: python entrenamiento_automatico.py")

if __name__ == "__main__":
    main()
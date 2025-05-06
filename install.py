#!/usr/bin/env python
# Script de instalación para el modelo mejorado

import os
import sys
import shutil
import subprocess

def instalar_dependencias():
    """Instala las dependencias necesarias para el modelo mejorado."""
    print("Instalando dependencias...")
    
    # Lista de dependencias
    dependencias = [
        "torch",
        "nltk",
        "scikit-learn",
        "numpy",
        "sentence-transformers"  # Opcional, para análisis semántico profundo
    ]
    
    # Instalar cada dependencia
    for dep in dependencias:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
            print(f"✓ {dep} instalado correctamente")
        except subprocess.CalledProcessError:
            print(f"✗ Error al instalar {dep}")
            if dep == "sentence-transformers":
                print("  La dependencia sentence-transformers es opcional. El modelo funcionará sin ella.")
            elif dep == "torch":
                print("  Intente instalar PyTorch manualmente siguiendo las instrucciones en https://pytorch.org/")
                return False
    
    return True

def copiar_archivos():
    """Copia los archivos del modelo mejorado al directorio del proyecto."""
    print("\nCopiando archivos del modelo mejorado...")
    
    # Verificar si los archivos existen
    if not os.path.exists("modelo_ml_enhanced.py"):
        print("✗ Error: No se encontró el archivo modelo_ml_enhanced.py")
        return False
    
    if not os.path.exists("integration_script.py"):
        print("✗ Error: No se encontró el archivo integration_script.py")
        return False
    
    # Crear backup del modelo actual si existe
    if os.path.exists("modelo_ml_scikit.py"):
        backup_path = "modelo_ml_scikit.py.bak"
        shutil.copy2("modelo_ml_scikit.py", backup_path)
        print(f"✓ Backup del modelo actual creado en {backup_path}")
    
    # Preguntar si se desea reemplazar el modelo actual
    reemplazar = input("\n¿Desea reemplazar el modelo actual? (s/n): ").lower() == 's'
    
    if reemplazar:
        # Copiar el modelo mejorado sobre el actual
        shutil.copy2("modelo_ml_enhanced.py", "modelo_ml_scikit.py")
        print("✓ Modelo mejorado instalado como modelo_ml_scikit.py")
    else:
        print("✓ Modelo mejorado disponible como modelo_ml_enhanced.py")
    
    return True

def crear_directorio_modelos():
    """Crea el directorio para almacenar los modelos entrenados."""
    print("\nCreando directorio para modelos...")
    
    os.makedirs("modelos", exist_ok=True)
    print("✓ Directorio 'modelos' creado/verificado")
    
    return True

def main():
    print("=== Instalación del Modelo Mejorado ===\n")
    
    # Instalar dependencias
    if not instalar_dependencias():
        print("\n✗ Error al instalar dependencias. Instalación cancelada.")
        return
    
    # Copiar archivos
    if not copiar_archivos():
        print("\n✗ Error al copiar archivos. Instalación cancelada.")
        return
    
    # Crear directorio de modelos
    if not crear_directorio_modelos():
        print("\n✗ Error al crear directorio de modelos. Instalación cancelada.")
        return
    
    print("\n✓ Instalación completada con éxito!")
    print("\nPara integrar completamente el modelo mejorado, ejecute:")
    print("  python integration_script.py")
    
    # Preguntar si se desea ejecutar el script de integración
    ejecutar = input("\n¿Desea ejecutar el script de integración ahora? (s/n): ").lower() == 's'
    
    if ejecutar:
        print("\nEjecutando script de integración...")
        subprocess.call([sys.executable, "integration_script.py"])

if __name__ == "__main__":
    main()
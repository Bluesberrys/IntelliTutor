try:
    from modelo_ml_enhanced import EnhancedModeloEvaluacionInteligente as GeneradorPracticasML
    print("Usando modelo mejorado (EnhancedModeloEvaluacionInteligente)")
except ImportError:
    try:
        from modelo_ml_enhanced import EnhancedModeloEvaluacionInteligente as GeneradorPracticasML
        print("Usando modelo original mejorado (EnhancedModeloEvaluacionInteligente)")
    except ImportError:
        from modelo_ml_scikit import ModeloEvaluacionInteligente as GeneradorPracticasML
        print("Usando modelo original (ModeloEvaluacionInteligente)")

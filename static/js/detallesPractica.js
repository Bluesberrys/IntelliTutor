function mostrarDetalles() {
    return `
      <main id="ver-practica">
      <div class="practica-content">
        <h1 class="practica-title">Visualización de practicas</h1>
        <h2><span>{{ practica.titulo }}</span></h2>
        <div>
          <h4><span>Materia:</span> {{ practica.nombre_materia }}</h4>
          <h4><span>Fecha de Entrega:</span> {{ practica.fecha_entrega.strftime('%Y-%m-%d') }}</h4>
          <h4><span>Estado:</span> {{ practica.estado }}</h4>
        </div>
        <div>
          <h3><span>Descripción</span></h3>
          <p>{{ contenido.descripcion }}</p>
        </div>
        <div>
          <h3><span>Objetivos Específicos</span></h3>
          <ul>
            {% for objetivo in contenido.objetivos_especificos %}
            <li>{{ objetivo }}</li>
            {% endfor %}
          </ul>
        </div>
        <div>
          <h3><span>Actividades</span></h3>
          <ol>
            {% for actividad in contenido.actividades %}
            <li>{{ actividad.descripcion }} ({{ actividad.tiempo_estimado }})</li>
            {% endfor %}
          </ol>
        </div>
        <div>
          <h3><span>Recursos</span></h3>
          <ul>
            {% for recurso in contenido.recursos %}
            <li>{{ recurso }}</li>
            {% endfor %}
          </ul>
        </div>
        <div>
          <h3><span>Criterios de Evaluación</span></h3>
          <ul>
            {% for criterio in contenido.criterios_evaluacion %}
            <li>{{ criterio }}</li>
            {% endfor %}
          </ul>
        </div>
        <div>
          <h3><span>Recomendaciones</span></h3>
          <p>{{ contenido.recomendaciones }}</p>
        </div>
        <h3>Datos Generados por IA</h3>
        <p><span>Predicción de Éxito:</span> {{ modelo_ml.prediccion_exito }}</p>
        <p><span>Recomendaciones Personalizadas:</span> {{ modelo_ml.recomendaciones_personalizadas }}</p>
      </div>
    </main>
    `;
    }
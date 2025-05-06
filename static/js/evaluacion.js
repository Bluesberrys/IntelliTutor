document.addEventListener('DOMContentLoaded', function() {
    const burgerMenu = document.querySelector('.burger-menu');
    const menu = document.querySelector('.menu');
    
    if (burgerMenu) {
      burgerMenu.addEventListener('click', function() {
        menu.classList.toggle('active');
      });
    }
    
    // Mostrar todas las materias al inicio
    // document.querySelectorAll('.materia-section').forEach(section => {
    //   section.style.display = 'block';
    //   animateHeader(section);
    // });
    
    // Inicializar filtros
    initFiltros();
    
    // Inicializar búsqueda
    initBusqueda();
    
    // Inicializar selector de grupo
    initSelectorGrupo();
  });
  
  
  // Función para cargar las evaluaciones de una práctica
  function loadEvaluaciones(practicaId) {
console.log(`Cargando evaluaciones para la práctica ID: ${practicaId}`);
const evaluacionesGrid = document.getElementById('evaluacionesGrid');

// Limpiar el contenedor
evaluacionesGrid.innerHTML = '';

// Obtener las evaluaciones de la práctica desde el backend
fetch(`/api/evaluaciones/${practicaId}`)
  .then(response => {
    if (!response.ok) {
      throw new Error('Error al cargar evaluaciones');
    }
    return response.json();
  })
  .then(data => {
    console.log('Datos de evaluaciones recibidos:', data);
    if (data.evaluaciones && data.evaluaciones.length > 0) {
      // Renderizar cada evaluación
      data.evaluaciones.forEach(evaluacion => {
        const evaluacionHTML = createEvaluacionCard(evaluacion);
        evaluacionesGrid.innerHTML += evaluacionHTML;
      });
      
      // Inicializar los eventos de flip para las tarjetas
      initFlipCards();
    } else {
      evaluacionesGrid.innerHTML = `
        <div class="no-evaluaciones">
          <i class="fas fa-exclamation-circle"></i>
          <p>No hay evaluaciones disponibles para esta práctica.</p>
        </div>
      `;
    }
  })
  .catch(error => {
    console.error('Error al cargar evaluaciones:', error);
    evaluacionesGrid.innerHTML = `
      <div class="no-evaluaciones">
        <i class="fas fa-exclamation-triangle"></i>
        <p>Error al cargar las evaluaciones. Por favor, intenta de nuevo.</p>
      </div>
    `;
  });
}
  
  // Función para crear el HTML de una tarjeta de evaluación
  function createEvaluacionCard(evaluacion) {
return `
  <div class="card-container">
    <div class="evaluacion-card" data-estado="${evaluacion.calificacion ? 'calificado' : evaluacion.estado}" id="card-${evaluacion.estudiante_id}">
      <!-- Frente de la tarjeta -->
      <div class="card-front">
        <div class="evaluacion-card-header">
          <h3>
            ${evaluacion.estado === 'pendiente' ? 'Esperando Entrega' : 
             evaluacion.estado === 'entregado' && evaluacion.archivo_url ? 'Esperando Evaluación' :
             evaluacion.estado === 'calificado' ? `Calificado: ${evaluacion.calificacion}` :
             'Sin archivo disponible'}
          </h3>
          ${evaluacion.estado === 'entregado' && evaluacion.archivo_url ? 
            `<a href="/static/uploads/${evaluacion.archivo_url}" target="_blank" class="archivo-link-header ver_archivo">Ver archivo</a>` : ''}
          <button class="btn-flip" onclick="flipCard(event, '${evaluacion.estudiante_id}')">
            <i class="fas fa-sync-alt"></i>
          </button>
        </div>
        <div class="evaluacion-card-body">
          <div class="evaluacion-info">
            <div class="evaluacion-info-item">
              <i class="fas fa-user"></i>
              <span><strong>Estudiante:</strong> ${evaluacion.estudiante_nombre}</span>
            </div>
            <div class="evaluacion-info-item">
              <i class="fas fa-calendar-alt"></i>
              <span><strong>Fecha de Entrega:</strong> ${new Date(evaluacion.fecha_entrega).toLocaleDateString()}</span>
            </div>
            <span class="estado estado-${evaluacion.estado}">${evaluacion.estado}</span>
          </div>
        </div>
      </div>
      
      <!-- Reverso de la tarjeta -->
      <div class="card-back">
        <div class="evaluacion-card-back-header">
          <h3>Detalles de Evaluación</h3>
          <button class="btn-flip btn-flip-back" onclick="flipCardBack(event, '${evaluacion.estudiante_id}')">
            <i class="fas fa-undo"></i>
          </button>
        </div>
        <div class="evaluacion-card-body">
          <div class="evaluacion-info">
            <div class="evaluacion-info-item">
              <i class="fas fa-user"></i>
              <span><strong>Estudiante:</strong> ${evaluacion.estudiante_nombre}</span>
            </div>
            ${evaluacion.archivo_url ? `
            <div class="evaluacion-info-item">
              <i class="fas fa-file"></i>
              <span>
                <a href="/static/uploads/${evaluacion.archivo_url}" target="_blank" class="archivo-link">
                  Ver documento
                </a>
              </span>
            </div>
            ` : ''}
          </div>

          ${evaluacion.calificacion ? `
            <div class="comentarios">
              <p><strong>Calificación:</strong> ${evaluacion.calificacion}</p>
              ${evaluacion.comentarios ? `<p><strong>Comentarios:</strong> ${evaluacion.comentarios}</p>` : ''}
            </div>
          ` : evaluacion.estado === 'entregado' ? `
            ${evaluacion.uso_ia == 0 ? `
              <!-- Formulario de calificación manual en el reverso -->
              <form class="calificacion-form" onsubmit="return calificarManualmente(event, '${evaluacion.evaluacion_id}')">
                <input type="hidden" name="evaluacion_id" value="${evaluacion.evaluacion_id}">
                
                <div>
                  <label for="calificacion-${evaluacion.evaluacion_id}">Calificación (1-10):</label>
                  <input type="number" id="calificacion-${evaluacion.evaluacion_id}" name="calificacion" min="1" max="10" required>
                </div>
                
                <div>
                  <label for="comentarios-${evaluacion.evaluacion_id}">Comentarios:</label>
                  <textarea id="comentarios-${evaluacion.evaluacion_id}" name="comentarios" required></textarea>
                </div>
                
                <button type="submit" class="btn-evaluacion" id="btn-calificar-${evaluacion.evaluacion_id}">
                  <i class="fas fa-save"></i> Guardar calificación
                </button>
              </form>
            ` : `
              <div class="card-actions">
                <button onclick="evaluarEntregaAutomatica('${evaluacion.evaluacion_id}')"  class="btn-evaluacion btn-ia" id="btn-ia-${evaluacion.evaluacion_id}"><i class="fas fa-robot"></i> Calificar Automáticamente</button>
              </div>
            `}
          ` : `
            <div class="comentarios">
              <p>Esta entrega aún no ha sido evaluada.</p>
            </div>
          `}
        </div>
      </div>
    </div>
  </div>
`;
}
  
  // Inicializar eventos de flip para las tarjetas
  function initFlipCards() {
    // Prevenir que los enlaces dentro de las tarjetas activen el volteo
    document.querySelectorAll('.archivo-link, .archivo-link-header').forEach(link => {
      link.addEventListener('click', function(e) {
        e.stopPropagation();
      });
    });
  }
  
  function flipCard(event, id) {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    const card = document.getElementById(`card-${id}`);
    if (card && !card.classList.contains('flipped')) {
      card.classList.add('flipping');
      requestAnimationFrame(() => {
        card.classList.add('flipped');
        card.classList.remove('flipping');
      });
    }
  }

  function flipCardBack(event, id) {
    if (event) {
      event.preventDefault();
      event.stopPropagation();
    }
    const card = document.getElementById(`card-${id}`);
    if (card && card.classList.contains('flipped')) {
      card.classList.add('flipping');
      requestAnimationFrame(() => {
        card.classList.remove('flipped');
        card.classList.remove('flipping');
      });
    }
  }
  
  // Función para filtrar prácticas por estado
  function initFiltros() {
    const filtros = document.querySelectorAll('.filtro-btn');
    
    filtros.forEach(filtro => {
      filtro.addEventListener('click', function() {
        // Desactivar todos los filtros
        filtros.forEach(f => f.classList.remove('active'));
        
        // Activar el filtro seleccionado
        this.classList.add('active');
        
        const tipoFiltro = this.getAttribute('data-filter');
        filtrarPorEstado(tipoFiltro);
      });
    });
    
    // Activar el filtro "todos" por defecto
    filtrarPorEstado('todos');
  }
  
  // Función para filtrar por estado
  function filtrarPorEstado(estado) {
    // Primero mostrar todas las materias
    // document.querySelectorAll('.materia-section').forEach(seccion => {
    //   seccion.style.display = 'block';
    // });
    
    // Filtrar prácticas según el estado seleccionado
    document.querySelectorAll('.practica-card').forEach(card => {
      const practicaId = card.getAttribute('data-practica-id');
      
      // Aquí deberías implementar la lógica para determinar si una práctica tiene evaluaciones en el estado seleccionado
      // Por ahora, mostraremos todas las prácticas para todos los estados
      if (estado === 'todos') {
        card.style.display = 'block';
      } else {
        // Aquí deberías hacer una petición al backend o usar datos precargados para filtrar
        // Por ahora, simplemente mostraremos todas
        card.style.display = 'block';
      }
    });
    
    // Mostrar/ocultar secciones de materias según si tienen resultados visibles
    document.querySelectorAll('.materia-section').forEach(seccion => {
      const practicasVisibles = Array.from(seccion.querySelectorAll('.practica-card')).filter(card => 
        window.getComputedStyle(card).display !== 'none'
      ).length;
      
      if (practicasVisibles === 0) {
        seccion.style.display = 'none';
      } else {
        seccion.style.display = 'block';
      }
    });
  }
  
  // Función para buscar prácticas
  function initBusqueda() {
    const busquedaInput = document.getElementById('busquedaPractica');
    
    busquedaInput.addEventListener('input', function() {
      const termino = this.value.toLowerCase().trim();
      
      // Primero mostrar todas las materias
      document.querySelectorAll('.materia-section').forEach(seccion => {
        seccion.style.display = 'block';
      });
      
      document.querySelectorAll('.practica-card').forEach(card => {
        const cardContent = card.textContent.toLowerCase();
        
        if (termino.length === 0 || cardContent.includes(termino)) {
          card.style.display = 'block';
        } else {
          card.style.display = 'none';
        }
      });
      
      // Mostrar/ocultar secciones de materias según si tienen resultados visibles
      document.querySelectorAll('.materia-section').forEach(seccion => {
        const practicasVisibles = Array.from(seccion.querySelectorAll('.practica-card')).filter(card => 
          window.getComputedStyle(card).display !== 'none'
        ).length;
        
        if (practicasVisibles === 0) {
          seccion.style.display = 'none';
        } else {
          seccion.style.display = 'block';
        }
      });
    });
  }
  
  // Función para filtrar por grupo
  function initSelectorGrupo() {
    const selectorGrupo = document.getElementById('selectorGrupo');
    const seccionesMateria = document.querySelectorAll('.materia-section');
  
    // Mostrar solo la seleccionada al inicio
    const selectedMateriaId = selectorGrupo.value;
    seccionesMateria.forEach(section => {
      const materiaId = section.getAttribute('data-materia');
      section.style.display = (materiaId === selectedMateriaId) ? 'block' : 'none';
    });
  
    selectorGrupo.addEventListener('change', function () {
      const selectedValue = this.value;
  
      const currentVisibleSection = Array.from(seccionesMateria).find(
        section => section.style.display !== 'none'
      );
  
      const newSection = Array.from(seccionesMateria).find(
        section => section.getAttribute('data-materia') === selectedValue
      );
  
      if (!newSection || currentVisibleSection === newSection) return;
  
      // NUEVO: Animar salida de tarjetas actuales antes de ocultar la sección
      const cardsToExit = currentVisibleSection.querySelectorAll('.practica-card');
      cardsToExit.forEach((card, i) => {
        card.classList.remove('entrando', 'saliendo');
        void card.offsetWidth;
        setTimeout(() => {
          card.classList.add('saliendo');
        }, i * 30);
      });
  
      // Animar salida del header actual y luego cambiar sección
      const currentHeader = currentVisibleSection.querySelector('.materia-header .word');
      const letrasActuales = Array.from(currentHeader.querySelectorAll('.letter'));
  
      animateOut(letrasActuales, () => {
        currentVisibleSection.style.display = 'none';
        newSection.style.display = 'block';
        animateHeader(newSection); // ← Esto ya incluye scroll, letras y tarjetas nuevas
      });
    });
  }
  
// Función para evaluar una entrega automáticamente
function evaluarEntregaAutomatica(entregaId) {
const btnIA = document.getElementById(`btn-ia-${entregaId}`);
    btnIA.innerHTML = '<span class="loading-spinner"></span> Evaluando...';
    btnIA.disabled = true;
  fetch('/evaluar_entrega_automatica', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ evaluacion_id: entregaId }) // Asegúrate de que el ID se envíe correctamente
  })
  .then(response => response.json())
  .then(data => {
      if (data.success) {
        showNotification('Evaluación con IA exitosa', 'success')
          location.reload(); // Recargar la página para ver los cambios
      } else {
          alert(`Error: ${data.error}`);
      }
  })
  .catch(error => {
      console.error('Error:', error);
      showNotification('Error al evaluar con IA', 'error');
  });
}

  // Función para calificar manualmente una entrega
function calificarManualmente(event, evaluacionId) {
  event.preventDefault();

  const form = event.target;
  const calificacionInput = form.querySelector('input[name="calificacion"]');
  const comentariosTextarea = form.querySelector('textarea[name="comentarios"]');
  const btnCalificar = document.getElementById(`btn-calificar-${evaluacionId}`);

  const calificacion = calificacionInput.value;
  const comentarios = comentariosTextarea.value;

  if (!calificacion || !comentarios) {
      showNotification('Por favor, complete todos los campos', 'error');
      return false;
  }

  btnCalificar.innerHTML = '<span class="loading-spinner"></span> Guardando...';
  btnCalificar.disabled = true;

  fetch(`/api/calificar/${evaluacionId}`, {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({
          calificacion: calificacion,
          comentarios: comentarios
      })
  })
  .then(response => response.json())
  .then(data => {
      if (data.success) {
          alert(`Calificación: ${data.calificacion}\nComentarios: ${data.comentarios}`);
          location.reload(); // Recargar la página para ver los cambios
      } else {
          alert(`Error: ${data.error}`);
      }
  })
  .catch(error => {
      console.error('Error:', error);
      alert('Ocurrió un error al calificar la entrega.');
  })
  .finally(() => {
      btnCalificar.innerHTML = '<i class="fas fa-save"></i> Guardar calificación';
      btnCalificar.disabled = false;
  });

  return false;
}

  // Función para calificar manualmente una entrega
function calificarManualmente(event, evaluacionId) {
  event.preventDefault();

  const form = event.target;
  const calificacionInput = form.querySelector('input[name="calificacion"]');
  const comentariosTextarea = form.querySelector('textarea[name="comentarios"]');
  const btnCalificar = document.getElementById(`btn-calificar-${evaluacionId}`);

  const calificacion = calificacionInput.value;
  const comentarios = comentariosTextarea.value;

  if (!calificacion || !comentarios) {
      showNotification('Por favor, complete todos los campos', 'error');
      return false;
  }

  btnCalificar.innerHTML = '<span class="loading-spinner"></span> Guardando...';
  btnCalificar.disabled = true;

  fetch(`/api/calificar/${evaluacionId}`, {
  method: 'POST',
  headers: {
      'Content-Type': 'application/json'
  },
  body: JSON.stringify({
      calificacion: calificacion,
      comentarios: comentarios
  })
})
.then(response => response.json())
.then(data => {
  if (data.success) {
      showNotification('Calificación guardada exitosamente', 'success');
      const modalContainer = document.querySelector('.modal-container');
      const practicaId = modalContainer.getAttribute('data-practica-id');
      loadEvaluaciones(practicaId);
  } else {
      showNotification('Error al guardar la calificación', 'error');
  }
})
.catch(error => {
  console.error('Error:', error);
  showNotification('Error al guardar la calificación', 'error');
});
}

// Function to show notification
function showNotification(message, type = 'success') {
    const notification = document.createElement('div');
    notification.classList.add('notification', type);

    let iconClass = 'fas fa-check-circle';
    if (type === 'error') {
      iconClass = 'fas fa-exclamation-circle';
    }

    notification.innerHTML = `
      <i class="${iconClass}"></i>
      <p>${message}</p>
    `;

    document.body.appendChild(notification);

    // Fade out after 3 seconds
    setTimeout(() => {
      notification.classList.add('fade-out');
      // Remove from DOM after animation
      setTimeout(() => {
        document.body.removeChild(notification);
      }, 300);
    }, 3000);
  }

document.addEventListener('DOMContentLoaded', function () {
  const selector = document.getElementById('selectorGrupo');
  const seccionesMateria = document.querySelectorAll('.materia-section');

  function splitLetters(targetElement, text) {
    targetElement.innerHTML = '';
    const letters = [];
  
    for (let i = 0; i < text.length; i++) {
      const span = document.createElement('span');
      span.className = 'letter behind';
      span.innerHTML = text[i] === ' ' ? '&nbsp;' : text[i];
      targetElement.appendChild(span);
      letters.push(span);
    }
  
    return letters;
  }


  function animateOut(letters, callback) {
    for (let i = 0; i < letters.length; i++) {
      setTimeout(() => {
        letters[i].className = 'letter out';
        if (i === letters.length - 1 && callback) {
          setTimeout(callback, 400);
        }
      }, i * 45);
    }
  }
  
  function animateIn(letters) {
    for (let i = 0; i < letters.length; i++) {
      setTimeout(() => {
        letters[i].className = 'letter in';
      }, 340 + i * 45);
    }
  }
  
  function animateHeader(section) {
    const header = section.querySelector('.materia-header .word');
    const nuevoNombre = section.querySelector('.materia-header').dataset.header;
    const headerContainer = section.querySelector('.materia-header');
    const letrasActuales = Array.from(header.querySelectorAll('.letter'));
  
    // Crear el ícono
    const icono = document.createElement('i');
    icono.className = 'fas fa-book';
  
    // Añadir el ícono antes de las letras (si aún no está añadido)
    if (!header.querySelector('.fas')) {
      header.appendChild(icono);  // Aseguramos que el ícono solo se agregue una vez
    }
  
    // Oculta el header inicialmente
    header.style.display = 'none';
  
    // Simula que el contenedor se encoge
    headerContainer.style.transform = 'scaleX(0.8)';
  
    // Inicia la animación de salida de las letras actuales
    animateOut(letrasActuales, () => {
      // Limpiar las letras actuales sin quitar el ícono
      header.innerHTML = '';  // Limpiar solo las letras, no el ícono
      header.appendChild(icono);  // Volver a añadir el ícono
  
      // Crear y agregar las nuevas letras
      const nuevasLetras = [];
      for (let i = 0; i < nuevoNombre.length; i++) {
        const span = document.createElement('span');
        span.className = 'letter behind';
        span.innerHTML = nuevoNombre[i] === ' ' ? '&nbsp;' : nuevoNombre[i];
        header.appendChild(span);
        nuevasLetras.push(span);
      }
  
      // Esperar un poco antes de mostrar el nuevo header
      setTimeout(() => {
        header.style.display = 'inline-block';  // Mostrar el header con el ícono
  
        // Animar las nuevas letras
        animateIn(nuevasLetras);
  
        // Desplazar la página hasta la nueva sección
        section.scrollIntoView({ behavior: 'smooth', block: 'start' });
  
        // Volver al tamaño natural con animación
        headerContainer.style.transform = 'scaleX(1)';
  
        // Animar la entrada de las tarjetas
        const cards = section.querySelectorAll('.practica-card');
        cards.forEach((card, i) => {
          card.classList.remove('saliendo', 'entrando');
          void card.offsetWidth;  // Forzar reflow
          setTimeout(() => {
            card.classList.add('entrando');
          }, i * 80);
        });
      }, 10);
    });
  }
  
  
  

  selector.addEventListener('change', function () {
    const selectedValue = this.value;
  
    // Encuentra la sección actualmente visible
    const currentVisibleSection = Array.from(seccionesMateria).find(
      section => section.style.display !== 'none'
    );
  
    const newSection = Array.from(seccionesMateria).find(
      section => section.getAttribute('data-materia') === selectedValue
    );
  
    if (!newSection || currentVisibleSection === newSection) return;
  
    const currentHeader = currentVisibleSection.querySelector('.materia-header .word');
    const letrasActuales = Array.from(currentHeader.querySelectorAll('.letter'));
  
    animateOut(letrasActuales, () => {
      currentVisibleSection.style.display = 'none';
      newSection.style.display = 'block';
      animateHeader(newSection);
    });
  });
  
  
  
  
  // Inicializar encabezados con letras separadas (sin animación)
  seccionesMateria.forEach(section => {
    const header = section.querySelector('.materia-header .word');
    const nombre = section.querySelector('.materia-header').dataset.header;
  
    if (header.children.length === 0) {
      const letras = nombre.split('');
      letras.forEach(letra => {
        const span = document.createElement('span');
        span.className = 'letter in';
        span.innerHTML = letra === ' ' ? '&nbsp;' : letra;
        header.appendChild(span);
      });
    }
  });
  
const selectorGrupo = document.getElementById('selectorGrupo');
const selectedMateriaId = selectorGrupo.value;

// Mostrar solo la materia seleccionada al inicio
seccionesMateria.forEach(section => {
  const materiaId = section.getAttribute('data-materia');
  section.style.display = (materiaId === selectedMateriaId) ? 'block' : 'none';
});
});

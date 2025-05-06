document.addEventListener('DOMContentLoaded', function() {
    // Inicializar menú móvil
    const burgerMenu = document.querySelector('.burger-menu');
    const menu = document.querySelector('.menu');
    
    if (burgerMenu) {
        burgerMenu.addEventListener('click', function() {
            menu.classList.toggle('active');
        });
    }
    
    // Inicializar tabs con mejoras de accesibilidad
    initTabs();
    
    // Inicializar filtros
    initFiltros();
    
    // Inicializar búsqueda con debounce para mejor rendimiento
    initBusqueda();
    
    // Inicializar switch de IA
    initIASwitch();
    
    // Inicializar botón de crear práctica en mensaje de no hay prácticas
    initCrearPracticaBtn();
    
    // Activar animaciones de tarjetas después de un pequeño retraso
    setTimeout(() => {
        document.querySelector('.practicas-grid').classList.add('loaded');
    }, 100);
    
    // Inicializar fecha de entrega con fecha mínima de hoy
    const fechaEntrega = document.getElementById('fecha_entrega');
    if (fechaEntrega) {
        const hoy = new Date();
        const yyyy = hoy.getFullYear();
        const mm = String(hoy.getMonth() + 1).padStart(2, '0');
        const dd = String(hoy.getDate()).padStart(2, '0');
        const fechaMinima = `${yyyy}-${mm}-${dd}`;
        fechaEntrega.min = fechaMinima;
        fechaEntrega.value = fechaMinima;
    }
});

// Función para inicializar las pestañas con mejoras de accesibilidad
function initTabs() {
    const tabs = document.querySelectorAll('.practicas-tab');
    
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            // Desactivar todas las pestañas
            tabs.forEach(t => {
                t.classList.remove('active');
                t.setAttribute('aria-pressed', 'false');
            });
            
            // Activar la pestaña seleccionada
            this.classList.add('active');
            this.setAttribute('aria-pressed', 'true');
            
            // Ocultar todos los contenidos
            document.querySelectorAll('.practicas-content').forEach(content => {
                content.classList.remove('active');
                content.setAttribute('aria-hidden', 'true');
            });
            
            // Mostrar el contenido seleccionado
            const tabId = this.getAttribute('data-tab');
            const activeContent = document.getElementById(tabId);
            activeContent.classList.add('active');
            activeContent.setAttribute('aria-hidden', 'false');
            
            // Animación suave al cambiar de pestaña
            activeContent.style.animation = 'none';
            activeContent.offsetHeight; // Forzar reflow
            activeContent.style.animation = 'fadeInUp 0.5s cubic-bezier(0.175, 0.885, 0.32, 1.275)';
        });
        
        // Soporte para navegación con teclado
        tab.addEventListener('keydown', function(e) {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                this.click();
            }
        });
    });
}

// Función para inicializar los filtros con mejoras visuales
function initFiltros() {
    const filtros = document.querySelectorAll('.filtro-btn');
    
    filtros.forEach(filtro => {
        filtro.addEventListener('click', function() {
            // Desactivar todos los filtros
            filtros.forEach(f => {
                f.classList.remove('active');
                f.setAttribute('aria-pressed', 'false');
            });
            
            // Activar el filtro seleccionado
            this.classList.add('active');
            this.setAttribute('aria-pressed', 'true');
            
            const tipoFiltro = this.getAttribute('data-filter');
            filtrarPracticas(tipoFiltro);
        });
    });
}

// Función para filtrar prácticas con animaciones
function filtrarPracticas(estado) {
  const cards = document.querySelectorAll('.practica-card');
  let visibles = 0;

  cards.forEach(card => {
      if (estado === 'todos') {
          card.style.display = 'flex';
          visibles++;
      } else {
          const cardEstado = card.getAttribute('data-estado');
          if (cardEstado === estado) {
              card.style.display = 'flex';
              visibles++;
          } else {
              card.style.display = 'none';
          }
      }
  });

  // Verificar si hay tarjetas visibles
  const noPracticas = document.querySelector('.no-practicas');

  if (visibles === 0 && !noPracticas) {
      const grid = document.querySelector('.practicas-grid');
      const mensaje = document.createElement('div');
      mensaje.className = 'no-practicas';
      mensaje.innerHTML = `
          <i class="fas fa-filter"></i>
          <p>No hay prácticas que coincidan con el filtro seleccionado.</p>
          <button class="filtro-btn" data-filter="todos" onclick="resetFiltros()">
              <i class="fas fa-sync-alt"></i> Ver todas
          </button>
      `;
      grid.appendChild(mensaje);
  } else if (visibles > 0 && noPracticas) {
      noPracticas.remove();
  }
}

// Función para resetear filtros
function resetFiltros() {
    document.querySelector('.filtro-btn[data-filter="todos"]').click();
}

// Función para buscar prácticas con debounce para mejor rendimiento
function initBusqueda() {
const busquedaInput = document.getElementById('busquedaPracticas');

if (busquedaInput) {
  let debounceTimeout;
  
  busquedaInput.addEventListener('input', function() {
      clearTimeout(debounceTimeout);
      
      debounceTimeout = setTimeout(() => {
          const termino = this.value.toLowerCase().trim();
          
          // Mostrar todas las tarjetas si el término es muy corto
          if (termino.length < 2) {
              document.querySelectorAll('.practica-card').forEach(card => {
                  card.style.display = 'flex';
              });
              
              eliminarMensajeNoPracticas();
              eliminarMensajeFiltro();
              return;
          }
          
          // Buscar en todas las tarjetas
          const cards = document.querySelectorAll('.practica-card');
          let encontrados = 0;
          
          cards.forEach(card => {
              const titulo = card.querySelector('.practica-card-header h3').textContent.toLowerCase();
              const materia = card.querySelector('.practica-info-item:first-child span').textContent.toLowerCase();
              
              if (titulo.includes(termino) || materia.includes(termino)) {
                  card.style.display = 'flex';
                  encontrados++;
              } else {
                  card.style.display = 'none';
              }
          });
          
          // Mostrar mensaje si no hay resultados
          if (encontrados === 0) {
              eliminarMensajeFiltro(); // Eliminar mensaje de filtro si existe
              mostrarMensajeNoPracticas(termino);
          } else {
              eliminarMensajeNoPracticas();
          }
      }, 300); // Debounce de 300ms para mejor rendimiento
  });
}
}

// Función para mostrar el mensaje de "No se encontraron prácticas"
function mostrarMensajeNoPracticas(termino) {
const grid = document.querySelector('.practicas-grid');
const mensaje = document.createElement('div');
mensaje.className = 'no-practicas';
mensaje.innerHTML = `
  <i class="fas fa-search"></i>
  <p>No se encontraron prácticas que coincidan con "<strong>${termino}</strong>".</p>
  <button class="filtro-btn" onclick="limpiarBusqueda()">
      <i class="fas fa-times-circle"></i> Limpiar búsqueda
  </button>
`;
grid.appendChild(mensaje);
}

// Función para eliminar el mensaje de "No se encontraron prácticas"
function eliminarMensajeNoPracticas() {
const noPracticas = document.querySelector('.no-practicas');
if (noPracticas) {
  noPracticas.remove();
}
}

// Función para manejar el cambio de secciones
function manejarCambioDeSeccion() {
// Eliminar el mensaje de búsqueda no encontrada al cambiar de sección
eliminarMensajeNoPracticas();
eliminarMensajeFiltro();
}

// Función para mostrar el mensaje de "No hay prácticas que coincidan con el filtro seleccionado"
function mostrarMensajeFiltro() {
const grid = document.querySelector('.practicas-grid');
const mensaje = document.createElement('div');
mensaje.className = 'no-practicas';
mensaje.innerHTML = `
  <i class="fas fa-filter"></i>
  <p>No hay prácticas que coincidan con el filtro seleccionado.</p>
  <button class="filtro-btn" onclick="resetFiltros()">
      <i class="fas fa-sync-alt"></i> Ver todas
  </button>
`;
grid.appendChild(mensaje);
}

// Función para eliminar el mensaje de "No hay prácticas que coincidan con el filtro seleccionado"
function eliminarMensajeFiltro() {
const noPracticasFiltro = document.querySelector('.no-practicas');
if (noPracticasFiltro) {
  noPracticasFiltro.remove();
}
}

// Función para limpiar búsqueda con animación
function limpiarBusqueda() {
const busquedaInput = document.getElementById('busquedaPracticas');
if (busquedaInput) {
  busquedaInput.value = '';
  busquedaInput.focus();
  busquedaInput.dispatchEvent(new Event('input'));
}
}

// Llama a manejarCambioDeSeccion en el evento de cambio de pestañas o secciones
document.querySelectorAll('.practicas-tab').forEach(tab => {
tab.addEventListener('click', manejarCambioDeSeccion);
});

// Función para inicializar el switch de IA con animaciones
function initIASwitch() {
const switchIA = document.getElementById('uso_ia_switch');
const inputIA = document.getElementById('uso_ia');
const labelIA = document.getElementById('uso_ia_label');

if (switchIA && inputIA && labelIA) {
  switchIA.addEventListener('change', function() {
      if (this.checked) {
          inputIA.value = '1'; // Actualizar el valor del campo oculto
          labelIA.textContent = 'Activado';
          labelIA.style.color = 'var(--azul-unam)';
          
          // Efecto visual al activar
          labelIA.style.animation = 'none';
          labelIA.offsetHeight; // Forzar reflow
          labelIA.style.animation = 'fadeIn 0.3s ease-out';
      } else {
          inputIA.value = '0'; // Actualizar el valor del campo oculto
          labelIA.textContent = 'Desactivado';
          labelIA.style.color = 'var(--gray-700)';
          
          // Efecto visual al desactivar
          labelIA.style.animation = 'none';
          labelIA.offsetHeight; // Forzar reflow
          labelIA.style.animation = 'fadeIn 0.3s ease-out';
      }
  });
}
}

// Función para inicializar el botón de crear práctica en mensaje de no hay prácticas
function initCrearPracticaBtn() {
    const btnCrear = document.querySelector('.practicas-tab-btn');
    
    if (btnCrear) {
        btnCrear.addEventListener('click', function() {
            const tabId = this.getAttribute('data-tab');
            document.querySelector(`.practicas-tab[data-tab="${tabId}"]`).click();
        });
    }
}

// Función para confirmar eliminación con animaciones mejoradas
function confirmarEliminar(practicaId) {
    const modal = document.getElementById('modal-eliminar');
    const form = document.getElementById('form-eliminar');
    
    // Construir la URL de eliminación en el lado del cliente
    const eliminarUrl = `/eliminar_practica/${practicaId}`;
    form.action = eliminarUrl;  // Establecer la acción del formulario

    modal.classList.add('active');
    modal.setAttribute('aria-hidden', 'false');
    
    // Enfocar el botón de cancelar para mejor accesibilidad
    setTimeout(() => {
        document.querySelector('.modal-btn-cancel').focus();
    }, 100);
    
    // Deshabilitar el scroll del body
    document.body.style.overflow = 'hidden';
}

// Función para cerrar modal con animación
function cerrarModal() {
    const modal = document.getElementById('modal-eliminar');
    modal.classList.remove('active');
    modal.setAttribute('aria-hidden', 'true');
    
    // Restaurar el scroll del body
    document.body.style.overflow = '';
}

// Cerrar modal al hacer clic fuera o con tecla Escape
window.addEventListener('click', function(event) {
    const modal = document.getElementById('modal-eliminar');
    if (event.target === modal) {
        cerrarModal();
    }
});

document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
        cerrarModal();
    }
});

// Animación para el formulario cuando se envía
document.getElementById('practica-gen-form').addEventListener('submit', function(e) {
    const btn = document.getElementById('form-btn');
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Creando...';
    btn.disabled = true;
    
    // Añadir clase para efecto de envío
    this.classList.add('submitting');
});

window.addEventListener('scroll', function() {
const filtroContainer = document.querySelector('.filtro-container');
const footer = document.querySelector('footer'); // Asegúrate de que el selector sea correcto para tu footer

// Obtener la posición del footer
const footerRect = footer.getBoundingClientRect();

// Calcular la distancia entre el filtro y el footer
const distanceToFooter = footerRect.top - window.innerHeight;

// Si el filtro está cerca del footer, ponemos visibility: hidden
if (distanceToFooter <= 0) {
filtroContainer.style.visibility = 'hidden'; // Filtro desaparece al tocar el footer
} else {
filtroContainer.style.visibility = 'visible'; // Filtro visible cuando no está cerca del footer
}
});
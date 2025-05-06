// // Funciones para manejar modales
// function openModal(modalId) {
//     const modal = document.getElementById(modalId);
//     if (modal) {
//       modal.classList.add('active');
      
//       // Bloquear scroll del body
//       document.body.style.overflow = 'hidden';
      
//       // Añadir evento para cerrar con Escape
//       document.addEventListener('keydzown', closeModalOnEscape);
//     }
//   }
  
//   function closeModal(modalId) {
//     const modal = document.getElementById(modalId);
//     if (modal) {
//       modal.classList.remove('active');
      
//       // Restaurar scroll del body
//       document.body.style.overflow = '';
      
//       // Eliminar evento de Escape
//       document.removeEventListener('keydown', closeModalOnEscape);
//     }
//   }
  
//   function closeModalOnEscape(e) {
//     if (e.key === 'Escape') {
//       // Cerrar el modal activo
//       const activeModal = document.querySelector('.modal-overlay.active');
//       if (activeModal) {
//         activeModal.classList.remove('active');
        
//         // Restaurar scroll del body
//         document.body.style.overflow = '';
        
//         // Eliminar este evento
//         document.removeEventListener('keydown', closeModalOnEscape);
//       }
//     }
//   }
  
//   // Inicializar modales
//   document.addEventListener('DOMContentLoaded', () => {
//     // Configurar botones para abrir modales
//     document.querySelectorAll('[data-modal]').forEach(button => {
//       const modalId = button.getAttribute('data-modal');
//       button.addEventListener('click', () => openModal(modalId));
//     });
    
    
//     // Configurar botones para cerrar modales
//     document.querySelectorAll('.modal-close, .modal-cancel').forEach(button => {
//       button.addEventListener('click', () => {
//         const modal = button.closest('.modal-overlay');
//         if (modal) {
//           modal.classList.remove('active');
          
//           // Restaurar scroll del body
//           document.body.style.overflow = '';
//         }
//       });
//     });
    
//     // Cerrar modal al hacer clic en el overlay
//     document.querySelectorAll('.modal-overlay').forEach(overlay => {
//       overlay.addEventListener('click', (e) => {
//         if (e.target === overlay) {
//           overlay.classList.remove('active');
          
//           // Restaurar scroll del body
//           document.body.style.overflow = '';
//         }
//       });
//     });
//   });
  
//   // Función para mostrar evaluaciones en un modal
//   function showEvaluaciones(practicaId, titulo) {
//     // Actualizar el título del modal
//     const modalTitle = document.getElementById('modalTitle');
//     if (modalTitle) {
//       modalTitle.textContent = `Evaluaciones: ${titulo}`;
//     }
    
//     // Obtener las evaluaciones mediante AJAX
//     fetch(`/api/evaluaciones/${practicaId}`)
//       .then(response => response.json())
//       .then(data => {
//         if (data.success) {
//           // Actualizar el contenido del modal
//           const evaluacionesGrid = document.getElementById('evaluacionesGrid');
//           if (evaluacionesGrid) {
//             evaluacionesGrid.innerHTML = '';
            
//             if (data.evaluaciones && data.evaluaciones.length > 0) {
//               data.evaluaciones.forEach(evaluacion => {
//                 const card = document.createElement('div');
//                 card.className = 'evaluacion-card';
                
//                 // Determinar el estado visual
//                 let estadoClass = 'status-pendiente';
//                 if (evaluacion.estado === 'entregado') estadoClass = 'status-entregado';
//                 if (evaluacion.estado === 'calificado') estadoClass = 'status-calificado';
//                 if (evaluacion.estado === 'atrasado') estadoClass = 'status-atrasado';
                
//                 // Determinar la calificación
//                 const calificacion = evaluacion.calificacion !== null ? 
//                   `<div class="evaluacion-calificacion">${evaluacion.calificacion}</div>` : 
//                   '<div class="evaluacion-calificacion">Sin calificar</div>';
                
//                 card.innerHTML = `
//                   <div class="evaluacion-header">
//                     <div class="evaluacion-estudiante">${evaluacion.estudiante_nombre}</div>
//                     <div class="entrega-status ${estadoClass}">${evaluacion.estado}</div>
//                   </div>
//                   <div class="evaluacion-body">
//                     ${calificacion}
//                     <div class="evaluacion-acciones">
//                       ${evaluacion.archivo_url ? 
//                         `<a href="/static/uploads/${evaluacion.archivo_url}" target="_blank" class="btn-sm btn-info">
//                           <i class="fas fa-download"></i> Ver entrega
//                         </a>` : 
//                         '<span class="no-entrega">Sin entrega</span>'}
//                       ${evaluacion.estado === 'entregado' || evaluacion.estado === 'calificado' ? 
//                         `<button class="btn-sm btn-success" onclick="calificarEntrega('${evaluacion.evaluacion_id}', '${evaluacion.estudiante_nombre}')">
//                           <i class="fas fa-check"></i> ${evaluacion.estado === 'calificado' ? 'Recalificar' : 'Calificar'}
//                         </button>` : ''}
//                     </div>
//                   </div>
//                 `;
                
//                 evaluacionesGrid.appendChild(card);
//               });
//             } else {
//               evaluacionesGrid.innerHTML = '<div class="no-data">No hay evaluaciones disponibles para esta práctica.</div>';
//             }
//           }
          
//           // Abrir el modal
//           openModal('evaluacionesModal');
//         } else {
//           showNotification(data.error || 'Error al cargar evaluaciones', 'error');
//         }
//       })
//       .catch(error => {
//         console.error('Error:', error);
//         showNotification('Error al cargar evaluaciones', 'error');
//       });
//   }
  
//   // Función para calificar una entrega
//   function calificarEntrega(evaluacionId, nombreEstudiante) {
//     // Cerrar el modal de evaluaciones
//     closeModal('evaluacionesModal');
    
//     // Abrir el modal de calificación
//     const calificacionModal = document.getElementById('calificacionModal');
//     if (calificacionModal) {
//       // Actualizar el título
//       const modalTitle = calificacionModal.querySelector('.modal-title');
//       if (modalTitle) {
//         modalTitle.textContent = `Calificar entrega de ${nombreEstudiante}`;
//       }
      
//       // Actualizar el ID de la evaluación
//       const evaluacionIdInput = calificacionModal.querySelector('input[name="evaluacion_id"]');
//       if (evaluacionIdInput) {
//         evaluacionIdInput.value = evaluacionId;
//       }
      
//       // Limpiar campos
//       const calificacionInput = calificacionModal.querySelector('input[name="calificacion"]');
//       const comentariosInput = calificacionModal.querySelector('textarea[name="comentarios"]');
//       if (calificacionInput) calificacionInput.value = '';
//       if (comentariosInput) comentariosInput.value = '';
      
//       // Abrir el modal
//       calificacionModal.classList.add('active');
//     }
//   }
  
//   // Función para guardar calificación
//   function guardarCalificacion(form) {
//     const evaluacionId = form.querySelector('input[name="evaluacion_id"]').value;
//     const calificacion = form.querySelector('input[name="calificacion"]').value;
//     const comentarios = form.querySelector('textarea[name="comentarios"]').value;
    
//     // Validar campos
//     if (!calificacion) {
//       showNotification('La calificación es requerida', 'error');
//       return false;
//     }
    
//     // Enviar datos mediante AJAX
//     fetch(`/api/calificar/${evaluacionId}`, {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//       },
//       body: JSON.stringify({
//         calificacion,
//         comentarios
//       })
//     })
//     .then(response => response.json())
//     .then(data => {
//       if (data.success) {
//         showNotification('Calificación guardada exitosamente', 'success');
        
//         // Cerrar el modal
//         closeModal('calificacionModal');
//       } else {
//         showNotification(data.error || 'Error al guardar calificación', 'error');
//       }
//     })
//     .catch(error => {
//       console.error('Error:', error);
//       showNotification('Error al guardar calificación', 'error');
//     });
    
//     return false; // Evitar envío del formulario
//   }

  // Funciones para manejar el modal de retroalimentación
  function openFeedbackModal() {
    document.getElementById('feedbackModal').classList.add('active');
    document.body.style.overflow = 'hidden'; // Evitar scroll en el fondo
  }
  
  function closeFeedbackModal() {
    document.getElementById('feedbackModal').classList.remove('active');
    document.body.style.overflow = ''; // Restaurar scroll
  }
  
  // Cerrar modal al hacer clic fuera de él
  document.getElementById('feedbackModal').addEventListener('click', function(event) {
    if (event.target === this) {
      closeFeedbackModal();
    }
  });
  
  // Evitar que el clic dentro del modal lo cierre
  document.querySelector('.modal-container').addEventListener('click', function(event) {
    event.stopPropagation();
  });
  
  // Cerrar modal con tecla Escape
  document.addEventListener('keydown', function(event) {
    if (event.key === 'Escape') {
      closeFeedbackModal();
    }
  });

// Función para abrir el modal de evaluaciones
  function openPracticaModal(practicaId, practicaTitulo) {
    // Actualizar el título del modal
    document.getElementById('modalTitle').innerHTML = `<div class="contenedor-bolitas">
    <div class="bolita roja"></div>
    <div class="bolita amarilla"></div>
    <div class="bolita verde"></div>
    </div> ${practicaTitulo}`;
    
    // Guardar el ID de la práctica en el contenedor del modal para referencia futura
    document.querySelector('.modal-container').setAttribute('data-practica-id', practicaId);
    
    // Cargar las evaluaciones para esta práctica
    loadEvaluaciones(practicaId);
    
    // Mostrar el modal con animación
    const modal = document.getElementById('evaluacionesModal');
    modal.classList.add('active');
    
    // Prevenir scroll en el body
    document.body.style.overflow = 'hidden';
  }
  
  // Función para cerrar el modal
  function closeModal() {
    const modal = document.getElementById('evaluacionesModal');
    modal.classList.remove('active');
    
    // Restaurar scroll en el body
    document.body.style.overflow = '';
  }
// Añadir animación para las métricas cuando las tarjetas se muestran
document.addEventListener('DOMContentLoaded', function() {
    const burgerMenu = document.getElementById('burgerMenu');
    const menu = document.querySelector('.menu');
    
    if (burgerMenu) {
        burgerMenu.addEventListener('click', function() {
            menu.classList.toggle('active');
        });
    }
    
    // Inicializar todos los reversos de tarjetas
    document.querySelectorAll('.aula-card-back, .create-class-back').forEach(back => {
        back.style.display = 'none';
    });
    
    // Animación de repartir cartas
    const cardContainers = document.querySelectorAll('.aula-card-container, .create-class-card-container');
    let delay = 0;
    cardContainers.forEach((container, index) => {
        // Calcular un retraso escalonado con un poco de aleatoriedad para simular barajeo
        delay = 80 * index + Math.random() * 50;
        setTimeout(() => {
            container.classList.add('show');
            
            // Animar las barras de progreso después de que la tarjeta aparezca
            const progressBar = container.querySelector('.aula-progress-bar');
            if (progressBar) {
                setTimeout(() => {
                    const currentWidth = progressBar.style.width;
                    progressBar.style.width = '0%';
                    setTimeout(() => {
                        progressBar.style.width = currentWidth;
                    }, 50);
                }, 300);
            }
            
            // Animar las métricas
            const metrics = container.querySelectorAll('.aula-metric');
            metrics.forEach((metric, i) => {
                metric.style.opacity = '0';
                metric.style.transform = 'translateY(20px)';
                setTimeout(() => {
                    metric.style.opacity = '1';
                    metric.style.transform = 'translateY(0)';
                    metric.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
                }, 300 + (i * 100));
            });
        }, delay);
    });
    
    // Inicializar el buscador de estudiantes
    initStudentSearch();
    
    // Inicializar el formulario de crear grupo
    initCreateGroupForm();
    
    // Cargar estudiantes disponibles para cada grupo
    cargarEstudiantesDisponibles();
});

// Función para voltear la tarjeta
function flipCard(grupoId, type) {
    const card = document.getElementById(`card-${grupoId}`);
    
    // Mostrar el reverso solicitado antes de iniciar la animación
    document.querySelectorAll(`#card-${grupoId} .aula-card-back`).forEach(back => {
        back.style.display = 'none';
    });
    
    const targetBack = document.getElementById(`${type}-${grupoId}`);
    targetBack.style.display = 'flex';
    
    // Aplicar la clase flipped para iniciar la animación
    card.classList.add('flipped');
    
    // Si es la vista de estudiantes, cargar los estudiantes disponibles
    if (type === 'students') {
        cargarEstudiantesParaGrupo(grupoId);
    }
}

// Función para volver al frente de la tarjeta
function flipCardBack(grupoId) {
    const card = document.getElementById(`card-${grupoId}`);
    card.classList.remove('flipped');
}

// Función para voltear la tarjeta de crear clase
function flipCreateCard() {
    const card = document.getElementById('create-card');
    document.querySelector('.create-class-back').style.display = 'flex';
    card.classList.add('flipped');
}

// Función para volver al frente de la tarjeta de crear clase
function flipCreateCardBack() {
    const card = document.getElementById('create-card');
    card.classList.remove('flipped');
}

// Función para eliminar estudiante
function eliminarEstudiante(grupoId, usuarioId, boton) {
    if (confirm('¿Estás seguro de que deseas eliminar a este estudiante del grupo?')) {
        const fila = boton.closest('tr');

        fetch(`/gestionar_estudiantes/${grupoId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                usuario_id: usuarioId,
                accion: 'eliminar'
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                fila.remove();
                const tabla = document.getElementById(`student-table-${grupoId}`);
                const filas = tabla.querySelectorAll('tbody tr');
                if (filas.length === 0) {
                    tabla.querySelector('tbody').innerHTML = '<tr><td colspan="2" class="empty-list">No hay estudiantes inscritos</td></tr>';
                }
                actualizarContadorEstudiantes(grupoId);
                alert('Estudiante eliminado correctamente');
            } else {
                alert('Error al eliminar estudiante: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error al procesar la solicitud');
        });
    }
}

// Función para agregar estudiante
function agregarEstudiante(grupoId) {
    const select = document.getElementById(`estudiante-select-${grupoId}`);
    const usuarioId = select.value;
    const nombreEstudiante = select.options[select.selectedIndex].text;
    
    if (!usuarioId) {
        alert('Por favor selecciona un estudiante');
        return;
    }
    
    // Enviar solicitud para agregar al estudiante
    fetch(`/gestionar_estudiantes/${grupoId}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            usuario_id: usuarioId,
            accion: 'agregar'
        })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Actualizar la tabla de estudiantes
            const tabla = document.getElementById(`student-table-${grupoId}`);
            const tbody = tabla.querySelector('tbody');
            
            // Verificar si hay un mensaje de "No hay estudiantes"
            const emptyRow = tbody.querySelector('.empty-list');
            if (emptyRow) {
                tbody.innerHTML = '';
            }
            
            // Agregar el nuevo estudiante a la tabla
            const newRow = document.createElement('tr');
            newRow.dataset.usuarioId = usuarioId;
            newRow.innerHTML = `
                <td>${nombreEstudiante}</td>
                <td>
                    <button onclick="eliminarEstudiante('${grupoId}', '${tbody.children.length + 1}')">Eliminar</button>
                </td>
            `;
            tbody.appendChild(newRow);
            
            // Actualizar el contador de estudiantes en el frente de la tarjeta
            actualizarContadorEstudiantes(grupoId);
            
            // Eliminar la opción del select
            select.remove(select.selectedIndex);
            select.value = '';
            
            // Mostrar mensaje de éxito
            alert('Estudiante agregado correctamente');
        } else {
            alert('Error al agregar estudiante: ' + data.error);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('Error al procesar la solicitud');
    });
}

// Función para actualizar el contador de estudiantes
function actualizarContadorEstudiantes(grupoId) {
    const tabla = document.getElementById(`student-table-${grupoId}`);
    const filas = tabla.querySelectorAll('tbody tr');
    let numEstudiantes = filas.length;
    
    // Si hay una fila con mensaje de "No hay estudiantes", el contador es 0
    if (filas.length === 1 && filas[0].querySelector('.empty-list')) {
        numEstudiantes = 0;
    }
    
    // Actualizar el contador en el frente de la tarjeta
    const card = document.getElementById(`card-${grupoId}`);
    const statsDiv = card.querySelector('.aula-stats div:first-child');
    statsDiv.innerHTML = `<i class="fas fa-users"></i> ${numEstudiantes} Estudiantes`;
    
    // Actualizar la barra de progreso
    const progressBar = card.querySelector('.aula-progress-bar');
    if (progressBar) {
        progressBar.style.width = `${(numEstudiantes / 30) * 100}%`;
    }
}

// Función para cargar estudiantes disponibles para un grupo
function cargarEstudiantesParaGrupo(grupoId) {
    // Hacer una solicitud para obtener todos los estudiantes
    fetch('/api/usuarios')
        .then(response => response.json())
        .then(data => {
            if (data.usuarios) {
                const estudiantes = data.usuarios.filter(u => u.rol === 'estudiante');
                const select = document.getElementById(`estudiante-select-${grupoId}`);
                
                // Limpiar opciones existentes excepto la primera
                while (select.options.length > 1) {
                    select.remove(1);
                }
                
                // Obtener los nombres de los estudiantes ya inscritos
                const tabla = document.getElementById(`student-table-${grupoId}`);
                const estudiantesInscritos = Array.from(tabla.querySelectorAll('tbody tr td:first-child'))
                    .map(td => td.textContent.trim());
                
                // Agregar solo los estudiantes que no están inscritos
                estudiantes.forEach(estudiante => {
                    if (!estudiantesInscritos.includes(estudiante.nombre)) {
                        const option = document.createElement('option');
                        option.value = estudiante.id;
                        option.textContent = estudiante.nombre;
                        select.appendChild(option);
                    }
                });
            }
        })
        .catch(error => {
            console.error('Error al cargar estudiantes:', error);
        });
}

// Función para cargar estudiantes disponibles para todos los grupos
function cargarEstudiantesDisponibles() {
    // Hacer una solicitud para obtener todos los estudiantes
    fetch('/api/usuarios')
        .then(response => response.json())
        .then(data => {
            if (data.usuarios) {
                // Almacenar los estudiantes en una variable global para usarlos después
                window.todosEstudiantes = data.usuarios.filter(u => u.rol === 'estudiante');
            }
        })
        .catch(error => {
            console.error('Error al cargar estudiantes:', error);
        });
}

// Inicializar el buscador de estudiantes
function initStudentSearch() {
    document.querySelectorAll('[id^="search-student-"]').forEach(input => {
        const grupoId = input.id.split('-')[2];
        const resultsDiv = document.getElementById(`search-results-${grupoId}`);
        
        input.addEventListener('input', function() {
            const searchTerm = this.value.toLowerCase().trim();
            
            if (searchTerm.length < 3) {
                resultsDiv.style.display = 'none';
                return;
            }
            
            // Buscar en la tabla de estudiantes
            const tabla = document.getElementById(`student-table-${grupoId}`);
            const estudiantes = Array.from(tabla.querySelectorAll('tbody tr td:first-child'))
                .map(td => td.textContent.toLowerCase());
            
            const found = estudiantes.some(estudiante => estudiante.includes(searchTerm));
            
            resultsDiv.style.display = 'block';
            if (found) {
                resultsDiv.textContent = `El estudiante "${searchTerm}" está inscrito en este grupo.`;
                resultsDiv.className = 'student-search-results found';
            } else {
                resultsDiv.textContent = `El estudiante "${searchTerm}" no está inscrito en este grupo.`;
                resultsDiv.className = 'student-search-results not-found';
            }
        });
    });
}

// Inicializar el formulario de crear grupo
function initCreateGroupForm() {
const semestreSelect = document.getElementById('semestre_id');
const turnoSelect = document.getElementById('turno');
const grupoSelect = document.getElementById('grupo_id');

// Función para actualizar los grupos disponibles
function updateGrupos() {
const semestreId = semestreSelect.value;
const turno = turnoSelect.value === 'matutino' ? 'A' : 'B';

grupoSelect.innerHTML = '<option value="">Seleccione un grupo</option>'; // Limpiar grupos anteriores

if (semestreId) {
    for (let i = 1; i <= 3; i++) {
        const grupoNumero = `0${i}`.slice(-2); // Asegura que el número tenga 2 dígitos ("01", "02", etc.)
        const option = document.createElement('option');
        option.value = grupoNumero; // Solo el número del grupo
        option.textContent = grupoNumero;
        grupoSelect.appendChild(option);
    }
}
}

// Eventos para actualizar los grupos
if (semestreSelect && turnoSelect) {
semestreSelect.addEventListener('change', updateGrupos);
turnoSelect.addEventListener('change', function() {
    updateGrupos(); // Actualizar grupos cuando cambie el turno
});

// Inicializar grupos
updateGrupos();
}

// Manejar el envío del formulario
const createForm = document.getElementById('create-form');
if (createForm) {
createForm.addEventListener('submit', function(e) {
    // No prevenimos el envío del formulario para que se procese normalmente
    // El backend se encargará de crear el grupo y redirigir
});
}
}

// Manejar el envío de los formularios de edición
document.querySelectorAll('.edit-form').forEach(form => {
    form.addEventListener('submit', function(e) {
        // No prevenimos el envío del formulario para que se procese normalmente
        // El backend se encargará de actualizar el grupo y redirigir
    });
});

// Search functionality
document.getElementById('searchAulas').addEventListener('input', function() {
    const searchTerm = this.value.toLowerCase();
    const cards = document.querySelectorAll('.aula-card-container');

    cards.forEach(card => {
        const cardElement = card.querySelector('.aula-card');
        if (!cardElement) return; // Saltar si no es una tarjeta de aula
        
        const groupName = cardElement.querySelector('.aula-card-header h2').textContent.toLowerCase();
        const description = cardElement.querySelector('.aula-card-header p').textContent.toLowerCase();
        
        const matchesSearch = groupName.includes(searchTerm) || description.includes(searchTerm);

        card.style.display = matchesSearch ? 'block' : 'none';
    });
    
    // La tarjeta de crear siempre visible
    document.querySelector('.create-class-card-container').style.display = 'block';
});

// Filter buttons
const filterButtons = document.querySelectorAll('.aulas-filter-btn');
filterButtons.forEach(button => {
    button.addEventListener('click', function() {
        // Quitar la clase "active" de todos los botones y activarla en el seleccionado
        filterButtons.forEach(btn => btn.classList.remove('active'));
        this.classList.add('active');

        // Obtener el tipo de filtro seleccionado
        const filterType = this.textContent.trim();
        const cards = Array.from(document.querySelectorAll('.aula-card-container'));
        const createCardContainer = document.querySelector('.create-class-card-container');

        // Restaurar visibilidad antes de aplicar filtros
        cards.forEach(card => card.style.display = 'block');

        if (filterType.includes("Matutino")) {
            // Filtrar grupos matutinos (nombres con "A-")
            cards.forEach(card => {
                const cardElement = card.querySelector('.aula-card');
                if (!cardElement) return; // Saltar si no es una tarjeta de aula
                
                const groupName = cardElement.querySelector('.aula-card-header h2').textContent;
                card.style.display = groupName.includes("A-") ? 'block' : 'none';
            });
        
        } else if (filterType.includes("Vespertino")) {
            // Filtrar grupos vespertinos (nombres con "B-")
            cards.forEach(card => {
                const cardElement = card.querySelector('.aula-card');
                if (!cardElement) return; // Saltar si no es una tarjeta de aula
                
                const groupName = cardElement.querySelector('.aula-card-header h2').textContent;
                card.style.display = groupName.includes("B-") ? 'block' : 'none';
            });
        
        } else if (filterType.includes("Más estudiantes") || filterType.includes("Menos estudiantes")) {
            // Obtener el contenedor de las aulas
            const container = document.querySelector('.aulas-grid');
            
            // Remover el botón de crear clase temporalmente
            if (createCardContainer) {
                container.removeChild(createCardContainer);
            }
        
            // Ordenar los grupos según el número de estudiantes
            const sortedCards = cards.sort((a, b) => {
                const cardElementA = a.querySelector('.aula-card');
                const cardElementB = b.querySelector('.aula-card');
                
                if (!cardElementA || !cardElementB) return 0;
                
                const studentsTextA = cardElementA.querySelector('.aula-stats div:first-child').textContent;
                const studentsTextB = cardElementB.querySelector('.aula-stats div:first-child').textContent;
                
                const studentsA = parseInt(studentsTextA.match(/\d+/)[0]) || 0;
                const studentsB = parseInt(studentsTextB.match(/\d+/)[0]) || 0;
                
                return filterType.includes("Más estudiantes") ? studentsB - studentsA : studentsA - studentsB;
            });
        
            // Vaciar el contenedor y agregar los elementos ordenados
            container.innerHTML = "";
            sortedCards.forEach(card => container.appendChild(card));
            
            // Volver a añadir el botón de crear clase
            if (createCardContainer) {
                container.appendChild(createCardContainer);
            }
        }
        
        // La tarjeta de crear siempre visible
        document.querySelector('.create-class-card-container').style.display = 'block';
    });
});
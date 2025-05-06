let isNotificationActive = false;

function showNotification(message, type = "success") {
  if (isNotificationActive) return; // 🚫 Ignora si ya hay una activa

  isNotificationActive = true;

  const notification = document.createElement("div");
  notification.classList.add("notification", type);

  let iconClass = "fas fa-exclamation-triangle";

  notification.innerHTML = `
    <i class="${iconClass}"></i>
    <p>${message}</p>
  `;

  document.body.appendChild(notification);

  // Reproducir sonido
  const audio = new Audio("static/music/notificacion.mp3");
  audio.play().catch(error => {
    console.error("No se pudo reproducir el sonido:", error);
  });

  // Ocultar notificación después de 3s
  setTimeout(() => {
    notification.classList.add("fade-out");
    setTimeout(() => {
      if (document.body.contains(notification)) {
        document.body.removeChild(notification);
      }
      isNotificationActive = false; // ✅ Permitir mostrar otra
    }, 600);
  }, 3000);
}

// Variables para seguimiento del progreso
let currentSection = 1;
const totalSections = 3;

// Función para avanzar a la siguiente sección
function nextSection(current, next) {
    // Validar la sección actual antes de avanzar
    if (!validateSection(current)) {
        return false;
    }

    // Ocultar sección actual y mostrar la siguiente
    document.getElementById(`section${current}`).classList.remove("active");
    document.getElementById(`section${next}`).classList.add("active");

    // Actualizar progreso
    currentSection = next;
    updateProgress();

    // Si es la última sección, actualizar el resumen
    if (next === 3) {
        updateSummary();
    }

    // Scroll al inicio del formulario con animación
    window.scrollTo({
        top: document.querySelector(".profile-content").offsetTop - 20,
        behavior: "smooth",
    });

    // Efecto de entrada para la nueva sección
    const newSection = document.getElementById(`section${next}`);
    newSection.style.animation = "none";
    setTimeout(() => {
        newSection.style.animation = "fadeIn 0.5s ease";
    }, 10);

    return true;
}

// Función para volver a la sección anterior
function prevSection(current, prev) {
    document.getElementById(`section${current}`).classList.remove("active");
    document.getElementById(`section${prev}`).classList.add("active");

    currentSection = prev;
    updateProgress();

    // Scroll al inicio del formulario con animación
    window.scrollTo({
        top: document.querySelector(".profile-content").offsetTop - 20,
        behavior: "smooth",
    });

    // Efecto de entrada para la nueva sección
    const newSection = document.getElementById(`section${prev}`);
    newSection.style.animation = "none";
    setTimeout(() => {
        newSection.style.animation = "fadeIn 0.5s ease";
    }, 10);

    return true;
}

// Función para actualizar la barra de progreso
function updateProgress() {
    // Actualizar la barra de progreso
    const progressFill = document.getElementById("progressFill");
    const progressPercentage = (currentSection / totalSections) * 100;
    progressFill.style.width = `${progressPercentage}%`;

    // Actualizar los pasos
    const steps = document.querySelectorAll(".step");
    steps.forEach((step, index) => {
        if (index + 1 <= currentSection) {
            step.classList.add("active");
        } else {
            step.classList.remove("active");
        }
    });
}

// Función para validar los campos de cada sección
function validateSection(sectionNumber) {
    let isValid = true;

    if (sectionNumber === 1) {
        // Validar campos de la sección 1
        const semestre = document.getElementById("semestre").value;
        const facultad = document.getElementById("facultad").value;
        const carrera = document.getElementById("carrera").value;

        if (!semestre || !facultad || !carrera) {
          showNotification('No tan rapido!! Ingresa los datos solicitados antes de continuar');
          isValid = false;
        }
    } else if (sectionNumber === 2) {
        // Validar que al menos un estilo de aprendizaje esté seleccionado
        const estilosSeleccionados = document.querySelectorAll('input[name="estilos_aprendizaje"]:checked');

        if (estilosSeleccionados.length === 0) {
          showNotification('El aprendizaje, el aprendizaje!!');
          isValid = false;
        }
    }

    return isValid;
}

// Función para actualizar el resumen
function updateSummary() {
    // Obtener valores de los campos
    const semestre = document.getElementById("semestre").value;
    const facultad = document.getElementById("facultad").value;
    const carrera = document.getElementById("carrera").value;

    // Obtener estilos de aprendizaje seleccionados
    const estilosSeleccionados = Array.from(document.querySelectorAll('input[name="estilos_aprendizaje"]:checked')).map(
        (input) => {
            const style = input.value;
            // Convertir primera letra a mayúscula
            return style.charAt(0).toUpperCase() + style.slice(1);
        },
    );

    // Actualizar el resumen
    document.getElementById("summary-semestre").textContent = semestre;
    document.getElementById("summary-facultad").textContent = facultad;
    document.getElementById("summary-carrera").textContent = carrera;
    document.getElementById("summary-estilos").textContent = estilosSeleccionados.join(", ") || "Ninguno seleccionado";
}

// Crear fondo interactivo
function createInteractiveBackground() {
    const bg = document.getElementById('interactiveBg');
    const numParticles = 50;
    
    for (let i = 0; i < numParticles; i++) {
        const particle = document.createElement('div');
        particle.classList.add('bg-particle');
        
        // Tamaño aleatorio
        const size = Math.random() * 100 + 50;
        particle.style.width = `${size}px`;
        particle.style.height = `${size}px`;
        
        // Posición aleatoria
        const posX = Math.random() * 100;
        const posY = Math.random() * 100;
        particle.style.left = `${posX}%`;
        particle.style.top = `${posY}%`;
        
        // Opacidad aleatoria
        const opacity = Math.random() * 0.3 + 0.1;
        particle.style.opacity = opacity;
        
        // Animación aleatoria
        const duration = Math.random() * 20 + 10;
        particle.style.animation = `float ${duration}s ease-in-out infinite`;
        particle.style.animationDelay = `${Math.random() * 5}s`;
        
        bg.appendChild(particle);
    }
    
    // Efecto de movimiento con el ratón
    document.addEventListener('mousemove', (e) => {
        const mouseX = e.clientX / window.innerWidth;
        const mouseY = e.clientY / window.innerHeight;
        
        const particles = document.querySelectorAll('.bg-particle');
        particles.forEach((particle) => {
            const moveX = (mouseX - 0.5) * 20;
            const moveY = (mouseY - 0.5) * 20;
            particle.style.transform = `translate(${moveX}px, ${moveY}px)`;
        });
    });
}

// Inicializar el formulario
document.addEventListener("DOMContentLoaded", () => {
    // Configurar el formulario
    updateProgress();
    
    // Crear fondo interactivo
    createInteractiveBackground();

    // Añadir animación a las tarjetas de estilos de aprendizaje
    const styleCards = document.querySelectorAll(".learning-style-card");
    styleCards.forEach((card, index) => {
        // Aplicar animación con retraso escalonado
        card.style.animation = `slideUp 0.5s ease forwards ${index * 0.1 + 0.5}s`;
        card.style.opacity = "0";
    });

    // Manejar envío del formulario
    document.getElementById("profileForm").addEventListener("submit", (e) => {
        if (!validateSection(currentSection)) {
            e.preventDefault();
            return false;
        }
        return true;
    });
});
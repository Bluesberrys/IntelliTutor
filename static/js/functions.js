
let tiempoInicio = Date.now(); // Marca el tiempo de inicio
let tiempoTotal = 0; // Variable para almacenar el tiempo total

document.addEventListener("DOMContentLoaded", function () {
  const inicioBtn = document.querySelector("#inicio-btn");
  const burgerMenu = document.querySelector(".burger-menu");

  if (inicioBtn) {
      inicioBtn.addEventListener("click", () => {
          const cardsContainer = document.querySelector("#container-cards");
          if (cardsContainer) {
              cardsContainer.scrollIntoView({ behavior: "smooth" });
          }
      });
  }

  if (burgerMenu) {
      burgerMenu.addEventListener("click", () => {
          const headerMenu = document.querySelector(".menu");
          if (headerMenu) {
              headerMenu.classList.toggle("active");
          }
      });
  }
});

document.querySelector('form[action="{{ url_for("actualizar_rol") }}"]').addEventListener('submit', async function (e) {
    e.preventDefault(); // Prevenir el envío tradicional

    const form = e.target;
    const formData = new FormData(form);

    try {
        const response = await fetch(form.action, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            console.log("✅ " + data.message);
            showNotification('Rol Actualizado!!');
        } else {
            console.error("❌ " + data.message);
            alert("⚠️ Error: " + data.message);
        }
    } catch (error) {
        console.error("Error de red o del servidor:", error);
        showNotification('Error del servidor');
    }
});

// Función para registrar el tiempo en la API
function registrarTiempo(usuarioId, tiempo) {
  fetch('/api/registrar_tiempo', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json'
      },
      body: JSON.stringify({
          usuario_id: usuarioId,
          tiempo: tiempo
      })
  })
  .then(response => {
      if (!response.ok) {
          throw new Error('Error en la respuesta de la API');
      }
      return response.json();
  })
  .then(data => {
      if (data.success) {
          console.log('Tiempo registrado exitosamente');
      } else {
          console.error('Error al registrar tiempo:', data.error);
      }
  })
  .catch(error => console.error('Error en la solicitud:', error));
}

// Evento para calcular el tiempo al salir de la página
window.addEventListener('beforeunload', function() {
    let tiempoFin = Date.now(); // Marca el tiempo de fin
    tiempoTotal = Math.round((tiempoFin - tiempoInicio) / 1000); // Calcula el tiempo en segundos
    const usuarioId = 20; // Reemplaza con el ID del usuario actual

    // Llama a la función para registrar el tiempo
    registrarTiempo(usuarioId, tiempoTotal);
});
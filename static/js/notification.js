function showNotification(message, type = "success") {
  const notification = document.createElement("div");
  notification.classList.add("notification", type);

  let iconClass = "fas fa-check-circle";
  if (type === "error") {
    iconClass = "fas fa-exclamation-circle";
  } else if (type === "warning") {
    iconClass = "fas fa-exclamation-triangle";
  } else if (type === "danger") {
    iconClass = "fas fa-times-circle";
  }

  notification.innerHTML = `
    <i class="${iconClass}"></i>
    <p>${message}</p>
  `;

  document.body.appendChild(notification);

  // Reproducir sonido
  const audio = new Audio("/static/music/notificacion.mp3"); // Ajusta la ruta según tu estructura de carpetas

  audio.play().catch(error => {
    console.error("No se pudo reproducir el sonido:", error);
  });

  setTimeout(() => {
    notification.classList.add("fade-out");
    setTimeout(() => {
      if (document.body.contains(notification)) {
        document.body.removeChild(notification);
      }
    }, 300);
  }, 3000);
}

// Gestionar solicitud
function gestionarSolicitud(id, accion) {
    fetch('/gestionar_solicitudes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id, accion })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data); // Para depurar si data.message existe
        showNotification('Operación realizada.', 'success');
        setTimeout(() => {
            location.reload(); // Espera para que se vea la notificación
        }, 2000);
    })
}
document.addEventListener('DOMContentLoaded', function() {
    // Gestión de pestañas
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');
    
    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabId = button.getAttribute('data-tab');
            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            button.classList.add('active');
            document.getElementById(tabId).classList.add('active');
            localStorage.setItem('activeConfigTab', tabId);
        });
    });

    const activeTab = localStorage.getItem('activeConfigTab');
    if (activeTab) {
        const activeButton = document.querySelector(`.tab-btn[data-tab="${activeTab}"]`);
        if (activeButton) activeButton.click();
    }

    // Selector de tema
    const themeOptions = document.querySelectorAll('.theme-option');
    themeOptions.forEach(option => {
        option.addEventListener('click', () => {
            themeOptions.forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
            const radio = option.querySelector('input[type="radio"]');
            radio.checked = true;
            const theme = option.getAttribute('data-theme');
            applyTheme(theme);
        });
    });

    // Selector de color de acento
    const colorOptions = document.querySelectorAll('.color-option');
    colorOptions.forEach(option => {
        option.addEventListener('click', () => {
            colorOptions.forEach(opt => opt.classList.remove('selected'));
            option.classList.add('selected');
            const radio = option.querySelector('input[type="radio"]');
            radio.checked = true;
            const colorHex = option.getAttribute('data-color') || radio.value;
            applyAccentColor(colorHex);
        });
    });

    // Range sliders
    const rangeSliders = document.querySelectorAll('.range-slider input[type="range"]');
    rangeSliders.forEach(slider => {
        slider.addEventListener('input', () => {
            const valueDisplay = slider.nextElementSibling;
            valueDisplay.textContent = `${slider.value}px`;
            if (slider.name === 'tamano_fuente') {
                document.documentElement.style.setProperty('--font-size-base', `${slider.value}px`);
            } else if (slider.name === 'editor_tamano_fuente') {
                localStorage.setItem('editorFontSize', slider.value);
            }
        });
    });

    // Formulario General
    const generalForm = document.getElementById('generalForm');
    if (generalForm) {
        generalForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveFormSettings(generalForm, 'general');
        });
    }

    // Formulario Apariencia
    const aparienciaForm = document.getElementById('aparienciaForm');
    if (aparienciaForm) {
        aparienciaForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveFormSettings(aparienciaForm, 'apariencia');
        });
    }

    // Formulario Notificaciones
    const notificacionesForm = document.getElementById('notificacionesForm');
    if (notificacionesForm) {
        notificacionesForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveFormSettings(notificacionesForm, 'notificaciones');
        });
    }

    // Formulario Editor
    const editorForm = document.getElementById('editorForm');
    if (editorForm) {
        editorForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveFormSettings(editorForm, 'editor');
        });
    }

    // Formulario Privacidad
    const privacidadForm = document.getElementById('privacidadForm');
    if (privacidadForm) {
        privacidadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            await saveFormSettings(privacidadForm, 'privacidad');
        });
    }

    // Botón para probar notificaciones
    const testNotificationBtn = document.getElementById('testNotification');
    if (testNotificationBtn) {
        testNotificationBtn.addEventListener('click', () => {
            const notifNavegador = document.getElementById('notificacionesNavegador');
            if (notifNavegador && notifNavegador.checked) {
                if (Notification.permission !== 'granted') {
                    Notification.requestPermission().then(permission => {
                        if (permission === 'granted') {
                            showBrowserNotification('Notificación de prueba', 'Esta es una notificación de prueba del sistema.');
                        } else {
                            showSystemNotification('Permiso denegado', 'No se pueden mostrar notificaciones del navegador.', 'warning');
                        }
                    });
                } else {
                    showBrowserNotification('Notificación de prueba', 'Esta es una notificación de prueba del sistema.');
                }
            } else {
                showSystemNotification('Notificación de prueba', 'Esta es una notificación de prueba del sistema.', 'info');
            }
        });
    }

    // Botón para restablecer configuración del editor
    const resetEditorConfigBtn = document.getElementById('resetEditorConfig');
    if (resetEditorConfigBtn) {
        resetEditorConfigBtn.addEventListener('click', () => {
            const defaultConfig = {
                editor_tema: 'dracula',
                editor_tamano_fuente: '14',
                editor_lenguaje_default: 'python',
                editor_autocompletado: true,
                editor_line_numbers: true,
                editor_wrap_lines: true,
                editor_highlight_active_line: true
            };

            const form = document.getElementById('editorForm');
            form.querySelector('select[name="editor_tema"]').value = defaultConfig.editor_tema;
            form.querySelector('input[name="editor_tamano_fuente"]').value = defaultConfig.editor_tamano_fuente;
            form.querySelector('.range-value').textContent = `${defaultConfig.editor_tamano_fuente}px`;
            form.querySelector('select[name="editor_lenguaje_default"]').value = defaultConfig.editor_lenguaje_default;
            form.querySelector('#editorAutocompletado').checked = defaultConfig.editor_autocompletado;
            form.querySelector('#editorLineNumbers').checked = defaultConfig.editor_line_numbers;
            form.querySelector('#editorWrapLines').checked = defaultConfig.editor_wrap_lines;
            form.querySelector('#editorHighlightActiveLine').checked = defaultConfig.editor_highlight_active_line;

            saveFormSettings(form, 'editor');
        });
    }

    // Botón para exportar datos
    const exportarDatosBtn = document.getElementById('exportarDatos');
    if (exportarDatosBtn) {
        exportarDatosBtn.addEventListener('click', async () => {
            try {
                const response = await fetch('/exportar-datos');
                if (response.ok) {
                    const blob = await response.blob();
                    const url = window.URL.createObjectURL(blob);
                    const a = document.createElement('a');
                    a.style.display = 'none';
                    a.href = url;
                    a.download = 'mis_datos.json';
                    document.body.appendChild(a);
                    a.click();
                    window.URL.revokeObjectURL(url);
                    showSystemNotification('Datos exportados', 'Tus datos han sido exportados correctamente.', 'success');
                } else {
                    throw new Error('Error al exportar datos');
                }
            } catch (error) {
                console.error('Error al exportar datos:', error);
                showSystemNotification('Error', 'No se pudieron exportar tus datos.', 'error');
            }
        });
    }

    // Función para guardar configuración
    async function saveFormSettings(form, section) {
        const formData = new FormData(form);
        const settings = {};
        for (const [key, value] of formData.entries()) {
            if (form.querySelector(`input[name="${key}"][type="checkbox"]`)) {
                settings[key] = true;
            } else {
                settings[key] = value;
            }
        }
        form.querySelectorAll('input[type="checkbox"]').forEach(checkbox => {
            if (!checkbox.checked) {
                settings[checkbox.name] = false;
            }
        });
        settings.section = section;

        try {
            const response = await fetch('/configuracion/guardar', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(settings)
            });

            if (response.ok) {
                showSuccessAnimation();
                if (section === 'apariencia') {
                    if (settings.tema) applyTheme(settings.tema);
                    if (settings.color_acento) applyAccentColor(settings.color_acento);
                    if (settings.tamano_fuente) {
                        document.documentElement.style.setProperty('--font-size-base', `${settings.tamano_fuente}px`);
                    }
                } else if (section === 'editor') {
                    localStorage.setItem('editorTheme', settings.editor_tema);
                    localStorage.setItem('editorFontSize', settings.editor_tamano_fuente);
                    localStorage.setItem('editorLanguage', settings.editor_lenguaje_default);
                    localStorage.setItem('editorAutoComplete', settings.editor_autocompletado);
                    localStorage.setItem('editorLineNumbers', settings.editor_line_numbers);
                    localStorage.setItem('editorWrapLines', settings.editor_wrap_lines);
                    localStorage.setItem('editorHighlightActiveLine', settings.editor_highlight_active_line);
                }
            } else {
                throw new Error('Error al guardar la configuración');
            }
        } catch (error) {
            console.error('Error al guardar configuración:', error);
            showSystemNotification('Error', 'No se pudo guardar la configuración.', 'error');
        }
    }

    // Función para aplicar tema
    function applyTheme(theme) {
        const body = document.body;
        if (theme === 'system') {
            const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
            body.setAttribute('data-theme', prefersDarkMode ? 'dark' : 'light');
            window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
                if (body.getAttribute('data-theme-source') === 'system') {
                    body.setAttribute('data-theme', e.matches ? 'dark' : 'light');
                }
            });
            body.setAttribute('data-theme-source', 'system');
        } else {
            body.setAttribute('data-theme', theme);
            body.setAttribute('data-theme-source', 'user');
        }
        localStorage.setItem('theme', theme);
    }

    // ✅ Nueva función para aplicar color de acento personalizado
    function applyAccentColor(hex) {
        document.documentElement.style.setProperty('--accent-primary', hex);
        document.documentElement.style.setProperty('--accent-secondary', hex);
        document.documentElement.style.setProperty('--accent-tertiary', hex);
        localStorage.setItem('accentColor', hex);
    }

    // Función para mostrar la animación de éxito
    function showSuccessAnimation() {
        const checkAnimation = document.getElementById('checkAnimation');
        if (checkAnimation) {
            checkAnimation.classList.remove('hidden');
            
            // Ocultar después de la animación
            setTimeout(() => {
                checkAnimation.classList.add('hidden');
            }, 2500);
        }
    }
    
    // Función para mostrar notificación del sistema
    function showSystemNotification(title, message, type = 'info') {
        // Crear contenedor de notificaciones si no existe
        let container = document.getElementById('notification-container');
        if (!container) {
            container = document.createElement('div');
            container.id = 'notification-container';
            container.style.position = 'fixed';
            container.style.top = '20px';
            container.style.right = '20px';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        
        // Crear notificación
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <div class="notification-icon">
                <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'times-circle' : type === 'warning' ? 'exclamation-triangle' : 'info-circle'}"></i>
            </div>
            <div class="notification-content">
                <div class="notification-title">${title}</div>
                <div class="notification-message">${message}</div>
            </div>
            <button class="notification-close">
                <i class="fas fa-times"></i>
            </button>
        `;
        
        // Estilos para la notificación
        notification.style.display = 'flex';
        notification.style.alignItems = 'flex-start';
        notification.style.backgroundColor = 'var(--bg-primary)';
        notification.style.color = 'var(--text-primary)';
        notification.style.borderRadius = '8px';
        notification.style.boxShadow = '0 4px 12px rgba(0, 0, 0, 0.15)';
        notification.style.margin = '0 0 10px 0';
        notification.style.padding = '15px';
        notification.style.width = '300px';
        notification.style.animation = 'slideInRight 0.3s ease forwards';
        notification.style.borderLeft = `4px solid ${type === 'success' ? 'var(--success-color)' : type === 'error' ? 'var(--error-color)' : type === 'warning' ? 'var(--warning-color)' : 'var(--accent-primary)'}`;
        
        // Estilos para el icono
        const icon = notification.querySelector('.notification-icon');
        icon.style.marginRight = '15px';
        icon.style.fontSize = '24px';
        icon.style.color = type === 'success' ? 'var(--success-color)' : type === 'error' ? 'var(--error-color)' : type === 'warning' ? 'var(--warning-color)' : 'var(--accent-primary)';
        
        // Estilos para el contenido
        const content = notification.querySelector('.notification-content');
        content.style.flex = '1';
        
        // Estilos para el título
        const notificationTitle = notification.querySelector('.notification-title');
        notificationTitle.style.fontWeight = 'bold';
        notificationTitle.style.marginBottom = '5px';
        
        // Estilos para el botón de cerrar
        const closeButton = notification.querySelector('.notification-close');
        closeButton.style.background = 'none';
        closeButton.style.border = 'none';
        closeButton.style.color = 'var(--text-secondary)';
        closeButton.style.cursor = 'pointer';
        closeButton.style.fontSize = '16px';
        closeButton.style.marginLeft = '10px';
        
        // Añadir al contenedor
        container.appendChild(notification);
        
        // Cerrar al hacer clic en el botón
        closeButton.addEventListener('click', () => {
            notification.style.animation = 'slideOutRight 0.3s ease forwards';
            setTimeout(() => {
                container.removeChild(notification);
            }, 300);
        });
        
        // Cerrar automáticamente después de 5 segundos
        setTimeout(() => {
            if (container.contains(notification)) {
                notification.style.animation = 'slideOutRight 0.3s ease forwards';
                setTimeout(() => {
                    if (container.contains(notification)) {
                        container.removeChild(notification);
                    }
                }, 300);
            }
        }, 5000);
        
        // Añadir estilos de animación si no existen
        if (!document.getElementById('notification-styles')) {
            const style = document.createElement('style');
            style.id = 'notification-styles';
            style.textContent = `
                @keyframes slideInRight {
                    from {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                    to {
                        transform: translateX(0);
                        opacity: 1;
                    }
                }
                
                @keyframes slideOutRight {
                    from {
                        transform: translateX(0);
                        opacity: 1;
                    }
                    to {
                        transform: translateX(100%);
                        opacity: 0;
                    }
                }
            `;
            document.head.appendChild(style);
        }
    }
    
    // Función para mostrar notificación del navegador
    function showBrowserNotification(title, message) {
        if (Notification.permission === 'granted') {
            const notification = new Notification(title, {
                body: message,
                icon: '/static/img/Escudo-UNAM-white.svg'
            });
            
            notification.onclick = function() {
                window.focus();
                this.close();
            };
        }
    }
    
    // Cargar configuración inicial
    function loadInitialConfig() {
        // Cargar tema
        const savedTheme = localStorage.getItem('theme');
        if (savedTheme) {
            applyTheme(savedTheme);
        }
        
        // Cargar color de acento
        const savedAccentColor = localStorage.getItem('accentColor');
        if (savedAccentColor) {
            applyAccentColor(savedAccentColor);
        }
        
        // Cargar tamaño de fuente
        const savedFontSize = localStorage.getItem('fontSize');
        if (savedFontSize) {
            document.documentElement.style.setProperty('--font-size-base', `${savedFontSize}px`);
        }
    }
    
    // Cargar configuración inicial
    loadInitialConfig();
});
    
    
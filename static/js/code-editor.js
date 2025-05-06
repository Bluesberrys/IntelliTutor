document.addEventListener('DOMContentLoaded', function() {
    // Inicializar CodeMirror
    const codeEditor = CodeMirror.fromTextArea(document.getElementById('codeEditor'), {
        mode: 'python',
        theme: 'dracula',
        lineNumbers: true,
        indentUnit: 4,
        smartIndent: true,
        tabSize: 4,
        indentWithTabs: false,
        lineWrapping: true,
        autoCloseBrackets: true,
        matchBrackets: true,
        styleActiveLine: true
    });

    // Ajustar tamaño del editor
    codeEditor.setSize('100%', '100%');

    // Cargar configuración guardada
    loadEditorConfig();

    // Guardar configuración cuando cambie
    function saveEditorConfig() {
        const config = {
            language: document.getElementById('languageSelector').value,
            theme: document.getElementById('themeSelector').value
        };
        localStorage.setItem('editorConfig', JSON.stringify(config));
    }

    // Cargar configuración guardada
    function loadEditorConfig() {
        const savedConfig = localStorage.getItem('editorConfig');
        if (savedConfig) {
            const config = JSON.parse(savedConfig);
            document.getElementById('languageSelector').value = config.language || 'python';
            document.getElementById('themeSelector').value = config.theme || 'dracula';
            
            // Aplicar configuración
            changeLanguage(config.language);
            changeTheme(config.theme);
        }
    }

    // Cambiar lenguaje
    function changeLanguage(language) {
        let mode;
        switch (language) {
            case 'javascript':
                mode = 'javascript';
                break;
            case 'python':
                mode = 'python';
                break;
            case 'cpp':
                mode = 'text/x-c++src';
                break;
            case 'java':
                mode = 'text/x-java';
                break;
            default:
                mode = 'python';
        }
        codeEditor.setOption('mode', mode);
        document.getElementById('languageIndicator').textContent = language.charAt(0).toUpperCase() + language.slice(1);
        
        // Cargar código guardado para este lenguaje
        const savedCode = localStorage.getItem(`code_${language}`);
        if (savedCode) {
            codeEditor.setValue(savedCode);
        } else {
            // Código de ejemplo para cada lenguaje
            let sampleCode = '';
            switch (language) {
                case 'python':
                    sampleCode = '# Ejemplo de código Python\n\ndef saludar(nombre):\n    return f"Hola, {nombre}!"\n\n# Llamar a la función\nprint(saludar("Estudiante"))';
                    break;
                case 'javascript':
                    sampleCode = '// Ejemplo de código JavaScript\n\nfunction saludar(nombre) {\n    return `Hola, ${nombre}!`;\n}\n\n// Llamar a la función\nconsole.log(saludar("Estudiante"));';
                    break;
                case 'cpp':
                    sampleCode = '// Ejemplo de código C++\n\n#include <iostream>\n#include <string>\n\nusing namespace std;\n\nstring saludar(string nombre) {\n    return "Hola, " + nombre + "!";\n}\n\nint main() {\n    cout << saludar("Estudiante") << endl;\n    return 0;\n}';
                    break;
                case 'java':
                    sampleCode = '// Ejemplo de código Java\n\npublic class Main {\n    public static String saludar(String nombre) {\n        return "Hola, " + nombre + "!";\n    }\n    \n    public static void main(String[] args) {\n        System.out.println(saludar("Estudiante"));\n    }\n}';
                    break;
            }
            codeEditor.setValue(sampleCode);
        }
    }

    // Cambiar tema
    function changeTheme(theme) {
        codeEditor.setOption('theme', theme);
    }

    // Eventos para cambiar lenguaje y tema
    document.getElementById('languageSelector').addEventListener('change', function() {
        const language = this.value;
        changeLanguage(language);
        saveEditorConfig();
    });

    document.getElementById('themeSelector').addEventListener('change', function() {
        const theme = this.value;
        changeTheme(theme);
        saveEditorConfig();
    });

    // Actualizar posición del cursor
    codeEditor.on('cursorActivity', function() {
        const cursor = codeEditor.getCursor();
        document.getElementById('cursorPosition').textContent = `Línea: ${cursor.line + 1}, Columna: ${cursor.ch + 1}`;
    });

    // Guardar código automáticamente
    codeEditor.on('change', function() {
        const language = document.getElementById('languageSelector').value;
        localStorage.setItem(`code_${language}`, codeEditor.getValue());
        updateStatusMessage('Guardado automático');
    });

    // Función para ejecutar el código
    document.getElementById('runButton').addEventListener('click', function() {
        executeCode();
    });

    // Función para guardar el código
    document.getElementById('saveButton').addEventListener('click', function() {
        openSaveModal();
    });

    // Función para entregar el código
    document.getElementById('submitButton').addEventListener('click', function() {
        openSubmitModal();
    });

    // Ejecutar código
    function executeCode() {
        const code = codeEditor.getValue();
        const language = document.getElementById('languageSelector').value;
        
        if (!code.trim()) {
            showNotification('Por favor, escribe algo de código primero', 'warning');
            return;
        }
        
        // Mostrar indicador de carga
        const terminalOutput = document.getElementById('terminalOutput');
        terminalOutput.innerHTML += `<div class="terminal-command"><span class="terminal-prompt">$</span> Ejecutando código ${language}...</div>`;
        terminalOutput.innerHTML += `<div class="spinner"></div>`;
        terminalOutput.scrollTop = terminalOutput.scrollHeight;
        
        // Actualizar estado
        updateStatusMessage('Ejecutando código...');
        
        // Enviar código al servidor para ejecución real
        fetch('/ejecutar-codigo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                codigo: code,
                lenguaje: language
            })
        })
        .then(response => response.json())
        .then(data => {
            // Eliminar spinner
            const spinner = terminalOutput.querySelector('.spinner');
            if (spinner) {
                terminalOutput.removeChild(spinner);
            }
            
            if (data.success) {
                terminalOutput.innerHTML += `<div class="terminal-success">${formatOutput(data.output)}</div>`;
                updateStatusMessage('Ejecución completada');
            } else {
                terminalOutput.innerHTML += `<div class="terminal-error">${formatOutput(data.error)}</div>`;
                updateStatusMessage('Error en la ejecución');
            }
            
            terminalOutput.scrollTop = terminalOutput.scrollHeight;
        })
        .catch(error => {
            // Eliminar spinner
            const spinner = terminalOutput.querySelector('.spinner');
            if (spinner) {
                terminalOutput.removeChild(spinner);
            }
            
            terminalOutput.innerHTML += `<div class="terminal-error">Error al comunicarse con el servidor: ${error.message}</div>`;
            updateStatusMessage('Error de conexión');
            terminalOutput.scrollTop = terminalOutput.scrollHeight;
        });
    }

    // Formatear salida para el terminal
    function formatOutput(output) {
        if (!output) return '';
        
        // Escapar HTML
        output = output.replace(/&/g, '&amp;')
                       .replace(/</g, '&lt;')
                       .replace(/>/g, '&gt;')
                       .replace(/"/g, '&quot;')
                       .replace(/'/g, '&#039;');
        
        // Convertir saltos de línea en <br>
        output = output.replace(/\n/g, '<br>');
        
        return output;
    }

    // Abrir modal de guardado
    function openSaveModal() {
        const modal = document.getElementById('saveModal');
        modal.classList.add('active');
    }

    // Abrir modal de entrega
    function openSubmitModal() {
        const modal = document.getElementById('submitModal');
        const code = codeEditor.getValue();
        const language = document.getElementById('languageSelector').value;
        
        document.getElementById('submitCode').value = code;
        document.getElementById('submitLanguage').value = language;
        
        modal.classList.add('active');
    }

    // Cerrar modales
    document.getElementById('closeSaveModal').addEventListener('click', function() {
        document.getElementById('saveModal').classList.remove('active');
    });

    document.getElementById('closeSubmitModal').addEventListener('click', function() {
        document.getElementById('submitModal').classList.remove('active');
    });

    // Enviar formulario de guardado
    document.getElementById('saveForm').addEventListener('submit', function(e) {
        e.preventDefault();
        
        const fileName = document.getElementById('fileName').value;
        const description = document.getElementById('description').value;
        const practiceId = document.getElementById('practiceId').value;
        const code = codeEditor.getValue();
        const language = document.getElementById('languageSelector').value;
        
        fetch('/guardar-codigo', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                nombre_archivo: fileName,
                descripcion: description,
                practica_id: practiceId,
                codigo: code,
                lenguaje: language
            })
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                showNotification('Código guardado correctamente', 'success');
                document.getElementById('saveModal').classList.remove('active');
            } else {
                showNotification('Error al guardar el código: ' + data.error, 'error');
            }
        })
        .catch(error => {
            showNotification('Error al comunicarse con el servidor', 'error');
        });
    });

    submitForm.addEventListener('submit', async function (e) {
        e.preventDefault();
    
        const formData = new FormData(submitForm);
    
        // Agregar el código y lenguaje actual del editor
        formData.set('codigo', codeEditor.getValue());
        formData.set('lenguaje', document.getElementById('languageSelector').value);
    
        try {
            const response = await fetch(submitForm.action, {
                method: 'POST',
                body: formData,
            });
    
            if (response.ok) {
                // Cerrar modal de entrega
                document.getElementById('submitModal').classList.remove('active');
    
                // Mostrar notificación
                showNotification('Entrega realizada con éxito', 'success');
            } else {
                const error = await response.json();
                showNotification('Error al entregar: ' + (error.message || 'Desconocido'), 'error');
            }
        } catch (err) {
            console.error('Error en la entrega:', err);
            showNotification('Error de red al entregar la práctica', 'error');
        }
    });
    
    // Limpiar terminal
    document.getElementById('clearTerminal').addEventListener('click', function() {
        const terminalOutput = document.getElementById('terminalOutput');
        terminalOutput.innerHTML = '<div class="terminal-welcome"><span class="terminal-prompt">$</span> Terminal limpiada.</div>';
    });

    // Mostrar/ocultar terminal
    document.getElementById('toggleTerminal').addEventListener('click', function() {
        const terminalContainer = document.getElementById('terminalContainer');
        const icon = this.querySelector('i');
        
        if (terminalContainer.style.height === '30px') {
            terminalContainer.style.height = '200px';
            icon.className = 'fas fa-chevron-down';
        } else {
            terminalContainer.style.height = '30px';
            icon.className = 'fas fa-chevron-up';
        }
    });

    // Redimensionar terminal
    const resizer = document.getElementById('editorResizer');
    let startY, startHeight;

    resizer.addEventListener('mousedown', function(e) {
        startY = e.clientY;
        const terminalContainer = document.getElementById('terminalContainer');
        startHeight = parseInt(window.getComputedStyle(terminalContainer).height, 10);
        
        document.documentElement.addEventListener('mousemove', resize);
        document.documentElement.addEventListener('mouseup', stopResize);
    });

    function resize(e) {
        const terminalContainer = document.getElementById('terminalContainer');
        const newHeight = startHeight - (e.clientY - startY);
        
        if (newHeight > 50 && newHeight < window.innerHeight - 200) {
            terminalContainer.style.height = newHeight + 'px';
        }
    }

    function stopResize() {
        document.documentElement.removeEventListener('mousemove', resize);
        document.documentElement.removeEventListener('mouseup', stopResize);
    }

    // Actualizar mensaje de estado
    function updateStatusMessage(message) {
        const statusMessage = document.getElementById('statusMessage');
        statusMessage.textContent = message;
        
        // Resetear después de 3 segundos
        setTimeout(() => {
            if (statusMessage.textContent === message) {
                statusMessage.textContent = 'Listo';
            }
        }, 3000);
    }
});
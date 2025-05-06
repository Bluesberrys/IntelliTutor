document.addEventListener("DOMContentLoaded", () => {
  // Tema oscuro/claro - Prevenir flash al cargar la página
  const savedTheme = localStorage.getItem("theme") || "light"

  // Aplicar tema inmediatamente antes de que se renderice la página
  document.documentElement.style.setProperty("--transition-speed", "0s")
  document.body.setAttribute("data-theme", savedTheme)

  // Restaurar transiciones después de la carga
  setTimeout(() => {
    document.documentElement.style.setProperty("--transition-speed", "0.3s")
  }, 50)

  const themeToggle = document.getElementById("themeToggle")
  const body = document.body
  const themeIcon = themeToggle ? themeToggle.querySelector("i") : null

  // Actualizar el icono según el tema
  if (themeIcon) {
    if (savedTheme === "dark") {
      themeIcon.classList.remove("fa-moon")
      themeIcon.classList.add("fa-sun")
    } else {
      themeIcon.classList.remove("fa-sun")
      themeIcon.classList.add("fa-moon")
    }
  }

  // Manejar el cambio de tema - Arreglar el flash de modo oscuro
  if (themeToggle) {
    themeToggle.addEventListener("click", () => {
      const currentTheme = body.getAttribute("data-theme")
      const newTheme = currentTheme === "light" ? "dark" : "light"

      // Aplicar el tema inmediatamente para evitar el flash
      document.documentElement.style.setProperty("--transition-speed", "0s")
      body.setAttribute("data-theme", newTheme)
      localStorage.setItem("theme", newTheme)

      // Restaurar la transición después de un breve retraso
      setTimeout(() => {
        document.documentElement.style.setProperty("--transition-speed", "0.3s")
      }, 50)

      // Enviar el tema al servidor para guardarlo en la sesión
      fetch("/configuracion", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ tema: newTheme }),
      })

      // Cambiar el icono
      if (newTheme === "dark") {
        themeIcon.classList.remove("fa-moon")
        themeIcon.classList.add("fa-sun")
      } else {
        themeIcon.classList.remove("fa-sun")
        themeIcon.classList.add("fa-moon")
      }
    })
  }

  // Toggle mobile menu
  const burgerMenu = document.getElementById("burgerMenu")
  const navMenu = document.getElementById("navMenu")

  if (burgerMenu && navMenu) {
    burgerMenu.addEventListener("click", () => {
      navMenu.classList.toggle("active")
    })
  }

  // Tabs functionality
  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const tabId = tab.getAttribute("data-tab")

      // Remove active class from all tabs and contents
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"))
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"))

      // Add active class to current tab and content
      tab.classList.add("active")
      if (document.getElementById(tabId)) {
        document.getElementById(tabId).classList.add("active")
      }
    })
  })

  // File input display with filename
  document.querySelectorAll('input[type="file"]').forEach((input) => {
    input.addEventListener("change", (e) => {
      const fileName = e.target.files[0]?.name
      const fileLabel = e.target.nextElementSibling
      const submitButton = e.target.closest("form")?.querySelector('button[type="submit"]')

      if (fileName && fileLabel) {
        fileLabel.innerHTML = `<i class="fas fa-file"></i> ${fileName.substring(0, 15)}${fileName.length > 15 ? "..." : ""}`

        // Actualizar el texto del botón de envío
        if (submitButton) {
          submitButton.innerHTML = `<i class="fas fa-paper-plane"></i> Entregar "${fileName.substring(0, 10)}${fileName.length > 10 ? "..." : ""}"`
        }
      }
    })
  })

  // Course filter
  document.querySelectorAll(".course-filter-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const course = btn.getAttribute("data-course")

      // Remove active class from all buttons
      document.querySelectorAll(".course-filter-btn").forEach((b) => b.classList.remove("active"))

      // Add active class to clicked button
      btn.classList.add("active")

      // Filter rows
      document.querySelectorAll(".course-row").forEach((row) => {
        if (course === "all" || row.getAttribute("data-course") === course) {
          row.style.display = ""
        } else {
          row.style.display = "none"
        }
      })
    })
  })

  // Inicializar CodeMirror para cada editor independiente
  window.codeEditors = {}

  // Verificar fecha de entrega para cambiar estado
  document.querySelectorAll('form[action*="subir_archivo"]').forEach((form) => {
    const practicaId = form.getAttribute("action")?.split("/").pop()
    const fechaEntrega = form.getAttribute("data-fecha-entrega")

    if (fechaEntrega) {
      const fechaLimite = new Date(fechaEntrega)
      const ahora = new Date()

      if (ahora > fechaLimite) {
        // Añadir clase para indicar que está atrasado
        form.classList.add("entrega-atrasada")

        // Añadir un campo oculto para indicar que está atrasado
        const estadoInput = document.createElement("input")
        estadoInput.type = "hidden"
        estadoInput.name = "estado"
        estadoInput.value = "atrasado"
        form.appendChild(estadoInput)
      }
    }
  })

  // Función para mostrar mensajes flash
  window.showFlashMessage = (message, type = "info") => {
    const flashContainer = document.getElementById("flash-messages")
    if (!flashContainer) return

    const flashMessage = document.createElement("div")
    flashMessage.className = `flash-message ${type}`
    flashMessage.textContent = message

    const closeButton = document.createElement("button")
    closeButton.className = "close-button"
    closeButton.innerHTML = "&times;"
    closeButton.addEventListener("click", () => {
      flashContainer.removeChild(flashMessage)
    })

    flashMessage.appendChild(closeButton)
    flashContainer.appendChild(flashMessage)

    // Auto-remove after 5 seconds
    setTimeout(() => {
      if (flashContainer.contains(flashMessage)) {
        flashContainer.removeChild(flashMessage)
      }
    }, 5000)
  }
})

// Función para abrir el editor de código
window.openCodeEditor = (practicaId, title, form) => {
  const codeEditorModal = document.getElementById("codeEditorModal")
  const codeEditorTitle = document.getElementById("codeEditorTitle")
  const codeEditorOutput = document.getElementById("codeEditorOutput")
  const codeEditorSubmit = document.getElementById("codeEditorSubmit")

  if (codeEditorModal && codeEditorTitle) {
    codeEditorTitle.textContent = title || "Editor de Código"

    // Limpiar el área de salida
    if (codeEditorOutput) {
      codeEditorOutput.innerHTML = ""
      codeEditorOutput.style.display = "none"
    }

    // Marcar el formulario actual y guardar el ID de la práctica
    if (form) {
      form.setAttribute("data-editor", "true")
      form.setAttribute("data-practica-id", practicaId)
    }

    // Mostrar el modal
    codeEditorModal.classList.add("active")

    // Obtener o crear el editor para esta práctica
    const codeEditorElement = document.getElementById("codeEditor")

    if (codeEditorElement) {
      let codeEditor = window.codeEditors[practicaId]

      if (!codeEditor && typeof CodeMirror !== "undefined") {
        // Obtener configuración del editor
        const editorTheme = localStorage.getItem("editorTheme") || "dracula"
        const editorFontSize = localStorage.getItem("editorFontSize") || "14"
        const editorLanguage = localStorage.getItem("editorLanguage") || "javascript"

        try {
          codeEditor = CodeMirror.fromTextArea(codeEditorElement, {
            lineNumbers: true,
            mode: getLanguageMode(editorLanguage),
            theme: editorTheme,
            indentUnit: 4,
            smartIndent: true,
            tabSize: 4,
            indentWithTabs: false,
            lineWrapping: true,
            autoCloseBrackets: true,
            matchBrackets: true,
            autoCloseTags: true,
            matchTags: true,
            foldGutter: true,
            gutters: ["CodeMirror-linenumbers", "CodeMirror-foldgutter"],
          })

          // Aplicar tamaño de fuente
          codeEditor.getWrapperElement().style.fontSize = `${editorFontSize}px`

          // Guardar el editor para esta práctica
          window.codeEditors[practicaId] = codeEditor

          // Cargar contenido guardado si existe
          const savedContent = localStorage.getItem(`code_${practicaId}`)
          if (savedContent) {
            codeEditor.setValue(savedContent)
          }

          // Guardar automáticamente mientras se escribe
          codeEditor.on("change", () => {
            localStorage.setItem(`code_${practicaId}`, codeEditor.getValue())
          })
        } catch (error) {
          console.error("CodeMirror initialization error:", error)
          return
        }
      }

      // Asegurar que el editor se renderice correctamente
      if (codeEditor) {
        setTimeout(() => {
          codeEditor.refresh()
          codeEditor.focus()
        }, 10)
      }

      // Configurar el botón de envío
      if (codeEditorSubmit) {
        codeEditorSubmit.onclick = (event) => {
          const codeEditor = window.codeEditors[practicaId]
          if (!codeEditor) return

          const code = codeEditor.getValue()

          // Mostrar área de salida
          if (codeEditorOutput) {
            codeEditorOutput.style.display = "block"
            codeEditorOutput.innerHTML = "<div class='code-output-success'>Compilando código...</div>"
          }

          // Simular compilación
          setTimeout(() => {
            // Verificar si hay errores de sintaxis
            let hasError = false
            try {
              // Intentar evaluar el código (solo para JavaScript)
              if (codeEditor.getOption("mode") === "javascript") {
                new Function(code)
              }
            } catch (error) {
              hasError = true
              if (codeEditorOutput) {
                codeEditorOutput.innerHTML = `<div class='code-output-error'>Error de compilación: ${error.message}</div>`
              }
            }

            if (!hasError) {
              if (codeEditorOutput) {
                codeEditorOutput.innerHTML =
                  "<div class='code-output-success'>Compilación exitosa. Listo para entregar.</div>"
              }

              // Guardar el código en el formulario
              const currentForm = document.querySelector('form[data-editor="true"]')
              if (currentForm) {
                let contenidoInput = currentForm.querySelector('input[name="contenido"]')
                if (!contenidoInput) {
                  contenidoInput = document.createElement("input")
                  contenidoInput.type = "hidden"
                  contenidoInput.name = "contenido"
                  currentForm.appendChild(contenidoInput)
                }
                contenidoInput.value = code

                // Enviar el formulario automáticamente si se presiona el botón de enviar
                if (event.target.id === "codeEditorSubmit") {
                  // Cerrar el editor
                  codeEditorModal.classList.remove("active")

                  // Enviar el formulario
                  currentForm.submit()
                }
              }
            }
          }, 500)
        }
      }
    }
  }
}

// Función para obtener el modo de lenguaje para CodeMirror
function getLanguageMode(language) {
  switch (language) {
    case "javascript":
      return "javascript"
    case "python":
      return "python"
    case "java":
      return "text/x-java"
    case "cpp":
      return "text/x-c++src"
    default:
      return "javascript"
  }
}

// Cerrar el editor de código
document.addEventListener("DOMContentLoaded", () => {
  const codeEditorClose = document.getElementById("codeEditorClose")
  const codeEditorModal = document.getElementById("codeEditorModal")

  if (codeEditorClose && codeEditorModal) {
    codeEditorClose.addEventListener("click", () => {
      // Guardar el contenido en un campo oculto del formulario
      const currentForm = document.querySelector('form[data-editor="true"]')
      if (currentForm) {
        const practicaId = currentForm.getAttribute("data-practica-id")
        const codeEditor = window.codeEditors[practicaId]

        if (codeEditor) {
          let contenidoInput = currentForm.querySelector('input[name="contenido"]')
          if (!contenidoInput) {
            contenidoInput = document.createElement("input")
            contenidoInput.type = "hidden"
            contenidoInput.name = "contenido"
            currentForm.appendChild(contenidoInput)
          }
          contenidoInput.value = codeEditor.getValue()
        }
      }

      codeEditorModal.classList.remove("active")
    })
  }
})
// User dropdown menu
const userAvatar = document.getElementById("userAvatar")
const userDropdown = document.getElementById("userDropdown")

if (userAvatar && userDropdown) {
  // Añadir el marco de perfil al avatar en la barra de navegación
  const profileFrame = document.querySelector(".profile-frame")
  if (profileFrame && userAvatar) {
    // Crear el marco para el avatar en la barra de navegación si no existe
    // if (!userAvatar.querySelector(".profile-frame")) {
    //   const navFrame = document.createElement("div")
    //   navFrame.className = profileFrame.className
    //   userAvatar.insertBefore(navFrame, userAvatar.firstChild)
    // } else {
    //   // Actualizar el marco existente
    //   userAvatar.querySelector(".profile-frame").className = profileFrame.className
    // }
  }

  userAvatar.addEventListener("click", (e) => {
    e.stopPropagation()
    userDropdown.classList.toggle("active")
  })

  document.addEventListener("click", (e) => {
    if (!userDropdown.contains(e.target) && e.target !== userAvatar) {
      userDropdown.classList.remove("active")
    }
  })
}

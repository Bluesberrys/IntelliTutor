document.addEventListener("DOMContentLoaded", function () {
  const inicioBtn = document.querySelector("#inicio-btn");
  const cardsContainer = document.querySelector("#container-cards");
  const burgerMenu = document.querySelector(".burger-menu");
  const headerMenu = document.querySelector(".menu");

  // Scroll to cards section
  inicioBtn.addEventListener("click", () => {
    cardsContainer.scrollIntoView({ behavior: "smooth" });
  });
  // Toggle menu for responsive design
  burgerMenu.addEventListener("click", () => {
    headerMenu.classList.toggle("active");
  });
});

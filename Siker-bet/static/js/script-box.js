function showAgeConfirmation(event) {
    event.preventDefault();
    var ageConfirmationDialog = document.getElementById('ageConfirmationDialog');
    ageConfirmationDialog.classList.add('open');
}

function confirmAge() {
    var ageConfirmationDialog = document.getElementById('ageConfirmationDialog');
    ageConfirmationDialog.classList.remove('open');
    window.location.href = "/promo";
    // Aquí puedes agregar el código para redirigir a la página correspondiente después de confirmar la edad
}

function cancelAgeConfirmation() {
    var ageConfirmationDialog = document.getElementById('ageConfirmationDialog');
    ageConfirmationDialog.classList.remove('open');
    window.location.href = "/error";
    // Aquí puedes agregar el código para manejar la acción cuando el usuario cancela la confirmación de edad
}

const promoLink = document.getElementById('promoLink');
const ageConfirmationDialog = document.getElementById('ageConfirmationDialog');

promoLink.addEventListener('click', showAgeConfirmation);

const dialog = document.getElementById('ageConfirmationDialog');
dialog.addEventListener("click", function(event) {
    if (event.target === dialog) {
        dialog.style.display = 'none';
    }
});



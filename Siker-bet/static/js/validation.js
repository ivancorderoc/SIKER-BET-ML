$(document).ready(function () {
    const teamAcronyms = [
        "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN",
        "DET", "GB", "HOU", "IND", "JAX", "KC", "LV", "LAC", "LAR", "MIA",
        "MIN", "NE", "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS"
    ];

    // Populate text inputs with NFL team acronyms
    const teamHomeInput = document.getElementById("team_home_encoded");
    const teamAwayInput = document.getElementById("team_away_encoded");
    const teamFavoriteInput = document.getElementById("team_favorite_id_encoded");

    for (const teamAcronym of teamAcronyms) {
        const option = document.createElement("option");
        option.value = teamAcronym;
        option.textContent = teamAcronym;
        teamHomeInput.appendChild(option.cloneNode(true));
        teamAwayInput.appendChild(option.cloneNode(true));
        teamFavoriteInput.appendChild(option.cloneNode(true));
    }

    // Validate input fields before form submission
    $("#predictionForm").on("submit", function (event) {
        event.preventDefault();

        const totalPoints = parseInt($("#total_points").val());
        if (isNaN(totalPoints) || totalPoints < -10 || totalPoints > 30) {
            displayErrorMessage("total_points", "Total points must be a number between -10 and 30.");
            return;
        }

        const teamHome = $("#team_home_encoded").val();
        const teamAway = $("#team_away_encoded").val();
        const teamFavorite = $("#team_favorite_id_encoded").val();

        if (teamHome === teamAway || teamHome === teamFavorite || teamAway === teamFavorite) {
            displayErrorMessage("team_home_encoded", "Teams must be different.");
            displayErrorMessage("team_away_encoded", "Teams must be different.");
            displayErrorMessage("team_favorite_id_encoded", "Teams must be different.");
            return;
        }

        // If all validations pass, submit the form
        this.submit();
    });

    // Clear error messages on input focus
    $("input, select").on("focus", function () {
        const inputId = $(this).attr("id");
        hideErrorMessage(inputId);
    });

    function displayErrorMessage(inputId, message) {
        $("#" + inputId).addClass("error");
        $("#" + inputId + "_error").text(message).show();
    }

    function hideErrorMessage(inputId) {
        $("#" + inputId).removeClass("error");
        $("#" + inputId + "_error").hide();
    }
});

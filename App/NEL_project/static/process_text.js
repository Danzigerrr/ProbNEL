function showEntityDetails(entityText, entityTag, probabilities, uri) {
    const entityDetails = document.getElementById("entity-details");
    entityDetails.innerHTML += `
                <h5>Entity: ${entityText}</h5>
                <p><strong>Class:</strong> ${entityTag}</p>
                <p><strong>URI:</strong> <a href="${uri}" target="_blank">${uri}</a></p>
                <p><strong>Top Probabilities:</strong></p>
                <ul>
                    ${probabilities.map(prob => `<li>${prob[0]}: ${prob[1].toFixed(2)}</li>`).join("")}
                </ul>
                <hr>
            `;
}

function submitInput(event, knowledge_graph) {
    event.preventDefault();

    const userInput = document.getElementById("user_input").value;

    const loader = document.getElementById("loader-text-input");
    loader.style.display = "block";

    fetch("/NEL_app", {
        method: "POST",
        headers: {
            "Content-Type": "application/x-www-form-urlencoded",
            "X-Requested-With": "XMLHttpRequest",
        },
        body: new URLSearchParams({
            user_input: userInput,
            knowledge_graph: knowledge_graph
        })
    })
        .then(response => response.json())
        .then(data => {
            // Hide the loader
            loader.style.display = "none";

            const resultDiv = document.getElementById("result");

            clearCurrentEntityDetails();

            if (data.error) {
                resultDiv.innerHTML = `<p class="text-danger">Error: ${data.error}</p>`;
            } else {
                // Highlight entities in the text
                let textWithEntities = data.text;

                // Replace entities with clickable spans
                data.entities.forEach(entity => {
                    const { entity_label, probabilities, entity_type, best_candidate_uri } = entity;
                    const regex = new RegExp(`\\b${entity_label}\\b`, 'g'); // Match whole word
                    const entity_uri = best_candidate_uri;

                    // Replace text with a clickable link
                    textWithEntities = textWithEntities.replace(
                        regex,
                        `<a href="${entity_uri}" class="entity" target="_blank">${entity_label}</a>`
                    );

                    // Show entity details
                    showEntityDetails(entity_label, entity_type, probabilities, entity_uri);
                });


                document.getElementById("text-with-entities").innerHTML = textWithEntities;
            }
        })
        .catch(error => {
            console.error("Error:", error);
            const resultDiv = document.getElementById("result");
            resultDiv.innerHTML = `<p class="text-danger">An error occurred while processing your request.</p>`;
        });
}

function clearCurrentEntityDetails() {
    // Get the entity details container
    const entityDetailsTable = document.getElementById("entity-details-table");

    // Clear the container before adding new details
    entityDetailsTable.innerHTML = `
        <div id="entity-details"></div>
    `;
}
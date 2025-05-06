
// utility to build a table of scores for one candidate
function buildScoresTable(candidate) {
    return `
      <table class="table table-sm">
        <thead>
          <tr><th>Feature</th><th>Value</th></tr>
        </thead>
        <tbody>
          <tr><td>levenshtein</td><td>${candidate.score_levenshtein.toFixed(3)}</td></tr>
          <tr><td>popularity</td><td>${candidate.score_popularity.toFixed(3)}</td></tr>
          <tr><td>context</td><td>${candidate.score_context.toFixed(3)}</td></tr>
          <tr><td>position</td><td>${candidate.score_position.toFixed(3)}</td></tr>
          <tr><td>basic_types_embedding</td><td>${candidate.score_basic_types_embedding.toFixed(3)}</td></tr>
          <tr><td>topk_types_embedding</td><td>${candidate.score_topk_types_embedding.toFixed(3)}</td></tr>
          <tr><td>maxner_types_embedding</td><td>${candidate.score_maxner_types_embedding.toFixed(3)}</td></tr>
        </tbody>
      </table>`;
}

function fetchCandidateImage(label, imgElem) {
    const params = new URLSearchParams({
        action:    'query',
        format:    'json',
        prop:      'pageimages',
        titles:    label,
        piprop:    'thumbnail',
        pithumbsize: '300',
        origin:    '*'           // CORS
    });

    fetch('https://en.wikipedia.org/w/api.php?' + params.toString())
        .then(res => res.json())
        .then(json => {
            if (!json.query || !json.query.pages) {
                imgElem.style.display = 'none';
                return;
            }
            const pages = json.query.pages;
            // pages is an object keyed by pageid
            for (let pid in pages) {
                const page = pages[pid];
                if (page.thumbnail && page.thumbnail.source) {
                    imgElem.src = page.thumbnail.source;
                    imgElem.style.display = 'block';
                } else {
                    imgElem.style.display = 'none';
                }
                break;  // only first page
            }
        })
        .catch(_=>{
            imgElem.style.display = 'none';
        });
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
        .then(res => res.json())
        .then(data => {
            loader.style.display = "none";
            clearCurrentEntityDetails();

            if (data.error) {
                document.getElementById("result").innerHTML = `<p class="text-danger">Error: ${data.error}</p>`;
                return;
            }

            // render processed text with clickable links
            let textWithEntities = data.text;
            data.entities.forEach(ent => {
                const regex = new RegExp(`\\b${ent.entity_label}\\b`, 'g');
                textWithEntities = textWithEntities.replace(
                    regex,
                    `<a class="entity" href="${ent.best_candidate_uri}" target="_blank">${ent.entity_label}</a>`
                );
            });
            document.getElementById("text-with-entities").innerHTML = textWithEntities;

            // render accordion of entity details + candidate cards
            const container = document.getElementById("entity-details-table");
            data.entities.forEach((ent, i) => {
                const accId      = `acc-${i}`;
                const headingId  = `heading-${i}`;
                const collapseId = `collapse-${i}`;

                const rows = ent.candidates.map(c => {
                    const isBest = c.uri === ent.best_candidate_uri;
                    return `
          <div class="card ${isBest? 'border-success':''}">
            <div class="card-body p-2">
              <h4 class="${isBest? 'text-success fw-bold':'mb-1'}">
                ${isBest? '★ ': ''}<a href="${c.uri}" target="_blank">${c.label}</a>
              </h4>
              <div class="d-flex align-items-start mb-2">
                <p class="flex-grow-1 mb-0">${c.comment}</p>
                <img
                  class="candidate-thumb"
                  data-label="${c.label}"
                  alt="${c.label}"
                  style="max-width:20rem; margin:1rem; display:none;"
                />
              </div>
              ${buildScoresTable(c)}
            </div>
          </div>`;
                }).join("");

                container.innerHTML += `
        <div class="accordion mb-3" id="${accId}">
          <div class="accordion-item">
            <h2 class="accordion-header" id="${headingId}">
              <div class="d-flex align-items-center">
                <button 
                  class="accordion-button collapsed flex-grow-1" 
                  type="button"
                  data-bs-toggle="collapse" 
                  data-bs-target="#${collapseId}"
                  aria-expanded="false" 
                  aria-controls="${collapseId}">
                  Entity: ${ent.entity_label}
                </button>
                <a 
                  href="${ent.best_candidate_uri}" 
                  class="btn btn-link ms-2" 
                  target="_blank"
                  onclick="event.stopPropagation();">
                  View chosen URI
                </a>
              </div>
            </h2>
            <div id="${collapseId}" class="accordion-collapse collapse" aria-labelledby="${headingId}" data-bs-parent="#${accId}">
              <div class="accordion-body">
                ${rows}
              </div>
            </div>
          </div>
        </div>`;
            });

            // now fetch images for all placeholders
            document.querySelectorAll('img.candidate-thumb').forEach(imgElem => {
                const label = imgElem.getAttribute('data-label');
                fetchCandidateImage(label, imgElem);
            });
        })
        .catch(err => {
            console.error(err);
            document.getElementById("result").innerHTML =
                `<p class="text-danger">An error occurred.</p>`;
        });
}


function clearCurrentEntityDetails() {
    document.getElementById("entity-details-table").innerHTML = "";
}

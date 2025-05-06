
// utility to build a table of scores for one candidate
function buildScoresTable(candidate) {
    // read whether the user has checked “Use type‑score features”
    const useTypes = document.getElementById('use_types_score').checked;

    // always show these basic metrics
    let rows = `
    <tr><td>levenshtein</td><td>${candidate.score_levenshtein.toFixed(3)}</td></tr>
    <tr><td>popularity</td><td>${candidate.score_popularity.toFixed(3)}</td></tr>
    <tr><td>context</td><td>${candidate.score_context.toFixed(3)}</td></tr>
    <tr><td>position</td><td>${candidate.score_position.toFixed(3)}</td></tr>
  `;

    // only if the checkbox is selected, append the three embedding‑type scores
    if (useTypes) {
        rows += `
<!--      <tr><td>basic_types_embedding</td><td>${candidate.score_basic_types_embedding.toFixed(3)}</td></tr>-->
      <tr><td>topk_types_embedding</td><td>${candidate.score_topk_types_embedding.toFixed(3)}</td></tr>
      <tr><td>maxner_types_embedding</td><td>${candidate.score_maxner_types_embedding.toFixed(3)}</td></tr>
    `;
    }

    return `
    <table class="table table-sm">
      <thead>
        <tr><th>Feature (Score)</th><th>Value</th></tr>
      </thead>
      <tbody>
        ${rows}
      </tbody>
    </table>
  `;
}


function fetchCandidateImage(label, imgElem) {
    const params = new URLSearchParams({
        action:    'query',
        format:    'json',
        prop:      'pageimages',
        titles:    label,
        piprop:    'thumbnail',
        pithumbsize: '200',
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

const NER_MODEL_MAP = {
    conllpp:  'tomaarsen/span-marker-xlm-roberta-large-conllpp-doc-context',
    ontonotes5:'tomaarsen/span-marker-roberta-large-ontonotes5',
    fewnerd:   'tomaarsen/span-marker-bert-base-fewnerd-fine-super'
};

// Updated submitInput with hash-based badge coloring
function stringToColor(str) {
    // simple string hash to HSL hue
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = str.charCodeAt(i) + ((hash << 5) - hash);
    }
    const hue = Math.abs(hash) % 360;
    return `hsl(${hue}, 60%, 70%)`;
}

// Updated submitInput to display NER probabilities as badges in the header
function submitInput(event, knowledge_graph) {
    event.preventDefault();
    const userInput = document.getElementById("user_input").value;
    const nerKey = document.getElementById("ner_model").value;
    const nerModel = NER_MODEL_MAP[nerKey];
    const useTypes = document.getElementById("use_types_score").checked;

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
            knowledge_graph: knowledge_graph,
            ner_model: nerModel,
            use_types_score: useTypes ? '1' : '0'
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

                // build NER probability badges for header
                const nerBadges = ent.probabilities.map(p => {
                    const bgColor = stringToColor(p[0]);
                    return `<span class="badge ms-1" style="background-color:${bgColor}; color:#000">${p[0]}: ${p[1].toFixed(2)}</span>`;
                }).join('');

                // build probability list for body (if needed)
                const probsHtml = ent.probabilities.map(p =>
                    `<li>${p[0]}: ${p[1].toFixed(2)}</li>`
                ).join('');

                const rows = ent.candidates.map(c => {
                    const isBest = c.uri === ent.best_candidate_uri;
                    const typesHtml = c.ontology_types.map(t => `<span class="badge bg-secondary me-1">${t}</span>`).join('');
                    return `
          <div class="card ${isBest? 'border-success':''}">
            <div class="card-body p-2">
              <h4 class="${isBest? 'text-success fw-bold':'mb-1'}">
                ${isBest? '★ ': ''}<a href="${c.uri}" target="_blank">${c.label}</a>
              </h4>
              <div class="mb-2">
                <strong>Ontology types:</strong> ${typesHtml}
              </div>
              <div class="d-flex align-items-start mb-2">
                <p class="flex-grow-1 mb-0">${c.comment}</p>
                <img
                  class="candidate-thumb"
                  data-label="${c.label}"
                  alt="${c.label}"
                  style="max-width:15rem; margin:1rem; display:none;"
                />
              </div>
              ${buildScoresTable(c)}
            </div>
          </div>`;
                }).join("");

                container.innerHTML += `
        <div class="accordion mb-3" id="${accId}">
          <div class="accordion-item">
            <h2 class="accordion-header d-flex justify-content-between align-items-center" id="${headingId}">
              <button 
                class="accordion-button collapsed flex-grow-1" 
                type="button"
                data-bs-toggle="collapse" 
                data-bs-target="#${collapseId}"
                aria-expanded="false" 
                aria-controls="${collapseId}" 
                style="font-size: 0.6em;"
              >
                Entity: ${ent.entity_label}${nerBadges}
              </button>
              <a 
                href="${ent.best_candidate_uri}" 
                class="btn btn-link ms-2" 
                target="_blank"
                onclick="event.stopPropagation();"
              >
                View chosen URI
              </a>
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

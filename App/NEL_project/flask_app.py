from flask import Flask, render_template, request, jsonify, redirect

from App.NEL_project.NEL_app.Models.Text import Text
from App.NEL_project.NEL_app.NED_utlis.NEDHandler import NEDHandler
from App.NEL_project.NEL_app.NER_utils.NERConfig import NERConfig
from App.NEL_project.NEL_app.NER_utils.NERHandler import NERHandler

app = Flask(__name__)

def create_json_response(text_obj: Text):
    entities_data = []
    for entity in text_obj.entities:
        candidates_data = []
        for candidate in entity.candidates:
            candidates_data.append({
                "label": candidate.label,
                "ontology_types": candidate.ontology_types,
                "comment": candidate.comment,
                "uri": candidate.uri,
                "ref_count": candidate.ref_count,
                "position": candidate.position,
                "score_types_embeddings_similarity": candidate.score_types_embeddings_similarity,
                "score_levenshtein": candidate.score_levenshtein,
                "score_popularity": candidate.score_popularity,
                "score_context": candidate.score_context,
                "score_position": candidate.score_position,
                "score_basic_types_embedding": candidate.score_basic_types_embedding,
                "score_topk_types_embedding": candidate.score_topk_types_embedding,
                "score_maxner_types_embedding": candidate.score_maxner_types_embedding,
            })
        entities_data.append({
            "entity_label": entity.entity_label,
            "entity_type": entity.entity_type,
            "start_position": entity.start_position,
            "end_position": entity.end_position,
            "best_candidate_uri": entity.best_candidate_uri,
            "probabilities": [(prob[0], float(prob[1])) for prob in entity.probabilities],
            "candidates": candidates_data,
        })

    return jsonify({
        "text": text_obj.content,
        "entities": entities_data
    })

# --- Flask App Routes ---

@app.route("/")
def redirect_root():
    return redirect("/NEL_app")

@app.route("/NEL_app", methods=["GET", "POST"])
def nel_app():
    if request.method == "POST":
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            user_input      = request.form.get("user_input", "")
            knowledge_graph = request.form.get("knowledge_graph", "")
            ner_model        = request.form.get("ner_model")
            use_types_score  = request.form.get("use_types_score") == '1'

            if not user_input:
                return jsonify({"error": "Input text is required."}), 400

            try:
                text_obj = Text(user_input)
                ner_config = NERConfig(ner_model)
                ner = NERHandler(ner_config)
                text_obj = ner.perform_ner(text_obj)

                if knowledge_graph in ["dbpedia", "wikidata"]:
                    ned = NEDHandler(ner_config, knowledge_graph, "xgboost", use_types_score)
                    text_obj = ned.perform_ned(text_obj)
                    return create_json_response(text_obj)
                else:
                    return jsonify({"error": "Invalid knowledge_graph specified. Allowed values: dbpedia, wikidata"}), 400

            except Exception as e:
                return jsonify({"error": f"Error processing input: {str(e)}"}), 500

        return redirect("/NEL_app")  # Non-AJAX POST

    # GET request
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
def extract_entity_probabilities(entity):
    entity_probabilities = {}

    for token in entity:
        token_probabilities = token.get_tags_proba_dist("ner")
        for token_prob in token_probabilities:
            # Skip "O" class (non-entity tokens)
            if token_prob.value == 'O':
                label = "O"
            else:
                label = token_prob.value[2:]  # Remove the prefix (e.g., B-, I-, E-)
            score = token_prob.score
            entity_probabilities[label] = entity_probabilities.get(label, 0) + score / len(entity)

    # Sort probabilities by score in descending order
    sorted_probabilities = sorted(entity_probabilities.items(), key=lambda x: x[1], reverse=True)

    return sorted_probabilities[:3]


def get_entities_and_probabilities(sentence):
    """
    Extract entities and their top 3 class probabilities from a sentence.
    :param sentence: A Flair Sentence object containing the text.
    :return: A list of dictionaries containing entity text, start/end positions,
             entity group, and their top 3 probabilities.
    """
    ner_results = []

    # Iterate through the entities in the sentence
    for entity in sentence.get_spans('ner'):
        entity_probabilities = extract_entity_probabilities(entity)

        top_3_probabilities = "<ul>"
        for i, (label, probability) in enumerate(entity_probabilities):
            top_3_probabilities += f"<li>{label}: {probability:.4f}</li>"
        top_3_probabilities += "</ul>"

        ner_results.append({
            "text": entity.text,
            "start": entity.start_position,
            "end": entity.end_position,
            "entity_group": entity.get_label("ner").value,
            "probabilities": top_3_probabilities
        })

    return ner_results
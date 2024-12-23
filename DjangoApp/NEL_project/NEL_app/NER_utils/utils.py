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

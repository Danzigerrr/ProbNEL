from flair.models import SequenceTagger
from flair.data import Sentence
from ..classes import FoundEntity


class NERHandler:
    def __init__(self, ner_tagger_model_name: str = "flair/ner-english-ontonotes-fast"):
        """
        Initializes the NERHandler with a specified model.
        :param model: The name of the Flair NER model to load.
        """
        print(f"Loading NER model: {ner_tagger_model_name}")
        self.tagger = SequenceTagger.load(ner_tagger_model_name)
        print("Model NER loaded.")

    def process_text(self, user_input: str):
        """
        Processes the given text using the Flair NER model.
        :param user_input: The input text to process.
        :return: A Flair Sentence object annotated with NER tags.
        """
        sentence = Sentence(user_input)
        self.tagger.predict(sentence, return_probabilities_for_all_classes=True)
        return sentence

    def extract_entity_probabilities(self, entity):
        """
        Extracts the top probabilities for a given entity.
        :param entity: A Flair entity object.
        :return: A sorted list of the top 3 probabilities as tuples (label, score).
        """
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

    def extract_entities(self, text_with_ner_tags, text_obj):
        """
        Extracts entities from the NER-annotated text and associates them with a Text object.
        :param text_with_ner_tags: Flair Sentence object annotated with NER tags.
        :param text_obj: Text object to associate found entities with.
        :return: List of FoundEntity objects.
        """
        found_entities = []

        for entity in text_with_ner_tags.get_spans("ner"):
            probabilities = self.extract_entity_probabilities(entity)
            found_entities.append(
                FoundEntity(
                    text=text_obj,
                    entity_label=entity.text,
                    entity_type=entity.get_label("ner").value,
                    start_position=entity.start_position,
                    end_position=entity.end_position,
                    probabilities=probabilities
                )
            )

        return found_entities

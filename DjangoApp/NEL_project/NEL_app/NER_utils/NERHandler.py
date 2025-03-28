from flair.data import Sentence, Span
from ..Models.Text import Text
from ..Models.Entity import Entity
from .NERConfig import NERConfig


class NERHandler:
    tagger = None
    ignored_ner_types = []

    def perform_ner(self, text_obj: Text, tagger_name: str):
        """
        Processes the given text using the Flair NER model.
        :param tagger_name: name of the model used for NER tagging
        :param text_obj: A Flair Sentence object annotated with NER tags.
        """
        sentence = Sentence(text_obj.content)

        self.load_ner_configuration(tagger_name)

        self.tagger.predict(sentence, return_probabilities_for_all_classes=True)
        text_obj.entities = self.extract_entities(sentence)
        return text_obj

    def load_ner_configuration(self, tagger_name):
        ner_config = NERConfig(tagger_name)
        self.tagger = ner_config.tagger
        self.ignored_ner_types = ner_config.ignored_ner_types

    def extract_entities(self, sentence: Sentence):
        """
        Extracts entities from the NER-annotated text and associates them with a Text object.
        :return: List of FoundEntity objects.
        """
        found_entities = []

        for entity in sentence.get_spans("ner"):
            probabilities = self.extract_entity_probabilities(entity=entity)
            top_ner_type_name = probabilities[0][0]
            if top_ner_type_name not in self.ignored_ner_types:
                found_entities.append(
                    Entity(
                        entity_label=entity.text,
                        entity_type=entity.get_label("ner").value,
                        start_position=entity.start_position,
                        end_position=entity.end_position,
                        probabilities=probabilities,
                        best_candidate_uri=""
                    )
                )

        found_entities = found_entities
        return found_entities

    def extract_entity_probabilities(self, entity: Span, number_of_top_probabilities=3):
        """
        Extracts the top probabilities for a given entity.
        :param number_of_top_probabilities:
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

        return sorted_probabilities[:number_of_top_probabilities]

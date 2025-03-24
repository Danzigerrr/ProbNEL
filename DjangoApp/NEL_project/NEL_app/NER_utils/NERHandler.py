from flair.models import SequenceTagger
from flair.data import Sentence
from ..Models.Text import Text
from ..Models.Entity import Entity

tagger = SequenceTagger.load("flair/ner-english-ontonotes-fast", )


class NERHandler:
    found_entities = []  # list of FoundEntity
    sentence = None  # Flair sentence object

    ignored_ner_types = []

    def __init__(self, ner_tagger_model_name: str = "flair/ner-english-ontonotes-fast"):
        """
        Initializes the NERHandler with a specified model.
        :param model: The name of the Flair NER model to load.
        """
        print(f"Loading NER model: {ner_tagger_model_name}")
        # self.tagger = SequenceTagger.load(ner_tagger_model_name)
        self.tagger = tagger
        if ner_tagger_model_name == "flair/ner-english-ontonotes-fast":
            self.ignored_ner_types = ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "LANGUAGE"]
        print(f"Model NER \'{ner_tagger_model_name}\' loaded successfully")

    def perform_ner(self, text_obj: Text):
        """
        Processes the given text using the Flair NER model.
        :param text_obj: A Flair Sentence object annotated with NER tags.
        """
        self.sentence = Sentence(text_obj.content)
        self.tagger.predict(self.sentence, return_probabilities_for_all_classes=True)
        text_obj.entities = self.extract_entities(text_obj)

    def extract_entities(self, text_obj):
        """
        Extracts entities from the NER-annotated text and associates them with a Text object.
        :param text_obj: Text object to associate found entities with.
        :return: List of FoundEntity objects.
        """
        found_entities = []

        for entity in self.sentence.get_spans("ner"):
            probabilities = self.extract_entity_probabilities(entity=entity)
            top_ner_type_name = probabilities[0][0]
            if top_ner_type_name in self.ignored_ner_types:
                print(f"ignoring entity with typ ner type {top_ner_type_name}")
            else:
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

        self.found_entities = found_entities
        return found_entities

    def extract_entity_probabilities(self, entity, number_of_top_probabilities=3):
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

        return sorted_probabilities[:number_of_top_probabilities]

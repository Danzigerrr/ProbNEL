from ..Models.Text import Text
from ..Models.Entity import Entity
from .NERConfig import NERConfig
from .NERHelperFunctions import *
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')  # Download tokenizer model (needed only once)


class NERHandler:

    def __init__(self, ner_config: NERConfig):
        self.ner_config = ner_config

    def perform_ner(self, text_obj: Text):
        """
        Processes the given text using the NER model.
        :param text_obj: A Text object (with original text)
        """

        tokenized_text = word_tokenize(text_obj.content)

        predictions = predict_named_entities(self.ner_config.tagger_model, tokenized_text)

        predictions = map_word_indices_to_char_indices(text_obj.content, predictions)

        self.save_named_entities(text_obj, predictions)

        return text_obj


    def save_named_entities(self, text_obj: Text, predictions: List):
        """
        Extracts entities from the NER-annotated text and associates them with a Text object.
        :return: List of Entity objects.
        """

        for entity in predictions:
            top_ner_type_name = entity['labels'][0]
            if top_ner_type_name not in self.ner_config.ignored_ner_types:
                text_obj.entities.append(
                    Entity(
                        entity_label=" ".join(entity['span']),  # Corrected join syntax
                        entity_type=entity['labels'][0],  # Fixed dictionary access
                        start_position=entity['char_start_index'],
                        end_position=entity['char_end_index'],
                        probabilities=list(zip(entity['labels'], entity['scores'])),
                        best_candidate_uri=""
                    )
                )


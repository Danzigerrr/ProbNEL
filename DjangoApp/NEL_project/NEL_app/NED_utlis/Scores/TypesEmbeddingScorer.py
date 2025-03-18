import numpy as np
from flair.data import Sentence
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings
from scipy.spatial.distance import cosine
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity


class TypesEmbeddingScorer:
    """
    Class to calculate scores for candidates based on NER entity type and candidate ontology types.
    """
    ontonotes_classes = {
        "PERSON": "PERSON - A name referring to an individual, real or fictional.",
        "NORP": "NORP - A word that identifies a nationality, religious group, or political group.",
        "FAC": "FAC - The name of a man-made structure such as a building, airport, or bridge.",
        "ORG": "ORG - A name representing a company, agency, institution, or organized entity.",
        "GPE": "GPE - A geographic name used for political entities like countries, cities, or states.",
        "LOC": "LOC - A name associated with a geographic place that is not politically defined, such as a mountain or body of water.",
        "PRODUCT": "PRODUCT - The name of a manufactured item, vehicle, or consumable good.",
        "EVENT": "EVENT - A term referring to a specific occurrence, such as a war, festival, or natural disaster.",
        "WORK_OF_ART": "WORK OF ART - A title given to a creative work, including books, films, and paintings.",
        "LAW": "LAW - A name for an official legal document, regulation, or treaty.",
        "LANGUAGE": "LANGUAGE - A word identifying a system of communication spoken or written by a group of people.",
        "DATE": "DATE - A reference to a calendar-based point in time, including specific days or time periods.",
        "TIME": "TIME - An expression denoting a specific moment within a day, such as an hour or part of the day.",
        "PERCENT": "PERCENT - A numerical expression representing a proportion relative to 100, often using the '%' symbol.",
        "MONEY": "MONEY - A value that represents an amount of currency, including units like dollars or euros.",
        "QUANTITY": "QUANTITY - A measurement of an amount, such as weight, volume, or distance.",
        "ORDINAL": "ORDINAL - A number indicating position in a sequence, such as first or second.",
        "CARDINAL": "CARDINAL - A number representing a count or total, without implying order or rank.",
        "O": "O - Unknown"
    }

    def __init__(self, round_to_decimal_places=3):
        self.stacked_embeddings = StackedEmbeddings([
            WordEmbeddings('glove'),
            FlairEmbeddings('news-forward-fast'),
            FlairEmbeddings('news-backward-fast'),
        ])
        self.round_to_decimal_places = round_to_decimal_places

    def get_embedding(self, word):
        """Generate word embedding using Flair."""
        sentence = Sentence(word)
        self.stacked_embeddings.embed(sentence)
        return sentence[0].embedding.cpu().detach().numpy()

    def calculate_score(self, entity: Entity):
        """
        Calculates scores for candidates of an entity based on NER entity type and candidate ontology types.
        """
        if not entity.candidates:
            return

        ner_classes = [self.ontonotes_classes[item[0]] for item in entity.probabilities]
        ner_probabilities = np.array([float(item[1]) for item in entity.probabilities])

        for candidate in entity.candidates:
            kg_types = candidate.ontology_types

            ner_embeddings = {cls: self.get_embedding(cls) for cls in ner_classes}
            kg_embeddings = {kg_type: self.get_embedding(kg_type) for kg_type in kg_types}

            similarity_matrix = np.zeros((len(ner_classes), len(kg_types)))

            for i, ner_cls in enumerate(ner_classes):
                for j, kg_type in enumerate(kg_types):
                    similarity_matrix[i, j] = 1 - cosine(ner_embeddings[ner_cls], kg_embeddings[kg_type])

            final_scores = {}

            for j, kg_type in enumerate(kg_types):
                weighted_score = sum(ner_probabilities[i] * similarity_matrix[i, j] for i in range(len(ner_classes)))

                final_scores[kg_type] = weighted_score

            score_types_embeddings_similarity = 0
            for ontology_type in candidate.ontology_types:
                score_types_embeddings_similarity += sum(ner_probabilities * similarity_matrix[:, kg_types.index(ontology_type)])
            score_embeddings = score_types_embeddings_similarity/len(candidate.ontology_types) if len(candidate.ontology_types) > 0 else 0
            candidate.score_types_embeddings_similarity = round(score_embeddings, self.round_to_decimal_places)


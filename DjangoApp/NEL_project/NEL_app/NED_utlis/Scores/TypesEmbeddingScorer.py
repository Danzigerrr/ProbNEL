import numpy as np
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity
from sentence_transformers import SentenceTransformer, util


class TypesEmbeddingScorer:
    """
    Class to calculate scores for candidates based on NER entity type and candidate ontology types.
    """
    ontonotes_classes = {
        "PERSON": "PERSON - Proper names of people including first names, last names, individual or family names, fictional names and unique nicknames.",
        "NORP": "NORP - Adjectival forms of GPE and non-GPE place names (such as American), named religions, heritage, and political affiliation.",
        "FAC": "FAC - Names of man-made structures, including the buildings, airports, stations, infrastructures (bridges and streets), monuments, oil fields, golf courses, hospitals, zoos, shopping centers, etc.",
        "ORG": "ORG - Names of companies, government agencies, political parties, educational institutions, sport teams, hospitals, museums, libraries etc.",
        "GPE": "GPE - Names of geographical administrative entities including countries, villages, cities, states, provinces, prefectures, and other forms of municipalities",
        "LOC": "LOC - Names of locations other than GPEs including celestial bodies, stars, continents, mountains, oceans, coasts, rivers, lakes, borders, etc.",
        "PRODUCT": "PRODUCT - Name of any product including non-commercial vehicles (automobiles, rockets, aircraft, ships).",
        "EVENT": "EVENT - Named events and phenomena including natural disasters, hurricanes, revolutions, battles, wars, demonstrations, concerts, sports events, etc.",
        "WORK_OF_ART": "WORK OF ART - Titles of books, songs, films, plays and other creations such as awards, stock price indexes, and social security systems including health insurance systems or pension plans.",
        "LAW": "LAW - Named legal documents including laws, treaties, sections, and chapters.",
        "LANGUAGE": "LANGUAGE - Any named language including programming languages.",
        "DATE": "DATE - Date or period of 24 hours or more, including day, week, month, certain named period, season, year, etc.",
        "TIME": "TIME - Times of day and time duration less than 24 hours.",
        "PERCENT": "PERCENT - Percentage.",
        "MONEY": "MONEY - Monetary value.",
        "QUANTITY": "QUANTITY - Measurements including length, distance, area, weight, heat, velocity, temperature, byte size, etc.",
        "ORDINAL": "ORDINAL - Ordinal number.",
        "CARDINAL": "CARDINAL - Cardinal number.",
        "O": "O - Unknown"
    }

    def __init__(self, round_to_decimal_places=3):
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.round_to_decimal_places = round_to_decimal_places

    def get_embedding(self, sentence):
        """Generate word embedding using embeddings model."""
        embedding = self.embeddings_model.encode(sentence)
        return embedding

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
                    similarity_matrix[i, j] = 1 - util.cos_sim(ner_embeddings[ner_cls], kg_embeddings[kg_type])

            final_scores = {}

            for j, kg_type in enumerate(kg_types):
                weighted_score = sum(ner_probabilities[i] * similarity_matrix[i, j] for i in range(len(ner_classes)))

                final_scores[kg_type] = weighted_score

            score_types_embeddings_similarity = 0
            for ontology_type in candidate.ontology_types:
                score_types_embeddings_similarity += sum(ner_probabilities * similarity_matrix[:, kg_types.index(ontology_type)])
            score_embeddings = score_types_embeddings_similarity/len(candidate.ontology_types) if len(candidate.ontology_types) > 0 else 0
            candidate.score_types_embeddings_similarity = round(score_embeddings, self.round_to_decimal_places)


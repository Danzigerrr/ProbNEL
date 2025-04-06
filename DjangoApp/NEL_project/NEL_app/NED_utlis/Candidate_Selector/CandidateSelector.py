import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity

class CandidateSelector:
    def __init__(self, candidate_selection_strategy):
        self.candidate_selection_strategy = candidate_selection_strategy

        if candidate_selection_strategy == "sum_of_metrics":
            self.model = CandidateSelectorSumOfScores()
        elif candidate_selection_strategy == "candidate_selector_neural_network":
            self.model = CandidateSelectorNN()
            model_path_nn = os.path.join(os.path.dirname(__file__), "neural_network_model.pth")
            self.model.load_state_dict(torch.load(model_path_nn, map_location=torch.device('cpu')))
            self.model.eval()
        elif candidate_selection_strategy == "candidate_selector_random_forest_classifier":
            self.model = CandidateSelectorRFC()
        elif candidate_selection_strategy == "candidate_selector_svm":
            self.model = CandidateSelectorSVM()
        else:
            raise ValueError("Invalid candidate selection strategy")


class CandidateSelectorSumOfScores:
    def select_best_candidate_for_entity(self, entity: Entity):
        entity.best_candidate_uri = entity.candidates[0].uri


class CandidateSelectorRFC:
    def __init__(self):
        model_path_rfc = os.path.join(os.path.dirname(__file__), 'random_forest_model3.pth')
        self.model = torch.load(model_path_rfc, weights_only=False)

    def select_best_candidate_for_entity(self, entity: Entity):
        features = [
            attr for candidate in entity.candidates for attr in [
                candidate.score_types_embeddings_similarity,
                candidate.score_levenshtein_distance,
                candidate.score_popularity,
                candidate.score_context
            ]
        ]

        # Ensure the feature vector is exactly 40 elements long
        features.extend([0] * (40 - len(features)))
        features = np.array(features).reshape(1, -1)  # Reshape for a single entity

        best_candidate_index = int(self.model.predict(features))
        entity.best_candidate_uri = entity.candidates[best_candidate_index].uri if best_candidate_index < len(entity.candidates) else None



class CandidateSelectorSVM:
    def __init__(self):
        # Load the saved SVM model from the .pkl file
        model_path_svm = os.path.join(os.path.dirname(__file__), 'best_svm_model_1.pkl')
        self.model = joblib.load(model_path_svm)  # Load the model using joblib

        print("✅ SVM Model Loaded from .pkl")

    def select_best_candidate_for_entity(self, entity):
        # Extract features from candidates
        features = [
            attr for candidate in entity.candidates for attr in [
                candidate.score_types_embeddings_similarity,
                candidate.score_levenshtein_distance,
                candidate.score_popularity,
                candidate.score_context
            ]
        ]

        # Ensure the feature vector is exactly 40 elements long
        features.extend([0] * (40 - len(features)))  # Pad if necessary
        features = np.array(features).reshape(1, -1)  # Reshape for a single entity

        # Use the model to predict the best candidate
        best_candidate_index = int(self.model.predict(features))

        # Update the entity's best candidate
        entity.best_candidate_uri = entity.candidates[best_candidate_index].uri if best_candidate_index < len(entity.candidates) else None


import os
import numpy as np
import torch
import torch.nn as nn
import joblib
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity

class CandidateSelector:
    def __init__(self, candidate_selection_strategy, use_types_score):
        self.candidate_selection_strategy = candidate_selection_strategy
        self.use_types_score = use_types_score

        if candidate_selection_strategy == "sum_of_metrics":
            self.model = CandidateSelectorSumOfScores()
        elif candidate_selection_strategy == "candidate_selector_random_forest_classifier":
            self.model = CandidateSelectorRFC()
        elif candidate_selection_strategy == "candidate_selector_svm":
            self.model = CandidateSelectorSVM(use_types_score)
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
                candidate.score_context
            ]
        ]

        # Ensure the feature vector is exactly 40 elements long
        features.extend([0] * (40 - len(features)))
        features = np.array(features).reshape(1, -1)  # Reshape for a single entity

        best_candidate_index = int(self.model.predict(features))
        entity.best_candidate_uri = entity.candidates[best_candidate_index].uri if best_candidate_index < len(entity.candidates) else None


class CandidateSelectorSVM:
    def __init__(self, use_types_score: bool):
        """Initialize and load two SVM models (4-feature and 3-feature)."""
        base_dir = os.path.dirname(__file__)
        self.model_4_features = joblib.load(os.path.join(base_dir, 'best_svm_model_4_features_v1.pkl'))
        self.model_3_features = joblib.load(os.path.join(base_dir, 'best_svm_model_3_features_only_v1.pkl'))
        print("✅ Two SVM Models Loaded Successfully")
        self.use_types_score = use_types_score

    def _extract_features(self, entity: Entity) -> np.ndarray:
        """
        Extracts features for each candidate and flattens them into a single input vector.

        Args:
            entity: The entity object containing candidates with feature scores.

        Returns:
            A 1D numpy array of features, reshaped for model input.
        """
        # Select appropriate feature set
        if self.use_types_score:
            features = [
                attr for candidate in entity.candidates for attr in [
                    candidate.score_types_embeddings_similarity,
                    candidate.score_context
                ]
            ]
            expected_length = 40  # 10 candidates × 4 features
        else:
            features = [
                attr for candidate in entity.candidates for attr in [
                    candidate.score_context
                ]
            ]
            expected_length = 30  # 10 candidates × 3 features

        # Pad with zeros if fewer candidates were present
        if len(features) < expected_length:
            padding_size = expected_length - len(features)
            print(f"⚠️ Padding feature vector with {padding_size} zero(s).")
            features.extend([0.0] * padding_size)

        return np.array(features).reshape(1, -1)

    def select_best_candidate_for_entity(self, entity) -> None:
        """
        Selects the best candidate for a given entity using the appropriate SVM model.

        Args:
            entity: An object with a `.candidates` list and `.best_candidate_uri` attribute.
        """
        if len(entity.candidates) == 1:
            entity.best_candidate_uri = entity.candidates[0].uri
        else:
            features = self._extract_features(entity)

            # Choose the appropriate model
            model = self.model_4_features if self.use_types_score else self.model_3_features
            prediction = model.predict(features)

            # The model is expected to return an index (0–9) of the best candidate
            best_index = int(prediction[0])

            if 0 <= best_index < len(entity.candidates):
                entity.best_candidate_uri = entity.candidates[best_index].uri
            else:
                print(f"⚠️ Predicted index {best_index} is out of bounds.")
                entity.best_candidate_uri = "URI not defined"
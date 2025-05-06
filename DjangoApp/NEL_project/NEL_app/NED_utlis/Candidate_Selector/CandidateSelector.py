import os
import numpy as np
import joblib
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity


class CandidateSelector:


    def __init__(self, candidate_selection_strategy: str, use_types_score: bool):
        self.candidate_selection_strategy = candidate_selection_strategy
        self.use_types_score = use_types_score
        self.max_number_of_candidates = 10

        if use_types_score == False:
            self.number_of_features_for_candidate = 4
        else:
            self.number_of_features_for_candidate = 5

        self.max_number_of_features = self.max_number_of_candidates * self.number_of_features_for_candidate

        if candidate_selection_strategy == "sum_of_metrics":
            self.model = CandidateSelectorSumOfScores()
        elif candidate_selection_strategy == "xgboost":
            self.model = CandidateSelectorXGBoost(self.max_number_of_features, self.use_types_score)
        else:
            raise ValueError("Invalid candidate selection strategy")


class CandidateSelectorSumOfScores:
    def select_best_candidate_for_entity(self, entity: Entity):
        entity.best_candidate_uri = entity.candidates[0].uri

class CandidateSelectorXGBoost:
    model = None
    max_candidates_trained = 10  # Important: Should match the MAX_CANDIDATES used during training
    feature_names_base = ['levenshtein', 'context', 'popularity', 'position']
    use_types_score_trained = False  # Important: Should match if types score was used during training

    def __init__(self, max_number_of_features: int, use_types_score: bool):
        self.load_model('best_xgb_model_1_acc_0.8527.pkl')
        self.max_number_of_features = max_number_of_features
        self.use_types_score = use_types_score
        if self.use_types_score and not self.use_types_score_trained:
            print("Warning: 'use_types_score' is True, but the loaded model might not have been trained with type scores.")
        elif not self.use_types_score and self.use_types_score_trained:
            print("Warning: 'use_types_score' is False, but the loaded model was likely trained with type scores.")

    def load_model(self, model_name: str):
        model_path = os.path.join(os.path.dirname(__file__), model_name)
        try:
            self.model = joblib.load(model_path)
            # Infer the training configuration from the model (if possible)
            if hasattr(self.model, 'feature_names_in_'):
                if 'score_types' in self.model.feature_names_in_:
                    self.use_types_score_trained = True
        except FileNotFoundError:
            print(f"Error: Candidate selector model file not found at {model_path}")
            self.model = None

    def select_best_candidate_for_entity(self, entity: Entity):
        if self.model is None:
            print("Error: Model not loaded. Cannot select best candidate.")
            return

        features = []
        num_candidates = len(entity.candidates)

        for candidate in entity.candidates:
            candidate_features = [
                candidate.score_levenshtein,
                candidate.score_context,
                candidate.score_popularity,
                candidate.score_position,
            ]
            if self.use_types_score_trained:
                candidate_features.append(candidate.score_types if candidate.score_types is not None else 0)  # Handle missing type scores
            features.extend(candidate_features)

        # Pad or truncate the feature vector to match the expected length from training
        expected_length = self.max_candidates_trained * (4 + (1 if self.use_types_score_trained else 0))
        if len(features) < expected_length:
            features.extend([0] * (expected_length - len(features)))
        elif len(features) > expected_length:
            features = features[:expected_length]

        features_np = np.array(features).reshape(1, -1)  # Reshape for a single prediction
        print(f"[PYCHARM] Entity: {entity.entity_label}, Features before Prediction (shape: {features_np.shape}):\n {features_np}")

        try:
            best_candidate_index = self.model.predict(features_np)[0]
            if 0 <= best_candidate_index < num_candidates:
                entity.best_candidate_uri = entity.candidates[int(best_candidate_index)].uri
            else:
                entity.best_candidate_uri = None
                print(f"Warning: Predicted index {best_candidate_index} is out of bounds (number of candidates: {num_candidates}).")
        except Exception as e:
            print(f"Error during prediction: {e}")
            entity.best_candidate_uri = None


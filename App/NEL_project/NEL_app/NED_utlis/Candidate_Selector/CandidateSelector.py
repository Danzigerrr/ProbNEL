import os
import numpy as np
import joblib
from App.NEL_project.NEL_app.Models.Entity import Entity


class CandidateSelector:
    """
    A class to select the best candidate for a given entity using different strategies.
    Currently, supports "sum_of_metrics" and "xgboost".
    """
    def __init__(self, candidate_selection_strategy: str, use_types_score: bool):
        """
        Initializes the CandidateSelector.

        Args:
            candidate_selection_strategy (str): The strategy to use for candidate selection
                                                   ("sum_of_metrics" or "xgboost").
            use_types_score (bool): A flag indicating whether to use type-based scores
                                     (True for 6 features, False for 4 features).
        """
        self.candidate_selection_strategy = candidate_selection_strategy
        self.use_types_score = use_types_score
        self.max_number_of_candidates = 10  # Maximum number of candidates to consider

        # Determine the number of features per candidate based on whether type scores are used
        if not use_types_score:
            self.number_of_features_for_candidate = 4  # Levenshtein, context, popularity, position
        else:
            self.number_of_features_for_candidate = 6  # + topk_types, maxner_types

        # Calculate the total number of features expected for the XGBoost model
        self.max_number_of_features = self.max_number_of_candidates * self.number_of_features_for_candidate

        # Initialize the model based on the selected strategy
        if self.candidate_selection_strategy == "sum_of_metrics":
            self.model = CandidateSelectorSumOfScores()
        elif self.candidate_selection_strategy == "xgboost":
            self.model = CandidateSelectorXGBoost(self.max_number_of_features, self.use_types_score)
        else:
            raise ValueError("Invalid candidate selection strategy")


class CandidateSelectorSumOfScores:
    """
    A simple candidate selector that always selects the first candidate.
    This is used when the 'sum_of_metrics' strategy is chosen.
    """
    def select_best_candidate_for_entity(self, entity: Entity):
        """
        Selects the best candidate for an entity (currently always the first one).

        Args:
            entity (Entity): The entity for which to select the best candidate.
        """
        if entity.candidates:
            entity.best_candidate_uri = entity.candidates[0].uri
        else:
            entity.best_candidate_uri = None


class CandidateSelectorXGBoost:
    """
    A candidate selector that uses an XGBoost model to predict the best candidate.
    The model is loaded based on whether type-based scores are used (4 or 6 features).
    """
    model = None

    def __init__(self, max_number_of_features: int, use_types_score: bool):
        """
        Initializes the CandidateSelectorXGBoost.

        Args:
            max_number_of_features (int): The total number of features expected by the XGBoost model.
            use_types_score (bool): A flag indicating whether type-based scores were used during training.
        """
        self.max_number_of_features = max_number_of_features
        self.use_types_score = use_types_score

        # Load the appropriate XGBoost model based on the number of features
        if not self.use_types_score:
            self._load_model('xgb_model_enhanced_4_features.pkl')
        else:
            self._load_model('xgb_model_final_6_features.pkl')

    def _load_model(self, model_name: str):
        """
        Loads the XGBoost model from a pickle file.

        Args:
            model_name (str): The name of the pickle file containing the trained model.
        """
        model_path = os.path.join(os.path.dirname(__file__), model_name)
        try:
            self.model = joblib.load(model_path)
            print(f"✅ XGBoost model loaded from: {model_path} (using {'6' if self.use_types_score else '4'} features)")
        except FileNotFoundError:
            print(f"Error: Candidate selector model file not found at {model_path}")
            self.model = None

    def select_best_candidate_for_entity(self, entity: Entity):
        """
        Selects the best candidate for an entity using the loaded XGBoost model.

        Args:
            entity (Entity): The entity for which to select the best candidate.
        """
        if self.model is None:
            print("Error: Model not loaded. Cannot select best candidate.")
            return

        features = []
        num_candidates = len(entity.candidates)

        # Extract features for each candidate based on whether type scores are used
        if not self.use_types_score:
            for candidate in entity.candidates:
                candidate_features = [
                    candidate.score_levenshtein,
                    candidate.score_context,
                    candidate.score_popularity,
                    candidate.score_position,
                ]
                features.extend(candidate_features)
        else:
            for candidate in entity.candidates:
                candidate_features = [
                    candidate.score_levenshtein,
                    candidate.score_context,
                    candidate.score_popularity,
                    candidate.score_position,
                    candidate.score_topk_types_embedding,
                    candidate.score_maxner_types_embedding,
                ]
                features.extend(candidate_features)

        # Pad or truncate the feature vector to match the expected length from training
        if len(features) < self.max_number_of_features:
            padding_needed = self.max_number_of_features - len(features)
            features.extend([0.0] * padding_needed)  # Pad with zeros as float
        elif len(features) > self.max_number_of_features:
            features = features[:self.max_number_of_features]

        features_np = np.array(features).reshape(1, -1).astype(np.float32)  # Reshape and ensure float32
        # print(f"[PYCHARM] Entity: {entity.entity_label}, Features before Prediction (shape: {features_np.shape}):\n {features_np}")

        try:
            prediction = self.model.predict(features_np)
            if prediction.size > 0:
                best_candidate_index = prediction[0]
                if 0 <= best_candidate_index < num_candidates:
                    entity.best_candidate_uri = entity.candidates[int(best_candidate_index)].uri
                else:
                    entity.best_candidate_uri = None
                    print(f"Warning: Predicted index {best_candidate_index} is out of bounds (number of candidates: {num_candidates}).")
            else:
                entity.best_candidate_uri = None
                print("Warning: Prediction array is empty.")
        except Exception as e:
            print(f"Error during prediction: {e}")
            entity.best_candidate_uri = None



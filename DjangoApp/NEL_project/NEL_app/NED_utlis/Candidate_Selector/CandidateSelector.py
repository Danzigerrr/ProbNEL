import os
import torch
import torch.nn as nn
from DjangoApp.NEL_project.NEL_app.Models.Entity import Entity


class CandidateSelector:
    def __init__(self):
        # Load the pre-trained model
        self.model = CandidateSelectorNN()
        model_path = os.path.join(os.path.dirname(__file__), "model.pth")
        self.model.load_state_dict(torch.load(model_path, weights_only=False))
        self.model.eval()  # Set the model to evaluation mode

    def select_best_candidate_for_entity(self, entity: Entity):
        """
        Chooses the best candidate for an entity based on the calculated scores.
        """
        for candidate in entity.candidates:
            features = torch.tensor([candidate.score_types_embeddings_similarity,
                                     candidate.score_levenshtein_distance,
                                     candidate.score_popularity,
                                     candidate.score_context], dtype=torch.float32).unsqueeze(0)  # Shape: (1, 4)

            # Get the model's prediction score for this candidate
            with torch.no_grad():  # No need to track gradients during inference
                score = self.model(features)

            # Set the final score for the candidate
            candidate.score_final = score.item()

        # Sort candidates by their final score and select the best one
        entity.candidates.sort(key=lambda x: x.score_final, reverse=True)
        entity.best_candidate_uri = entity.candidates[0].uri  # Best candidate URI



class CandidateSelectorNN(nn.Module):
    def __init__(self):
        super(CandidateSelectorNN, self).__init__()
        self.fc1 = nn.Linear(4, 16)  # Input: 4 metrics -> Hidden layer
        self.fc2 = nn.Linear(16, 8)  # Hidden layer
        self.fc3 = nn.Linear(8, 1)   # Output: Score for each candidate

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x.squeeze(-1)  # Shape: (batch_size, 1)


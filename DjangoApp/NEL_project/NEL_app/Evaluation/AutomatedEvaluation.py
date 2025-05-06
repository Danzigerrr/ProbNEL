import json
import multiprocessing
import time
import os
import itertools
from datetime import datetime
from DjangoApp.NEL_project.NEL_app.Evaluation.DatasetLoader import DatasetLoader
from DjangoApp.NEL_project.NEL_app.Evaluation.EvaluationHandler import EvaluationHandler
from DjangoApp.NEL_project.NEL_app.Evaluation.EvaluationNED import EvaluationNED
from DjangoApp.NEL_project.NEL_app.Evaluation.EvaluationNER import EvaluationNER
from DjangoApp.NEL_project.NEL_app.NED_utlis.NEDHandler import NEDHandler
from DjangoApp.NEL_project.NEL_app.NER_utils.NERConfig import NERConfig
from DjangoApp.NEL_project.NEL_app.NER_utils.NERHandler import NERHandler


class AutomatedEvaluation:
    def __init__(self, dataset_path, ner_model, ned_knowledge_graph, ned_candidate_selection_strategy: str = "candidate_selector_neural_network",  ned_use_ontology_mapping_score=True, output_dir="evaluation_results"):
        self.dataset_path = dataset_path
        self.ner_model = ner_model
        self.ned_knowledge_graph = ned_knowledge_graph
        self.ned_candidate_selection_strategy = ned_candidate_selection_strategy
        self.ned_use_ontology_mapping_score = ned_use_ontology_mapping_score
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)  # Ensure output directory exists

    def load_evaluation_dataset_file(self):
        """Load the dataset from a JSON file."""
        try:
            with open(self.dataset_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None


    def run_evaluation(self):
        """Run evaluation on the dataset."""
        dataset_file = self.load_evaluation_dataset_file()
        if not dataset_file:
            return

        ned_evaluation_results = []
        start_time = time.time()
        # Load dataset using DatasetLoader
        dataset_loader = DatasetLoader()
        dataset = dataset_loader.load_dataset_content(dataset_file, self.dataset_path)
        dataset_loader.print_dataset_info(dataset)

        ner_config = NERConfig(self.ner_model)
        ner_handler = NERHandler(ner_config)
        ned_handler = NEDHandler(ner_config,
                                 self.ned_knowledge_graph,
                                 self.ned_candidate_selection_strategy,
                                 self.ned_use_ontology_mapping_score)

        # Run evaluation
        evaluation_handler = EvaluationHandler(ner_handler, ned_handler)
        ned_evaluation_results, ner_evaluation_results = evaluation_handler.run_test_on_dataset(dataset)

        ned_evaluation_results.print_results()
        ner_evaluation_results.print_results()
        # Save results
        self.save_results(ned_evaluation_results, ner_evaluation_results, start_time, dataset_loader)

    def save_results(self, ned_results: EvaluationNED, ner_results: EvaluationNER, start_time, dataset_loader):
        """Save evaluation results to a timestamped JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{self.output_dir}/evaluation_{timestamp}.json"

        output_data = {
            "configuration": {
                "dataset_path": self.dataset_path,
                "dataset_total_texts": dataset_loader.total_texts,
                "dataset_total_mentions": dataset_loader.total_mentions,
                "ner_model": self.ner_model,
                "ned_knowledge_graph": self.ned_knowledge_graph,
                "ned_candidate_selection_strategy": self.ned_candidate_selection_strategy,
                "ned_use_ontology_mapping_score": self.ned_use_ontology_mapping_score,
                "execution_time_seconds": round(time.time() - start_time, 2)
            },
            "ner_results": ner_results.to_json_dict(),
            "ned_results": ned_results.to_json_dict(),
            "ned_efficiency": round(ned_results.recall/ner_results.ner_accuracy, 2) if ner_results.ner_accuracy != 0 else 0
        }

        with open(output_filename, "w", encoding="utf-8") as file:
            json.dump(output_data, file, indent=4)

        print(f"Evaluation results saved to: {output_filename}")


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

    dataset_path = "./EvaluationDatasets/ace2004_short.json"
    ner_models = [
        "tomaarsen/span-marker-xlm-roberta-large-conllpp-doc-context",
        # "tomaarsen/span-marker-roberta-large-ontonotes5",
        # "tomaarsen/span-marker-bert-base-fewnerd-fine-super"
    ]
    ned_knowledge_graph = "dbpedia"
    # ned_candidate_selection_strategies = ["sum_of_metrics"]
    ned_candidate_selection_strategies = ["xgboost"]
    ned_use_ontology_mapping_scores = [True]

    # Iterate over all possible combinations of parameters
    for ner_model, ned_candidate_selection_strategy, ned_use_ontology_mapping_score in itertools.product(
            ner_models, ned_candidate_selection_strategies, ned_use_ontology_mapping_scores
    ):
        print(f"Running evaluation with:")
        print(f"NER Model: {ner_model}")
        print(f"NED Candidate Selection Strategy: {ned_candidate_selection_strategy}")
        print(f"Use Ontology Mapping Score: {ned_use_ontology_mapping_score}\n")

        evaluator = AutomatedEvaluation(
            dataset_path,
            ner_model,
            ned_knowledge_graph,
            ned_candidate_selection_strategy,
            ned_use_ontology_mapping_score
        )
        evaluator.run_evaluation()


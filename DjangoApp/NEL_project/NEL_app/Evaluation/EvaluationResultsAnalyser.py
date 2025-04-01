import os
import json
import pandas as pd

def load_json_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                content = json.load(file)
                content["filename"] = filename  # Store filename for reference
                data.append(content)
    return data

def create_comparison_table(data):
    rows = []
    for item in data:
        config = item.get("configuration", {})
        ner = item.get("ner_results", {})
        ned = item.get("ned_results", {})

        rows.append({
            "Filename": item["filename"],
            "Dataset": os.path.basename(config.get("dataset_path", "N/A")),
            "NER Model": config.get("ner_model", "N/A"),
            "KG": config.get("ned_knowledge_graph", "N/A"),
            "Candidate Strategy": config.get("ned_candidate_selection_strategy", "N/A"),
            "Use Ontology Mapping Score": config.get("ned_use_ontology_mapping_score", "N/A"),
            "Execution Time (s)": config.get("execution_time_seconds", 0),
            "NER Accuracy": ner.get("accuracy", 0),
            "NED Accuracy": ned.get("accuracy", 0),
            "NED Precision": ned.get("precision", 0),
            "NED Recall": ned.get("recall", 0),
            "NED F1 Score": ned.get("f1_score", 0),
            "NED Efficiency": item.get("ned_efficiency", 0),
        })

    return pd.DataFrame(rows)

def main():
    directory = "./evaluation_results/important"  # Change this to your actual directory
    data = load_json_files(directory)
    df = create_comparison_table(data)
    print(df)
    filename = "comparison_of_important_results.csv"
    df.to_csv(filename, index=False)
    print(f"Comparison table saved as '{filename}'")

if __name__ == "__main__":
    main()

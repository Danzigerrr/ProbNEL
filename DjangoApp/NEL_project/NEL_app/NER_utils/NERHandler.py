from span_marker import SpanMarkerModel
from ..Models.Text import Text
from ..Models.Entity import Entity
from .NERConfig import NERConfig
from typing import Union, List
from datasets import Dataset
from span_marker.trainer import Trainer
from tqdm.auto import trange
import torch
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')  # Download tokenizer model (needed only once)


class NERHandler:

    def __init__(self, ner_config: NERConfig):
        self.ner_config = ner_config

    def perform_ner(self, text_obj: Text):
        """
        Processes the given text using the Flair NER model.
        :param tagger_name: name of the model used for NER tagging
        :param text_obj: A Flair Sentence object annotated with NER tags.
        """

        self.ner_config.load_tagger_model()

        tokenized_text = word_tokenize(text_obj.content)
        predictions = self.predict_named_entites(self.ner_config.tagger_model, tokenized_text)
        predictions = self.map_word_indices_to_char_indices(text_obj.content, predictions)

        self.save_named_entities(text_obj, predictions)

        return text_obj


    def save_named_entities(self, text_obj: Text, predictions: List):
        """
        Extracts entities from the NER-annotated text and associates them with a Text object.
        :return: List of FoundEntity objects.
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


    def predict_named_entites(self,
                              model_obj: SpanMarkerModel,
                              inputs: Union[str, List[str], List[List[str]], Dataset],
                              top_n_types:int = 3,
                              batch_size: int = 4,
                              show_progress_bar=False):
        # Disable dropout, etc.
        model_obj.eval()

        if not inputs:
            return []

        # Determine input format and create a Dataset
        single_input = False
        if isinstance(inputs, str) or (
                isinstance(inputs, list) and all(isinstance(element, str) and " " not in element for element in inputs)
        ):
            single_input = True
            dataset = Dataset.from_dict({"tokens": [inputs]})
        elif isinstance(inputs, list):
            dataset = Dataset.from_dict({"tokens": inputs})
        elif hasattr(inputs, "column_names"):
            dataset = inputs
        else:
            raise ValueError(
                "`SpanMarkerModel.predict` could not recognize your input. Supported formats are string, list of strings, "
                "list of tokenized sentences, or a Dataset."
            )

        # Keep only relevant columns and add an 'id' column
        dataset = dataset.remove_columns(set(dataset.column_names) - {"tokens", "document_id", "sentence_id"})
        num_inputs = len(dataset)
        dataset = dataset.add_column("id", list(range(num_inputs)))
        results = [
            {
                "tokens": tokens,
                "scores": [],  # To store list of top-3 scores per candidate span.
                "labels": [],  # To store list of top-3 label indices per candidate span.
                "num_words": None,
            }
            for tokens in dataset["tokens"]
        ]

        # Tokenize & add start/end markers
        tokenizer_dict = model_obj.tokenizer(
            {"tokens": dataset["tokens"]},
            return_num_words=True,
            return_batch_encoding=True,
            return_offsets_mapping=True  # Add this line
        )

        batch_encoding = tokenizer_dict.pop("batch_encoding")
        dataset = dataset.remove_columns("tokens")
        for key, value in tokenizer_dict.items():
            dataset = dataset.add_column(key, value)

        # Add context if available
        if {"document_id", "sentence_id"} <= set(dataset.column_names):
            dataset = dataset.add_column("__sort_id", list(range(len(dataset))))
            dataset = dataset.sort(column_names=["document_id", "sentence_id"])
            dataset = Trainer.add_context(
                dataset,
                model_obj.tokenizer.model_max_length,
                max_prev_context=model_obj.config.max_prev_context,
                max_next_context=model_obj.config.max_next_context,
                show_progress_bar=show_progress_bar,
            )
            dataset = dataset.sort(column_names=["__sort_id"])
            dataset = dataset.remove_columns("__sort_id")

        dataset = dataset.map(
            Trainer.spread_sample,
            batched=True,
            desc="Spreading data between multiple samples",
            fn_kwargs={
                "model_max_length": model_obj.tokenizer.model_max_length,
                "marker_max_length": model_obj.config.marker_max_length,
            },
        )

        # Process in batches
        for batch_start_idx in trange(0, len(dataset), batch_size, leave=True, disable=not show_progress_bar):
            batch = dataset.select(range(batch_start_idx, min(len(dataset), batch_start_idx + batch_size)))
            batch = model_obj.data_collator(batch)
            batch = {key: value.to(model_obj.device) for key, value in batch.items()}
            with torch.no_grad():
                output = model_obj(**batch)
            # Compute probabilities from logits
            probs = output.logits.softmax(-1)
            # Instead of taking the maximum, retrieve the top 3 scores and indices per candidate span.
            topk_scores, topk_labels = torch.topk(probs, k=top_n_types, dim=-1)
            # print(topk_scores)
            # print(topk_labels)
            for iter_idx in range(output.num_marker_pairs.size(0)):
                input_id = dataset["id"][batch_start_idx + iter_idx]
                num_marker_pairs = output.num_marker_pairs[iter_idx]
                # For each sample, store top-3 scores and labels for candidate spans.
                results[input_id]["scores"].extend(topk_scores[iter_idx, :num_marker_pairs].tolist())
                results[input_id]["labels"].extend(topk_labels[iter_idx, :num_marker_pairs].tolist())
                results[input_id]["num_words"] = output.num_words[iter_idx]

        # Post-processing: aggregate candidate spans into final entity predictions.
        all_entities = []
        id2label = model_obj.config.id2label
        for sample_idx, sample in enumerate(results):
            candidate_scores = sample["scores"]  # List of lists (each candidate's top-3 scores)
            candidate_labels = sample["labels"]    # List of lists (each candidate's top-3 label indices)
            num_words = sample["num_words"]
            sentence = sample["tokens"]
            spans = list(model_obj.tokenizer.get_all_valid_spans(num_words, model_obj.config.entity_max_length))

            word_selected = [False] * num_words
            sentence_entities = []
            assert len(spans) == len(candidate_scores) and len(spans) == len(candidate_labels)
            # Sort candidate spans based on the highest score (first element of the top-3 list)
            for (word_start_index, word_end_index), score_list, label_list in sorted(
                    zip(spans, candidate_scores, candidate_labels),
                    key=lambda tup: tup[1][0],
                    reverse=True,
            ):
                if label_list[0] != model_obj.config.outside_id and not any(word_selected[word_start_index:word_end_index]):
                    offset_mapping = batch_encoding["offset_mapping"]
                    sample_offsets = offset_mapping[sample_idx]
                    char_start_index = sample_offsets[word_start_index][0]
                    char_end_index = sample_offsets[word_end_index - 1][1]
                    entity = {
                        "span": (sentence[char_start_index:char_end_index]
                                 if isinstance(sentence, str)
                                 else sentence[word_start_index:word_end_index]),
                        "scores": score_list,              # Top 3 scores.
                        "labels": [id2label[l] for l in label_list],  # Top 3 label strings.
                    }
                    if isinstance(sentence, str):
                        entity["char_start_index"] = char_start_index
                        entity["char_end_index"] = char_end_index
                    else:
                        entity["word_start_index"] = word_start_index
                        entity["word_end_index"] = word_end_index
                    sentence_entities.append(entity)
                    word_selected[word_start_index:word_end_index] = [True] * (word_end_index - word_start_index)
            all_entities.append(
                sorted(
                    sentence_entities,
                    key=lambda entity: entity["char_start_index"] if isinstance(sentence, str) else entity["word_start_index"],
                )
            )
        if single_input and len(all_entities) == 1:
            return all_entities[0]
        return all_entities


    def map_word_indices_to_char_indices(self, original_text: str, entities: list):
        """
        Maps word-based entity indices to character-based indices in the original text.

        :param original_text: The original, untokenized text.
        :param entities: A list of dictionaries containing 'word_start_index' and 'word_end_index' for entities.
        :return: The updated list with 'char_start_index' and 'char_end_index' added to each entity.
        """
        words = word_tokenize(original_text)
        char_positions = []
        current_pos = 0

        for word in words:
            start_pos = original_text.find(word, current_pos)
            end_pos = start_pos + len(word)
            char_positions.append((start_pos, end_pos))
            current_pos = end_pos

        for entity in entities:
            word_start_index = entity["word_start_index"]
            word_end_index = entity["word_end_index"]

            entity["char_start_index"] = char_positions[word_start_index][0]
            entity["char_end_index"] = char_positions[word_end_index - 1][1]

        return entities
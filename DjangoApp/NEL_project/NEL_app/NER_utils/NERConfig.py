from flair.models import SequenceTagger


class NERConfig:

    ALLOWED_MODEL_NAMES = ["flair/ner-english-ontonotes", "flair/ner-english"]

    def __init__(self, model_name: str = ALLOWED_MODEL_NAMES[0]):

        if model_name.lower() not in self.ALLOWED_MODEL_NAMES:
            raise ValueError(f"Error: model_name value must be on of the options: {self.ALLOWED_MODEL_NAMES}")

        self.model_name = model_name
        self.tagger = SequenceTagger.load(model_name)

        if model_name == self.ALLOWED_MODEL_NAMES[0]:
            self.ignored_ner_types = ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "LANGUAGE"]
        elif model_name == self.ALLOWED_MODEL_NAMES[1]:
            self.ignored_ner_types = []




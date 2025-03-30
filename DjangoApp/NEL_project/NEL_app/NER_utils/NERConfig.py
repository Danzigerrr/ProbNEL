from span_marker import SpanMarkerModel


class NERConfig:
    ALLOWED_TAGGER_NAMES = ["tomaarsen/span-marker-roberta-large-ontonotes5",
                            "tomaarsen/span-marker-xlm-roberta-large-conllpp-doc-context"]

    IGNORED_TYPES = {
        "flair/tomaarsen/span-marker-roberta-large-ontonotes5": ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "LANGUAGE"],
        "tomaarsen/span-marker-xlm-roberta-large-conllpp-doc-context": []
    }

    CLASS_DEFINITIONS = {
        "tomaarsen/span-marker-roberta-large-ontonotes5": {
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
        },
        "tomaarsen/span-marker-xlm-roberta-large-conllpp-doc-context": {
            "PER": "PER - Proper names of individuals, including first names, last names, fictional names, and unique nicknames.",
            "LOC": "LOC - Names of geographical locations such as cities, countries, states, provinces, and other physical locations.",
            "ORG": "ORG - Names of organizations including companies, government agencies, political parties, educational institutions, and sports teams.",
            "MISC": "MISC - Other named entities that do not fit into PER, LOC, or ORG categories, such as nationalities, artistic works, events, and other miscellaneous proper nouns.",
            "O": "O - Unknown"
        }
    }

    def __init__(self, tagger_name: str = None):
        self.tagger_name = tagger_name or self.ALLOWED_TAGGER_NAMES[0]

        if self.tagger_name not in self.ALLOWED_TAGGER_NAMES:
            raise ValueError(f"Error: tagger_name must be one of {self.ALLOWED_TAGGER_NAMES}")

        self.ignored_ner_types = self.IGNORED_TYPES.get(self.tagger_name, [])
        self.classes_definitions = self.CLASS_DEFINITIONS.get(self.tagger_name, {})
        self.tagger_model = None

    def load_tagger_model(self):
        """Load the SequenceTagger model."""
        self.tagger_model = SpanMarkerModel.from_pretrained(self.tagger_name)


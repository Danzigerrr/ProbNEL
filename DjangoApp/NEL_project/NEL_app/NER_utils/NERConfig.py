from flair.models import SequenceTagger


class NERConfig:
    ALLOWED_TAGGER_NAMES = ["flair/ner-english-ontonotes", "flair/ner-english"]

    tagger_name = None
    tagger_model = None
    ignored_ner_types = None
    classes_definitions = None

    def __init__(self, tagger_name: str = ALLOWED_TAGGER_NAMES[0]):
        if tagger_name.lower() not in self.ALLOWED_TAGGER_NAMES:
            raise ValueError(f"Error: tagger_name value must be on of the options: {self.ALLOWED_TAGGER_NAMES}")

        self.tagger_name = tagger_name
        self.set_ignored_ner_types(tagger_name)

    def set_ignored_ner_types(self, tagger_name):
        if tagger_name == self.ALLOWED_TAGGER_NAMES[0]:
            self.ignored_ner_types = ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "LANGUAGE"]
        elif tagger_name == self.ALLOWED_TAGGER_NAMES[1]:
            self.ignored_ner_types = []

    def load_tagger_model(self):
        if self.tagger_name == self.ALLOWED_TAGGER_NAMES[0] or self.tagger_name == self.ALLOWED_TAGGER_NAMES[1]:
            self.tagger_model = SequenceTagger.load(self.tagger_name)

    def set_classes_definitions(self):
        if self.tagger_name == self.ALLOWED_TAGGER_NAMES[0]:
            self.classes_definitions = {
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
            }
        elif self.tagger_name == self.ALLOWED_TAGGER_NAMES[1]:
            self.classes_definitions = {
                "PER": "PER - Proper names of individuals, including first names, last names, fictional names, and unique nicknames.",
                "LOC": "LOC - Names of geographical locations such as cities, countries, states, provinces, and other physical locations.",
                "ORG": "ORG - Names of organizations including companies, government agencies, political parties, educational institutions, and sports teams.",
                "MISC": "MISC - Other named entities that do not fit into PER, LOC, or ORG categories, such as nationalities, artistic works, events, and other miscellaneous proper nouns.",
                "O": "O - Unknown"
            }



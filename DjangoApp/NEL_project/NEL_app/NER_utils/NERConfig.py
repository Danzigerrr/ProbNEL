from span_marker import SpanMarkerModel


class NERConfig:
    ALLOWED_TAGGER_NAMES = ["tomaarsen/span-marker-xlm-roberta-large-conllpp-doc-context",
                            "tomaarsen/span-marker-roberta-large-ontonotes5",
                            "tomaarsen/span-marker-bert-base-fewnerd-fine-super"]

    IGNORED_TYPES = {
        "tomaarsen/span-marker-xlm-roberta-large-conllpp-doc-context": [],
        "tomaarsen/span-marker-roberta-large-ontonotes5": ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL", "LANGUAGE"],
        "tomaarsen/span-marker-bert-base-fewnerd-fine-super": ["other-language"]
    }

    CLASS_DEFINITIONS = {
        "tomaarsen/span-marker-xlm-roberta-large-conllpp-doc-context": {
            "PER": "PER - Proper names of individuals, including first names, last names, fictional names, and unique nicknames.",
            "LOC": "LOC - Names of geographical locations such as cities, countries, states, provinces, and other physical locations.",
            "ORG": "ORG - Names of organizations including companies, government agencies, political parties, educational institutions, and sports teams.",
            "MISC": "MISC - Other named entities that do not fit into PER, LOC, or ORG categories, such as nationalities, artistic works, events, and other miscellaneous proper nouns.",
            "O": "O - Unknown"
        },
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
        "tomaarsen/span-marker-bert-base-fewnerd-fine-super": {
            "art-broadcastprogram": "art-broadcastprogram - Broadcast programs including TV and radio shows.",
            "art-film": "art-film - Film titles and cinematic works.",
            "art-music": "art-music - Music-related works including performances, bands, and symphonies.",
            "art-other": "art-other - Other artistic creations not covered by film, music, painting, or written art.",
            "art-painting": "art-painting - Titles of paintings and art reproductions.",
            "art-writtenart": "art-writtenart - Literary and written art forms, including books and scripts.",
            "building-airport": "building-airport - Names of airports and aviation terminals.",
            "building-hospital": "building-hospital - Hospitals and medical facilities.",
            "building-hotel": "building-hotel - Hotels and lodging establishments.",
            "building-library": "building-library - Libraries and book repositories.",
            "building-other": "building-other - Other types of buildings not categorized elsewhere.",
            "building-restaurant": "building-restaurant - Restaurants and dining establishments.",
            "building-sportsfacility": "building-sportsfacility - Sports facilities and arenas.",
            "building-theater": "building-theater - Theaters and performance venues.",
            "event-attack/battle/war/militaryconflict": "event-attack/battle/war/militaryconflict - Military conflicts including attacks, battles, wars, and military engagements.",
            "event-disaster": "event-disaster - Natural or man-made disasters and catastrophic events.",
            "event-election": "event-election - Elections and voting events.",
            "event-other": "event-other - Other events not classified elsewhere.",
            "event-protest": "event-protest - Protests and demonstrations.",
            "event-sportsevent": "event-sportsevent - Sports events and competitions.",
            "location-GPE": "location-GPE - Geopolitical entities, typically countries, states, or cities.",
            "location-bodiesofwater": "location-bodiesofwater - Names of significant bodies of water such as lakes, rivers, and coasts.",
            "location-island": "location-island - Names of islands and archipelagos.",
            "location-mountain": "location-mountain - Mountain ranges, peaks, and related geographical features.",
            "location-other": "location-other - Other location names not covered by standard geographical categories.",
            "location-park": "location-park - Parks and recreational areas.",
            "location-road/railway/highway/transit": "location-road/railway/highway/transit - Transportation-related locations such as roads, railways, highways, and transit systems.",
            "organization-company": "organization-company - Companies and corporate entities.",
            "organization-education": "organization-education - Educational institutions and organizations.",
            "organization-government/governmentagency": "organization-government/governmentagency - Government bodies and agencies.",
            "organization-media/newspaper": "organization-media/newspaper - Media organizations and newspaper titles.",
            "organization-other": "organization-other - Other organizational entities not covered by other categories.",
            "organization-politicalparty": "organization-politicalparty - Political parties and related organizations.",
            "organization-religion": "organization-religion - Religious organizations and groups.",
            "organization-showorganization": "organization-showorganization - Organizations related to shows, entertainment, and performance groups.",
            "organization-sportsleague": "organization-sportsleague - Sports leagues and associations.",
            "organization-sportsteam": "organization-sportsteam - Sports teams and clubs.",
            "other-astronomything": "other-astronomything - Celestial objects and astronomy-related terms.",
            "other-award": "other-award - Awards and honors.",
            "other-biologything": "other-biologything - Biological terms and entities.",
            "other-chemicalthing": "other-chemicalthing - Chemical substances and compounds.",
            "other-currency": "other-currency - Currency symbols or names.",
            "other-disease": "other-disease - Diseases and medical conditions.",
            "other-educationaldegree": "other-educationaldegree - Academic degrees and certifications.",
            "other-god": "other-god - Deities and divine entities.",
            "other-language": "other-language - Language names or linguistic entities.",
            "other-law": "other-law - Legal terms and law-related entities.",
            "other-livingthing": "other-livingthing - Living organisms and fauna.",
            "other-medical": "other-medical - Medical specialties, professionals, or terms.",
            "person-actor": "person-actor - Actors and performers in film, television, or theater.",
            "person-artist/author": "person-artist/author - Artists and authors.",
            "person-athlete": "person-athlete - Athletes and sports figures.",
            "person-director": "person-director - Film or stage directors.",
            "person-other": "person-other - Other persons not categorized elsewhere.",
            "person-politician": "person-politician - Politicians and political figures.",
            "person-scholar": "person-scholar - Scholars, researchers, and academics.",
            "person-soldier": "person-soldier - Military personnel and soldiers.",
            "product-airplane": "product-airplane - Aircraft and airplanes.",
            "product-car": "product-car - Cars and automobiles.",
            "product-food": "product-food - Food items or brands.",
            "product-game": "product-game - Games or video game titles.",
            "product-other": "product-other - Other products or merchandise.",
            "product-ship": "product-ship - Ships and maritime vessels.",
            "product-software": "product-software - Software products or applications.",
            "product-train": "product-train - Trains and railway vehicles.",
            "product-weapon": "product-weapon - Weapons and armaments.",
            "O": "O - Unknown"
        }
    }


    def __init__(self, tagger_name: str = None):
        self.tagger_name = tagger_name or self.ALLOWED_TAGGER_NAMES[1]  # default Ontonotes - 18 classes

        if self.tagger_name not in self.ALLOWED_TAGGER_NAMES:
            raise ValueError(f"Error: tagger_name must be one of {self.ALLOWED_TAGGER_NAMES}")

        self.ignored_ner_types = self.IGNORED_TYPES.get(self.tagger_name, [])
        self.classes_definitions = self.CLASS_DEFINITIONS.get(self.tagger_name, {})

        self.tagger_model = SpanMarkerModel.from_pretrained(self.tagger_name)


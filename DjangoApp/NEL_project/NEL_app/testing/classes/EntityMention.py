class EntityMention:
    def __init__(self, surface_form, position_start, position_end, dbpedia_uri, wikidata_uri):
        self.surface_form = surface_form
        self.position_start = position_start
        self.position_end = position_end
        self.dbpedia_uri = dbpedia_uri
        self.wikidata_uri = wikidata_uri

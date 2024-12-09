from django.db import models

class Text(models.Model):
    content = models.TextField()

    def __str__(self):
        return self.content[:50]  # Display first 50 characters in admin

class Entity(models.Model):
    text = models.ForeignKey(Text, related_name="entities", on_delete=models.CASCADE)
    entity_text = models.CharField(max_length=255)
    entity_type = models.CharField(max_length=100)
    start_position = models.IntegerField()
    end_position = models.IntegerField()
    dbpedia_uri = models.URLField(null=True, blank=True)
    probabilities = models.JSONField(default=list)  # Store probabilities as JSON

    def __str__(self):
        return self.entity_text

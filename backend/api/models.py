from django.db import models

class Prediction(models.Model):
    image = models.ImageField(upload_to='predictions/')
    predicted_class = models.CharField(max_length=100)
    confidence = models.FloatField()
    all_predictions = models.JSONField(default=dict)
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.predicted_class} ({self.confidence:.2%})"

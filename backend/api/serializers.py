from rest_framework import serializers
from .models import Prediction

class PredictionSerializer(serializers.ModelSerializer):
    class Meta:
        model = Prediction
        fields = ['id', 'image', 'predicted_class', 'confidence', 'all_predictions', 'created_at']


class ImageUploadSerializer(serializers.Serializer):
    image = serializers.ImageField()

    def validate_image(self, value):
        if value.size > 5 * 1024 * 1024:  # 5MB max
            raise serializers.ValidationError("Image size must be less than 5MB")
        return value
 
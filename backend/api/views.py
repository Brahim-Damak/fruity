import json
import numpy as np
from pathlib import Path
from PIL import Image
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from .models import Prediction
from .serializers import PredictionSerializer, ImageUploadSerializer
import os

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Global variables
MODEL = None
CLASS_NAMES = []

def load_model():
    """Load model lazily (only when needed)"""
    global MODEL, CLASS_NAMES
    
    if MODEL is not None:
        return  # Already loaded
    
    print("⏳ Loading model...")
    try:
        # Import TensorFlow only when needed
        import tensorflow as tf
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        
        MODEL_PATH = Path(__file__).resolve().parent.parent.parent / 'models' / 'vegetables_model_best.h5'
        CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / 'models' / 'vegetables_model_config.json'
        
        print(f"Loading from: {MODEL_PATH}")
        print(f"Config from: {CONFIG_PATH}")
        
        # Check if files exist
        if not MODEL_PATH.exists():
            print(f"❌ Model file not found: {MODEL_PATH}")
            return
        if not CONFIG_PATH.exists():
            print(f"❌ Config file not found: {CONFIG_PATH}")
            return
        
        MODEL = tf.keras.models.load_model(str(MODEL_PATH))
        with open(CONFIG_PATH, 'r') as f:
            CONFIG = json.load(f)
        CLASS_NAMES = CONFIG.get('class_names', [])
        print(f"✅ Model loaded! Classes: {len(CLASS_NAMES)}")
        print(f"   Classes: {CLASS_NAMES}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        MODEL = None
        CLASS_NAMES = []


@api_view(['GET'])
def api_info(request):
    """Get API info"""
    return Response({
        'api_name': 'Vegetable Classifier API',
        'version': '1.0',
        'classes': CLASS_NAMES,
        'num_classes': len(CLASS_NAMES),
        'model_loaded': MODEL is not None,
    })


@api_view(['POST'])
def predict_vegetable(request):
    """Upload image and get prediction"""
    
    # Load model on first prediction
    load_model()
    
    if MODEL is None:
        return Response(
            {'error': 'Model not loaded. Check server logs for details.'},
            status=status.HTTP_503_SERVICE_UNAVAILABLE
        )
    
    serializer = ImageUploadSerializer(data=request.data)
    
    if not serializer.is_valid():
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
    
    try:
        # Get image
        image_file = serializer.validated_data['image']
        
        # Preprocess
        image = Image.open(image_file).convert('RGB')
        image_resized = image.resize((224, 224))
        image_array = np.array(image_resized) / 255.0
        image_batch = np.expand_dims(image_array, axis=0)
        
        # Predict
        predictions = MODEL.predict(image_batch, verbose=0)
        prediction_scores = predictions[0]
        
        # Get top prediction
        predicted_idx = np.argmax(prediction_scores)
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence = float(prediction_scores[predicted_idx])
        
        # All predictions
        all_predictions = {
            CLASS_NAMES[i]: float(prediction_scores[i])
            for i in range(len(CLASS_NAMES))
        }
        
        # Save
        prediction_obj = Prediction.objects.create(
            image=image_file,
            predicted_class=predicted_class,
            confidence=confidence,
            all_predictions=all_predictions
        )
        
        serializer = PredictionSerializer(prediction_obj)
        return Response(serializer.data, status=status.HTTP_200_OK)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return Response(
            {'error': str(e)},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


@api_view(['GET'])
def get_predictions(request):
    """Get all predictions"""
    predictions = Prediction.objects.all()[:50]
    serializer = PredictionSerializer(predictions, many=True)
    return Response(serializer.data)


@api_view(['GET'])
def get_prediction_detail(request, pk):
    """Get single prediction"""
    try:
        prediction = Prediction.objects.get(pk=pk)
        serializer = PredictionSerializer(prediction)
        return Response(serializer.data)
    except Prediction.DoesNotExist:
        return Response(
            {'error': 'Prediction not found'},
            status=status.HTTP_404_NOT_FOUND
        ) 
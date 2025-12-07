import json

# Read the unified class mapping
with open('C:/Users/USER/fruit_veg_mushroom_identifier/models/unified_class_mapping.json', 'r') as f:
    mapping = json.load(f)

# Extract class names
class_names = list(mapping.values()) if isinstance(mapping, dict) else mapping

# Create config
config = {
    'class_names': class_names,
    'num_classes': len(class_names),
    'model_type': 'EfficientNetB0',
    'input_size': [224, 224, 3]
}

# Save config
with open('C:/Users/USER/fruit_veg_mushroom_identifier/models/vegetables_model_config.json', 'w') as f:
    json.dump(config, f, indent=2)

print('✅ Config created!')
print(f'Classes: {class_names}')
print(f'Total: {len(class_names)} classes')
 
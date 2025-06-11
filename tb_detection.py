# Simple TB Detection Code for D:\project1\tb_dataset
# This is a simplified version for easy understanding

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set your dataset path here
DATA_PATH = r"D:\project1\tb_dataset"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("🚀 Starting TB Detection Project...")
print(f"📁 Dataset location: {DATA_PATH}")

# Check if dataset exists
if not os.path.exists(DATA_PATH):
    print(f"❌ Error: Dataset folder not found at {DATA_PATH}")
    print("Please make sure your tb_dataset folder exists with Normal and TB subfolders")
    exit()

# Check subfolders
normal_path = os.path.join(DATA_PATH, "Normal")
tb_path = os.path.join(DATA_PATH, "TB")

if not os.path.exists(normal_path):
    print(f"❌ Error: Normal folder not found at {normal_path}")
    exit()

if not os.path.exists(tb_path):
    print(f"❌ Error: TB folder not found at {tb_path}")
    exit()

# Count images
normal_count = len([f for f in os.listdir(normal_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
tb_count = len([f for f in os.listdir(tb_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

print(f"📊 Found {normal_count} normal images")
print(f"📊 Found {tb_count} TB images")
print(f"📊 Total images: {normal_count + tb_count}")

if normal_count == 0 or tb_count == 0:
    print("❌ Error: No images found in one or both folders")
    print("Please check that your images are in the correct folders")
    exit()

print("\n🔄 Setting up data generators...")

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    validation_split=0.2  # 20% for validation
)

# Training data
train_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)

# Validation data
val_generator = train_datagen.flow_from_directory(
    DATA_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)

print(f"✅ Training samples: {train_generator.samples}")
print(f"✅ Validation samples: {val_generator.samples}")
print(f"✅ Classes: {train_generator.class_indices}")

print("\n🏗️ Building the model...")

# Build model
model = Sequential([
    # First layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Second layer
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Third layer
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Fourth layer
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    
    # Dense layers
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')  # Binary: TB or Normal
])

# Compile model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("✅ Model built successfully!")
print("\n📋 Model Summary:")
model.summary()

print("\n🎯 Starting training...")
print("This will take 30-60 minutes depending on your computer and dataset size")
print("You can monitor the progress below...")

# Training callbacks
callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        'best_tb_model.h5',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

# Train the model
EPOCHS = 30  # Reduced for faster training, increase to 50 for better results

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=1
)

print("\n🎉 Training completed!")

# Save final model
model.save('tb_detection_final_model.h5')
print("✅ Model saved as 'tb_detection_final_model.h5'")

# Plot training history
print("\n📈 Creating training plots...")

plt.figure(figsize=(15, 5))

# Plot accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Plot loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Training plots saved as 'training_results.png'")

# Evaluate model
print("\n📊 Evaluating model...")

# Get predictions
val_generator.reset()
predictions = model.predict(val_generator, verbose=1)
predicted_classes = (predictions > 0.5).astype(int).flatten()

# Get true labels
true_classes = val_generator.classes
class_names = list(val_generator.class_indices.keys())

# Calculate accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(true_classes, predicted_classes)
print(f"\n🎯 Final Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Classification report
print("\n📋 Detailed Results:")
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Confusion matrix
cm = confusion_matrix(true_classes, predicted_classes)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

# Add numbers to confusion matrix
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j, i, cm[i, j], horizontalalignment="center", color="black")

plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ Confusion matrix saved as 'confusion_matrix.png'")

# Function to test single image
def predict_single_image(image_path):
    """Test a single image"""
    if not os.path.exists(image_path):
        print(f"❌ Image not found: {image_path}")
        return
    
    # Load and preprocess image
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)[0][0]
    
    # Interpret result
    if prediction > 0.5:
        result = "TB Detected"
        confidence = prediction * 100
    else:
        result = "Normal"
        confidence = (1 - prediction) * 100
    
    print(f"\n🔍 Prediction for {os.path.basename(image_path)}:")
    print(f"   Result: {result}")
    print(f"   Confidence: {confidence:.1f}%")
    print(f"   Raw score: {prediction:.4f}")
    
    return result, confidence

print("\n🎉 Project completed successfully!")
print("\n📝 Summary:")
print(f"   • Model accuracy: {accuracy*100:.1f}%")
print(f"   • Model saved as: tb_detection_final_model.h5")
print(f"   • Training plots: training_results.png")
print(f"   • Confusion matrix: confusion_matrix.png")

print("\n🚀 Next steps:")
print("   1. Run the web interface: streamlit run streamlit_app.py")
print("   2. Test with new images using the web interface")
print("   3. Check the saved plots to see how well your model trained")

# Example of testing a single image (uncomment to use)
# print("\n🧪 Testing single image prediction...")
# test_image_path = r"D:\project1\tb_dataset\Normal\some_image.jpg"  # Change this path
# if os.path.exists(test_image_path):
#     predict_single_image(test_image_path)
# else:
#     print("   No test image specified. You can test images using the web interface.")

print("\n✅ All done! Your TB detection model is ready to use.")
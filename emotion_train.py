import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# ? Step 1: Optimized Image Data Preprocessing
train_dir = "/home/pi/Downloads/train"
val_dir = "/home/pi/Downloads/test"

img_size = (64, 64)  # Smaller images for faster training
batch_size = 4  # Lower batch size to fit Raspberry Pi memory

train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=5,  # Lower rotation for speed
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.05,  # Minimal zoom
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# ? Step 2: Use Lightweight MobileNetV2 (Pretrained)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False  # Freeze layers

# ? Step 3: Custom Classification Head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  # Reduced number of neurons for speed
x = Dropout(0.3)(x)  # Dropout to prevent overfitting
x = Dense(len(train_generator.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# ? Step 4: Compile Model with Faster Optimizer
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# ? Step 5: Early Stopping to Prevent Wasted Time
early_stop = EarlyStopping(monitor='val_accuracy', patience=2, restore_best_weights=True)

# ? Step 6: Train Model (Only 5 Epochs for Speed)
epochs = 5  # Reduced epochs for 1-hour training

history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# ? Step 7: Fine-Tune Last 10 Layers for Better Accuracy
for layer in base_model.layers[-10:]:  # Unfreeze last 10 layers
    layer.trainable = True

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train again for quick fine-tuning (2 epochs)
fine_tune_epochs = 2
history_fine_tune = model.fit(
    train_generator,
    epochs=fine_tune_epochs,
    validation_data=val_generator,
    callbacks=[early_stop]
)

# ? Step 8: Save the Model
model.save("fast_model.h5")

# ? Step 9: Convert to TensorFlow Lite (For Fast Raspberry Pi Inference)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open("fast_model.tflite", "wb") as f:
    f.write(tflite_model)

# ? Step 10: Final Accuracy Check
_, final_accuracy = model.evaluate(val_generator)
print(f"Final Validation Accuracy: {final_accuracy * 100:.2f}%")

print("? Fast training complete! Ready for Raspberry Pi deployment ??")
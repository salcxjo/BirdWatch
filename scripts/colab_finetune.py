# BirdWatch Fine-Tuning — Google Colab
# See README for full instructions.
# Run in Colab with GPU runtime (Runtime → Change runtime type → T4 GPU)

from google.colab import files
import zipfile, os, json, tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, Model

# 1. Upload dataset.zip
uploaded = files.upload()
with zipfile.ZipFile('dataset.zip', 'r') as z:
    z.extractall('/content/')

# 2. Data generators
IMG_SIZE, BATCH_SIZE = 224, 32
datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=20, horizontal_flip=True,
    zoom_range=0.15, brightness_range=[0.8,1.2], validation_split=0.2
)
train_gen = datagen.flow_from_directory('/content/dataset', target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE, subset='training')
val_gen   = datagen.flow_from_directory('/content/dataset', target_size=(IMG_SIZE,IMG_SIZE),
    batch_size=BATCH_SIZE, subset='validation')
NUM_CLASSES = len(train_gen.class_indices)
print(f"Classes: {NUM_CLASSES}, Train: {train_gen.samples}, Val: {val_gen.samples}")

# 3. Build model
base = MobileNetV2(input_shape=(IMG_SIZE,IMG_SIZE,3), include_top=False, weights='imagenet')
base.trainable = False
x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(base.input, output)

# 4. Phase A — head only
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=10,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)])

# 5. Phase B — fine-tune top layers
base.trainable = True
for layer in base.layers[:-40]: layer.trainable = False
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_gen, validation_data=val_gen, epochs=20,
          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

# 6. Export
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
with open('/content/birdwatch_custom.tflite', 'wb') as f:
    f.write(converter.convert())
with open('/content/birdwatch_labels.json', 'w') as f:
    json.dump({str(v): k for k, v in train_gen.class_indices.items()}, f, indent=2)

files.download('/content/birdwatch_custom.tflite')
files.download('/content/birdwatch_labels.json')
print("Done — copy both files to ~/BirdWatch/model/ on your Pi")

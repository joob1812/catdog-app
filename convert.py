# Avec TensorFlow 2.15 installé localement
from tensorflow import keras

model = keras.models.load_model(
    "model/catdog_model.h5", compile=False
)  # charge l'ancien modèle
model.save("model/catdog_model_keras3.keras")  # nouveau format recommandé

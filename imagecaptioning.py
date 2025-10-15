# Step 1: Imports
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
import os

# Step 2: Load VGG16 Model for Image Feature Extraction
base_model = VGG16(weights='imagenet')
cnn_model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)

# Step 3: Extract Image Features
def extract_features(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    features = cnn_model.predict(x, verbose=0)
    return features

# Step 4: Tokenizer and Dummy Captions
captions = {
    "example.jpg": ["a dog is playing with a ball", "a puppy running in the grass"]
}

# Flatten all captions into one list
all_captions = []
for cap_list in captions.values():
    all_captions.extend(cap_list)

# Tokenize captions
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(cap.split()) for cap in all_captions)

# Step 5: Define the Image Captioning Model
def define_model(vocab_size, max_length):
    # Image feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Decoder (combine image + sequence)
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Final model
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# Step 6: Create the Model
model = define_model(vocab_size, max_length)

# Step 7: Show the Model Summary
model.summary()

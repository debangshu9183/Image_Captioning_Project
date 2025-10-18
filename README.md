# Image_Captioning_Project
This project implements an Image Captioning System using Deep Learning. The system generates natural language captions for input images by combining Convolutional Neural Networks (CNNs) for image feature extraction and LSTM networks for sequence modeling.

Features

Extracts features from images using VGG16 pre-trained model.

Cleans and preprocesses captions from the Flickr dataset.

Tokenizes text and converts captions to sequences for training.

Trains an encoder-decoder model to predict captions.

Generates predicted captions for both dataset images and new uploaded images.

Calculates BLEU scores for evaluating model performance.

Project Structure
image_captioning_project/
│
├─ Images/               # Folder containing dataset images
├─ working/              # Folder to store extracted features
├─ captions.txt          # File containing captions for images
├─ final_image_captioning_model.h5  # Trained captioning model
├─ feature_extractor.h5  # Optional: pre-trained VGG16 features
├─ image_captioning.py   # Main script for training and testing
└─ README.md             # Project documentation

Requirements

Python 3.9+ (3.11 recommended for compatibility with TensorFlow 2.13)

TensorFlow 2.x

Keras

Pillow

Matplotlib

Numpy

TQDM

NLTK (for BLEU score calculation)

Install required packages:

pip install tensorflow keras pillow matplotlib numpy tqdm nltk

Usage
1. Training the Model

The script image_captioning.py includes steps to:

Load images and extract features using VGG16.

Load and preprocess captions.

Tokenize text and determine maximum caption length.

Train the encoder-decoder LSTM model.

Save the trained model to final_image_captioning_model.h5.

# Run training
python image_captioning.py

2. Generating Captions for Dataset Images
from image_captioning import display_image_caption

display_image_caption('1000268201_693b08cb0e.jpg')


Prints true captions for the image.

Displays the image.

Prints the predicted caption.

3. Upload and Generate Caption for Your Own Image
from image_captioning import display_image_caption_new

# feature_extractor is the VGG16 model used for feature extraction
display_image_caption_new(model, tokenizer, max_length, mapping, feature_extractor)


Opens a file dialog to select any image.

Displays the image.

Prints the predicted caption.

Evaluation

BLEU-1 and BLEU-2 scores are calculated for the test set to measure the quality of generated captions:

from nltk.translate.bleu_score import corpus_bleu
print('BLEU-1:', corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('BLEU-2:', corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

Notes

Ensure images are stored in the Images/ folder for dataset testing.

For new images, the system extracts features dynamically using VGG16.

Captions are preprocessed with <start> and <end> tokens.

Adjust BASE_DIR and WORKING_DIR paths in the script according to your system.

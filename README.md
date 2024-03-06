# Deep Learning for Emotion Recognition: Exploring Image and Audio Modalities
## Overview
Emotion recognition, pivotal across human-computer interaction, affective computing, and psychology domains, encompasses discerning and comprehending human emotions expressed through diverse modalities like images and audio. Deep learning techniques, notably Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs), have revolutionized this field by deciphering intricate features from complex data. This project aims to leverage tailored deep learning models for image and audio data to enhance emotion recognition, optimizing recognition processes and improving overall system accuracy. Through separate models optimized for each modality's distinct traits, we seek to contribute to the advancement of emotion recognition technology and its applications across diverse domains.
## Data Sources

The data used in the notebooks for facial emotion recognition and emotion recognition from audio can be accessed from the following online sources:

### Facial Emotion Recognition Data:

- [CK+ dataset](https://www.kaggle.com/datasets/davilsena/ckdataset): The dataset comprises adapted data from the original CK+ dataset, containing up to 920 images. These images have been resized to 48x48 pixels, converted to grayscale format

### Audio Emotion Recognition Data:

- [CREMA-D](https://github.com/CheyneyComputerScience/CREMA-D): CREMA-D is a data set of 7,442 original clips from 91 actors. These clips were from 48 male and 43 female actors between the ages of 20 and 74 coming from a variety of races and ethnicities (African America, Asian, Caucasian, Hispanic, and Unspecified).Actors spoke from a selection of 12 sentences. The sentences were presented using one of six different emotions (Anger, Disgust, Fear, Happy, Neutral and Sad).
- [SAVEE](http://kahlan.eps.surrey.ac.uk/savee/Database.html): Surrey Audio-Visual Expressed Emotion (SAVEE) database has been recorded as a pre-requisite for the development of an automatic emotion recognition system. The database consists of recordings from 4 male actors in 7 different emotions, 480 British English utterances in total.

## Folder structure
The repository is organized into the following directories:

- **notebooks**: Contains Jupyter notebooks dedicated to facial emotion recognition and emotion recognition from audio.
  - **Facial expressions.ipynb**: Notebook exploring facial emotion recognition techniques using computer vision and deep learning.
  - **Audio.ipynb**: Notebook focusing on emotion recognition from audio signals, employing signal processing and machine learning approaches.

## Approach

In this project, we adopt an approach to emotion recognition, leveraging deep learning techniques tailored for image and audio data separately. The approach involves:

- Preprocessing: Preparing the data by resizing, converting to grayscale, and applying face detection for images, and extracting relevant features from audio signals.
- Model Development: Designing and training separate deep learning models for image and audio data, utilizing Convolutional Neural Networks (CNNs) and Recurrent Neural Networks (RNNs) respectively.
- Evaluation: Assessing the performance of the trained models using appropriate evaluation metrics and techniques.

## Dependencies

To run the notebooks and execute the code in this repository, you will need the following dependencies:

- Python 3.x
- Jupyter Notebook
- TensorFlow
- Librosa
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

You can install these dependencies using pip:

```bash
pip install tensorflow librosa opencv-python numpy pandas matplotlib scikit-learn
```
## Usage

To use the notebooks in this repository, clone the repository to your local machine:

```bash
git clone https://github.com/Ouma487/Emotion-recognition.git
```

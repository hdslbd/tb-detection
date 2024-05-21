## Chest X-ray Analysis for Tuberculosis Detection

This repository contains the code for a deep learning model that classifies chest X-ray images as normal or containing signs of tuberculosis (TB). 

### Project Goals

* Develop a prototype model to explore the feasibility of using deep learning for TB detection in chest X-rays.
* Gain experience with image classification tasks and building convolutional neural networks (CNNs).

### Data

The model is trained on a dataset of chest X-ray images categorized as normal and TB. 
* [Replace with details of your data source (if publicly available)]

**Note:** Due to privacy concerns, the dataset used for training might not be included in this repository.

### Model Architecture

The model utilizes a simple CNN architecture with convolutional layers for feature extraction, pooling layers for dimensionality reduction, and fully connected layers for classification.

### Getting Started

1. **Install required libraries:** Ensure you have TensorFlow, Keras, and other dependencies installed (refer to their documentation for installation instructions).
2. **Download the code:** Clone or download this repository to your local machine.
3. **Prepare your data:**  
   * If using your own data, organize it into folders named "normal" and "tb" containing the chest X-ray images.
   * Alternatively, if using a publicly available dataset, follow the data download and pre-processing instructions provided by the source.
4. **Update data paths:** Modify the `data_dir` variable in the `train.py` script to point to your data folder.
5. **Run the script:** Execute the `train.py` script to train the model. This script performs data preprocessing, trains the model, and saves it as "tb_detection_model.h5".

**Optional:**

* Modify the `train.py` script to adjust hyperparameters (epochs, batch size) or explore more complex CNN architectures.
* Utilize the saved model (`tb_detection_model.h5`) for prediction on new chest X-ray images (refer to TensorFlow documentation for model loading and prediction).

###  Disclaimer

This is a prototype model for educational purposes only. It should not be used for real-world medical diagnosis. 

### Resources

* TensorFlow Tutorials: [https://www.tensorflow.org/tutorials](https://www.tensorflow.org/tutorials)
* Keras Documentation: [https://keras.io/](https://keras.io/)
* Deep Learning for Chest X-ray Analysis (Research Paper): [Optional, replace with a relevant paper you used]

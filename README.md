# -Handwritten-Digit-Recognition-System
A Machine Learning-based Handwritten Digit Recognition System that accurately identifies and classifies handwritten digits (0–9) from images. The project uses image processing and deep learning techniques to recognize handwritten numbers, making it a practical application of computer vision and artificial intelligence.
"""
Handwritten Digit Recognition System using CNN
This module implements a Convolutional Neural Network for recognizing handwritten digits
from the MNIST dataset.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pickle
import os


class HandwrittenDigitRecognition:
    """
    A class to handle training and prediction of handwritten digits
    using a Convolutional Neural Network (CNN).
    """
    
    def __init__(self, model_path='digit_recognition_model.h5'):
        """
        Initialize the HandwrittenDigitRecognition system.
        
        Args:
            model_path (str): Path to save/load the trained model
        """
        self.model_path = model_path
        self.model = None
        self.history = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None
        
    def load_and_preprocess_data(self):
        """
        Load MNIST dataset and preprocess it for training.
        
        Returns:
            tuple: Preprocessed training and testing data
        """
        print("Loading MNIST dataset...")
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        
        # Normalize pixel values to [0, 1]
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        
        # Reshape data to include channel dimension (28, 28, 1)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)
        
        # Convert labels to one-hot encoding
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)
        
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        
        print(f"Training data shape: {x_train.shape}")
        print(f"Testing data shape: {x_test.shape}")
        
        return (x_train, y_train), (x_test, y_test)
    
    def build_model(self):
        """
        Build a Convolutional Neural Network model.
        
        Returns:
            keras.Model: Compiled CNN model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')  # 10 classes (digits 0-9)
        ])
        
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        print("\nModel architecture:")
        model.summary()
        
        return model
    
    def train(self, epochs=10, batch_size=128, validation_split=0.1):
        """
        Train the model on the MNIST dataset.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            validation_split (float): Fraction of data to use for validation
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if self.x_train is None:
            raise ValueError("Data not loaded. Call load_and_preprocess_data() first.")
        
        print(f"\nTraining model for {epochs} epochs...")
        self.history = self.model.fit(
            self.x_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Save the trained model
        self.model.save(self.model_path)
        print(f"\nModel saved to {self.model_path}")
        
        return self.history
    
    def evaluate(self):
        """
        Evaluate the model on the test dataset.
        
        Returns:
            tuple: Loss and accuracy on test data
        """
        if self.model is None:
            raise ValueError("Model not loaded. Train or load the model first.")
        
        print("\nEvaluating model on test data...")
        test_loss, test_accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        
        print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        return test_loss, test_accuracy
    
    def predict(self, image):
        """
        Predict the digit in a given image.
        
        Args:
            image (np.nd*

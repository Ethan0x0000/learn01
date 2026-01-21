#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SVM Classifier Implementation
-----------------------------
This script implements a Support Vector Machine (SVM) classifier using scikit-learn.
It supports Iris and Digits datasets, including data preprocessing, model training
(with optional GridSearch), evaluation, and visualization.

Author: Trae Assistant
Date: 2026-01-21
"""

import sys
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("svm_classifier.log", mode='w')
    ]
)
logger = logging.getLogger(__name__)

class SVMTask:
    """
    A class to handle SVM classification tasks including data loading,
    preprocessing, training, evaluation, and visualization.
    """

    def __init__(self, dataset_name='iris', test_size=0.2, random_state=42):
        """
        Initialize the SVM Task.

        Args:
            dataset_name (str): Name of the dataset ('iris' or 'digits').
            test_size (float): Proportion of the dataset to include in the test split.
            random_state (int): Random seed for reproducibility.
        """
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self.model = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.pca = None
        self.class_names = None

    def load_data(self):
        """
        Load the specified dataset from sklearn.datasets.
        """
        logger.info(f"Loading {self.dataset_name} dataset...")
        if self.dataset_name.lower() == 'iris':
            data = datasets.load_iris()
        elif self.dataset_name.lower() == 'digits':
            data = datasets.load_digits()
        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

        self.X = data.data
        self.y = data.target
        self.class_names = data.target_names
        logger.info(f"Data loaded. Shape: {self.X.shape}, Classes: {len(self.class_names)}")

    def preprocess(self):
        """
        Split the data into training and testing sets and standardize features.
        """
        logger.info("Preprocessing data...")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=self.random_state, stratify=self.y
        )
        logger.info(f"Data split: Train={self.X_train.shape[0]}, Test={self.X_test.shape[0]}")

        # Standardize features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        logger.info("Data standardization complete.")

    def train(self, use_grid_search=True):
        """
        Train the SVM model. Optionally uses GridSearchCV for hyperparameter tuning.

        Args:
            use_grid_search (bool): Whether to perform Grid Search for parameter tuning.
        """
        logger.info("Starting model training...")

        if use_grid_search:
            logger.info("Performing Grid Search for hyperparameter tuning...")
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'gamma': [1, 0.1, 0.01, 0.001],
                'kernel': ['rbf', 'linear']
            }
            grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=1, cv=5)
            grid.fit(self.X_train, self.y_train)
            
            self.model = grid.best_estimator_
            logger.info(f"Best parameters found: {grid.best_params_}")
        else:
            logger.info("Training with default parameters (kernel='rbf')...")
            self.model = SVC(kernel='rbf', probability=True)
            self.model.fit(self.X_train, self.y_train)

        logger.info("Model training complete.")

    def evaluate(self):
        """
        Evaluate the trained model on the test set.
        """
        if self.model is None:
            logger.error("Model not trained yet!")
            return

        logger.info("Evaluating model...")
        y_pred = self.model.predict(self.X_test)

        acc = accuracy_score(self.y_test, y_pred)
        logger.info(f"Accuracy: {acc:.4f}")

        logger.info("\nClassification Report:\n" + classification_report(self.y_test, y_pred, target_names=self.class_names.astype(str)))
        logger.info("\nConfusion Matrix:\n" + str(confusion_matrix(self.y_test, y_pred)))

    def visualize(self):
        """
        Visualize decision boundaries using PCA (reducing to 2D).
        Note: This is an approximation for high-dimensional data.
        """
        if self.model is None:
            logger.warning("Model not trained, cannot visualize.")
            return

        logger.info("Generating visualization...")
        
        # PCA reduction to 2D
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(self.X_train)
        
        # We need to retrain a simple model on 2D data for visualization purposes
        # purely to show decision boundaries in 2D space
        svc_2d = SVC(kernel=self.model.kernel, C=self.model.C, gamma=self.model.gamma if self.model.gamma != 'scale' else 'auto')
        svc_2d.fit(X_train_pca, self.y_train)

        # Create a mesh to plot in
        x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
        y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
        h = (x_max / x_min)/100 if x_min != 0 else 0.02 # step size
        h = 0.02
        
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))

        Z = svc_2d.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 6))
        plt.contourf(xx, yy, Z, alpha=0.8)
        
        # Plot also the training points
        scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=self.y_train, edgecolors='k', cmap=plt.cm.Paired)
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'SVM Decision Boundary (PCA 2D) - {self.dataset_name}\nKernel: {self.model.kernel}')
        plt.legend(*scatter.legend_elements(), title="Classes")
        
        # Mark support vectors (approximate for 2D view)
        sv_indices = svc_2d.support_
        plt.scatter(X_train_pca[sv_indices, 0], X_train_pca[sv_indices, 1], s=100,
                    linewidth=1, facecolors='none', edgecolors='k', label='Support Vectors')

        output_file = f"svm_visualization_{self.dataset_name}.png"
        plt.savefig(output_file)
        logger.info(f"Visualization saved to {output_file}")
        plt.close()

    def run(self):
        """
        Execute the full pipeline.
        """
        try:
            self.load_data()
            self.preprocess()
            self.train()
            self.evaluate()
            self.visualize()
        except Exception as e:
            logger.error(f"An error occurred: {e}", exc_info=True)

def main():
    parser = argparse.ArgumentParser(description="Run SVM Classifier Task")
    parser.add_argument('--dataset', type=str, default='iris', choices=['iris', 'digits'],
                        help="Dataset to use: 'iris' or 'digits'")
    args = parser.parse_args()

    task = SVMTask(dataset_name=args.dataset)
    task.run()

if __name__ == "__main__":
    main()

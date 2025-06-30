# =============================
# File: visualization.py
# Description: Visualization utilities for predictions and evaluation
# =============================

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def plot_confusion(y_true, y_pred, labels=("Normal", "Pneumonia")):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_sample_images(images, labels=None, predictions=None, n=9):
    plt.figure(figsize=(10, 10))
    for i in range(n):
        plt.subplot(3, 3, i+1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        title = ""
        if labels is not None:
            title += f"Label: {labels[i]}\n"
        if predictions is not None:
            title += f"Pred: {predictions[i]}"
        plt.title(title.strip(), fontsize=9)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def print_classification_metrics(y_true, y_pred):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["Normal", "Pneumonia"]))

def main():
    print("Visualization utilities ready. Import and use plot_confusion(), plot_sample_images(), etc.")

if __name__ == '__main__':
    main()

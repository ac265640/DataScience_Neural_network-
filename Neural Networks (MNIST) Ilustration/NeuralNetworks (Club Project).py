''' Implement a Neural Network on the Fashion MNIST dataset

-Both training and validation accuracy Graphs
-Both training and validation loss Graphs
-Test Accuracy and Loss Graphs
-Precision, Recall Graphs
-Confusion Matrix'''






import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import seaborn as sns

# importing all the required libraries,packages and models.




df = 'sample.csv'
info = pd.read_csv(df)
print(info.head())

#Loading csv file and displaying dataframe using pandas.






# Selection of row | column position.
label = info.iloc[:, 0].values
images = info.iloc[:, 1:].values

# Reshape and normalizing the images.
images = images.reshape(-1, 28, 28) / 255.0

# Converting integer class label to One-hot encode.
label = to_categorical(label)

# Split the data.(20% for test set and 80% for training)
train_images, test_images, train_label, test_label = train_test_split(images, label, test_size=0.2, random_state=42)

validation_images, train_images, validation_label, train_label = train_test_split(train_images, train_label, test_size=0.8, random_state=42)

# Building the Feed-forward Neural Network.
model = Sequential([Flatten(input_shape=(28, 28)),Dense(128, activation='relu'),Dense(64, activation='relu'),Dense(10, activation='softmax')])
model.summary()

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

# Training Model.
mod_train = model.fit(train_images, train_label, epochs=5, batch_size=32, validation_data=(validation_images, validation_label))

# Evaluating Model.
test_loss, test_accuracy = model.evaluate(test_images, test_label)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predicting and generating classification report.
test_pred = model.predict(test_images)
pred_classes = np.argmax(test_pred, axis=1)
true_classes = np.argmax(test_label, axis=1)
print(classification_report(true_classes, pred_classes))



#plotting required graphs 

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(mod_train.history['accuracy'], label='Training Accuracy')
plt.plot(mod_train.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy Graph')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(mod_train.history['loss'], label='Training Loss')
plt.plot(mod_train.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Graph')
plt.legend()
plt.show()



# 1. This code processes the Fashion MNIST dataset.
# 2. Builds and trains a feed-forward neural network model, evaluates its performance on the test set, and generates a classification report. 
# 3. Plotting the training and validation accuracy.







# Calculating precision, recall.
precision, recall, fscore, _ = precision_recall_fscore_support(true_classes, pred_classes)



# Plotiing Precision and Recall Graphs.
plt.figure(figsize=(10, 5))
plt.plot(range(len(precision)), precision, marker='o', label='Precision')
plt.plot(range(len(recall)), recall, marker='x', label='Recall')
plt.xticks(range(10))
plt.xlabel('Classes')
plt.ylabel('Tally')
plt.title('Precision and Recall')
plt.legend()
plt.show()







# Confusion Matrix.
confus_matrix = confusion_matrix(true_classes, pred_classes)

#Ploting Confusion Matrix.
plt.figure(figsize=(8, 8))
sns.heatmap(confus_matrix, annot=True, fmt='d', cmap='Blues',xticklabels=np.arange(10), yticklabels=np.arange(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()



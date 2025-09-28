import os
import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb
from wandb.integration.keras import WandbMetricsLogger
from nltk.tokenize import TreebankWordTokenizer

# ---------------- SETUP ---------------- #
lemmatizer = WordNetLemmatizer()
tokenizer = TreebankWordTokenizer()

# Make results folder if not exists
os.makedirs("results/plots", exist_ok=True)

# Init W&B
wandb.init(project="Artificial Neural Network", config={
    "learning_rate": 0.01,
    "epochs": 200,
    "batch_size": 5,
    "architecture": "Dense(128)-Dropout(0.5)-Dense(64)-Dropout(0.5)-Dense(Softmax)",
    "activation": "relu",
    "optimizer": "SGD with momentum 0.9"
})
config = wandb.config

# ---------------- LOAD DATA ---------------- #
intents = json.loads(open('intents.json').read())

words, classes, documents = [], [], []
ignoreLetters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        wordList = tokenizer.tokenize(pattern)
        words.extend(wordList)
        documents.append((wordList, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(set([lemmatizer.lemmatize(w.lower()) for w in words if w not in ignoreLetters]))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# ---------------- PREPARE TRAINING DATA ---------------- #
training = []
outputEmpty = [0] * len(classes)

for document in documents:
    bag = [1 if lemmatizer.lemmatize(w.lower()) in [lemmatizer.lemmatize(w.lower()) for w in document[0]] else 0 for w in words]
    outputRow = list(outputEmpty)
    outputRow[classes.index(document[1])] = 1
    training.append(bag + outputRow)

random.shuffle(training)
training = np.array(training)

trainX = training[:, :len(words)]
trainY = training[:, len(words):]

# ---------------- BUILD MODEL ---------------- #
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, input_shape=(len(trainX[0]),), activation='relu', name="dense_1"),
    tf.keras.layers.Dropout(0.5, name="dropout_1"),
    tf.keras.layers.Dense(64, activation='relu', name="dense_2"),
    tf.keras.layers.Dropout(0.5, name="dropout_2"),
    tf.keras.layers.Dense(len(trainY[0]), activation='softmax', name="output")
])

sgd = tf.keras.optimizers.SGD(learning_rate=config.learning_rate, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# ---------------- TRAINING ---------------- #
# Add EarlyStopping to reduce overfitting
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

history = model.fit(
    trainX, trainY,
    epochs=config.epochs,
    batch_size=config.batch_size,
    verbose=1,
    validation_split=0.2,
    callbacks=[WandbMetricsLogger(), early_stop]
)

# Save the trained model
model.save('chatbot_model.h5')
print("✅ Model Training Complete & Saved")

# ---------------- VISUALIZATION ---------------- #
# Loss Curve
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend(); plt.title("Loss Curve")
plt.savefig("results/plots/loss_curve.png")
plt.show()

# Accuracy Curve
plt.plot(history.history['accuracy'], label='train_acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend(); plt.title("Accuracy Curve")
plt.savefig("results/plots/accuracy_curve.png")
plt.show()

# Confusion Matrix
y_pred = model.predict(trainX)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(trainY, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, xticklabels=classes, yticklabels=classes, fmt='d', cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig("results/plots/confusion_matrix.png")
plt.show()

# Log images to W&B
wandb.log({
    "loss_curve": wandb.Image("results/plots/loss_curve.png"),
    "accuracy_curve": wandb.Image("results/plots/accuracy_curve.png"),
    "confusion_matrix": wandb.Image("results/plots/confusion_matrix.png")
})

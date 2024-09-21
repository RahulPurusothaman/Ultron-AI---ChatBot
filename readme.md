# 🤖 Chatbot Project

Welcome to the Chatbot Project! This repository contains a chatbot model that can understand user intents and respond accordingly.

## 🚀 Features
- Understands various user intents such as greetings, inquiries about programming, jokes, and more.
- Trained using a simple neural network model built with TensorFlow.
- Interactive command-line interface for chatting.

## 📁 Project Structure
- `intents.json`: Contains training data for the chatbot.
- `training.py`: Script to train the chatbot model using the intents dataset.
- `chatbot.py`: Script to interact with the trained model.
- `requirements.txt`: List of required Python libraries.


📖 How It Works
Data Processing: The training.py script reads intents.json, tokenizes the user patterns, and creates a bag-of-words representation of each input. It also prepares the output labels for each intent.

Model Training: A neural network is defined and trained using the processed data. The architecture includes:

An input layer with 128 neurons and ReLU activation.
A dropout layer for regularization.
An output layer with softmax activation to classify the intents.
Prediction: In chatbot.py, user inputs are processed using the same bag-of-words method. The trained model predicts the intent of the input, and the corresponding response is selected from the intents.json.

🔄 Updating Intents
You can easily extend the chatbot's capabilities by adding new intents or modifying existing ones in the intents.json file. Make sure to retrain the model after any changes.

📋 Usage
Train the Model: Run the following command to train the chatbot model:

bash
Copy code
python training.py
This will create chatbot_model.h5, words.pkl, and classes.pkl.

Start the Chatbot: Once the model is trained, run the chatbot using:

bash
Copy code
python chatbot.py
You can now start chatting with the bot! Type your message and hit enter.


## 📦 Installation

To get started, clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/your-chatbot-repo.git
cd your-chatbot-repo
pip install -r requirements.txt


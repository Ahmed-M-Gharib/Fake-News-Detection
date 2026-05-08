import pandas as pd
import re
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

print("1. Loading datasets...")
# This looks for the 'data' folder in the same directory as this script
df_fake = pd.read_csv(os.path.join("data", "Fake.csv"))
df_true = pd.read_csv(os.path.join("data", "True.csv"))

print("2. Merging and labeling data...")
df_fake['label'] = 0  # 0 for Fake News
df_true['label'] = 1  # 1 for True News
df_merged = pd.concat([df_fake, df_true], ignore_index=True)

# Shuffle the dataset so it's mixed well
df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
df = df_merged[['text', 'label']]

print("3. Cleaning text data... (this might take a minute)")
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\n', ' ', text)
    return text

df['clean_text'] = df['text'].apply(clean_text)

print("4. Tokenizing and padding sequences...")
max_vocab = 10000 
max_len = 300     

tokenizer = Tokenizer(num_words=max_vocab)
tokenizer.fit_on_texts(df['clean_text'])
X_sequences = tokenizer.texts_to_sequences(df['clean_text'])
X_padded = pad_sequences(X_sequences, maxlen=max_len, padding='post')

y = df['label'].values

print("5. Splitting into Train and Test...")
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(
    X_padded, y, test_size=0.20, random_state=42
)

print("\n--- Phase 1 Complete ---")
print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import matplotlib.pyplot as plt

print("\n--- Phase 2: Building the LSTM Architecture ---")

# We create a function to build the model so we can easily generate 
# two identical models to test our two different optimizers.
def build_lstm_model():
    model = Sequential()
    
    # 1. Embedding Layer: Turns our integer IDs into dense mathematical vectors
    model.add(Embedding(input_dim=max_vocab, output_dim=100, input_length=max_len))
    
    # 2. LSTM Layer: The memory core that understands the sequence of words
    model.add(LSTM(64))
    
    # 3. Hidden Dense Layer: Fulfilling the requirement to use different activation functions
    model.add(Dense(32, activation='relu'))
    
    # Dropout prevents the model from memorizing the data (overfitting)
    model.add(Dropout(0.5))
    
    # 4. Output Layer: Must be 'sigmoid' because we are doing binary classification (0 or 1)
    model.add(Dense(1, activation='sigmoid'))
    
    return model

print("\n--- Phase 3: Training & Comparing Optimizers ---")

# --- MODEL 1: ADAM OPTIMIZER ---
print("\nTraining Model 1 (Optimizer: Adam)...")
model_adam = build_lstm_model()
model_adam.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# We save the training history so we can plot it in Phase 4
history_adam = model_adam.fit(
    X_train, y_train, 
    epochs=3, # Keep this low for local CPU training
    batch_size=128, 
    validation_data=(X_test, y_test)
)

# --- MODEL 2: SGD OPTIMIZER ---
print("\nTraining Model 2 (Optimizer: SGD)...")
model_sgd = build_lstm_model()
model_sgd.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])

history_sgd = model_sgd.fit(
    X_train, y_train, 
    epochs=3, 
    batch_size=128, 
    validation_data=(X_test, y_test)
)

print("\n--- Phase 2 & 3 Complete ---")
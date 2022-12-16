# -*- coding: utf-8 -*-
"""
Exemplo com Dropout

@author: pri_k
"""

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import classification_report,  confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer, MinMaxScaler
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

def get_model():
  # Rede com duas cama escondidas
    model = Sequential([
    layers.Dense(12, activation="relu",input_shape=(19,)),
    layers.Dropout(0.1),
    layers.Dense(8, activation="relu"),
    layers.Dropout(0.1),
    layers.Dense(4, activation="softmax")  
    ])
    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    lb = LabelBinarizer()
    labelbin = lb.fit_transform(label)
    
    # Dividindo amostras em conjunto de treino e conjunto de teste
    seed = 42
    np.random.seed(seed)
    (trainX, testX, trainY, testY) = train_test_split(dataset, labelbin, test_size=0.2, random_state=seed)

    # Normalizando dados
    scaler = MinMaxScaler()
    trainX_scaled = scaler.fit_transform(trainX)
    testX_scaled = scaler.transform(testX)
    
    epocas = 50
    tam_batch = 50
    model = get_model()
    H = model.fit(trainX_scaled, trainY, batch_size=tam_batch, epochs=epocas, verbose=1)
      
    # Predição
    predictions = model.predict(testX_scaled, batch_size=tam_batch)
    result = model.evaluate(testX_scaled, testY, batch_size=tam_batch)
    print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1),
                            target_names=[str(label) for label in range(4)]))
    confusion_matrix(testY.argmax(axis=1), predictions.argmax(axis=1))
    
    # plotar loss e accuracy para os datasets 'train' e 'test'
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0,epocas), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0,epocas), H.history["accuracy"], label="train_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend()

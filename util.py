import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

def preprocess(datapath: str, max_words: int, max_len: int) -> tuple([list, list, list, list]):
    df: pd.DataFrame = pd.read_csv(datapath, sep=",")

    size = len(df)

    topics = df['topic']
    outlets = df['outlet']
    articles = df['text']
    leaning = df['type']
    bias = df['label']
    factual = df['factual']

    #articles = articles.str.split(' ')
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(articles.to_list())
    sequences = tokenizer.texts_to_sequences(articles.to_list())
    texts = pad_sequences(sequences, maxlen=max_len)

    topic_types = []
    outlet_types = []
    leanings = []
    biases = []
    facts = []
    for i in range(size):
        if topics[i] not in topic_types: topic_types.append(topics[i])
        if outlets[i] not in outlet_types: outlet_types.append(outlets[i])
        if leaning[i] not in leanings: leanings.append(leaning[i])
        if bias[i] not in biases: biases.append(bias[i])
        if factual[i] not in facts: facts.append(factual[i])

    for i in range(size):
        topics[i] = topic_types.index(topics[i])
        outlets[i] = outlet_types.index(outlets[i])
        leaning[i] = leanings.index(leaning[i])
        bias[i] = biases.index(bias[i])
        factual[i] = facts.index(factual[i])
    
    split = int(0.8 * size)
    x_train = [tf.convert_to_tensor(texts[:split]), tf.convert_to_tensor(topics[:split].to_numpy(np.int32)), tf.convert_to_tensor(outlets[:split].to_numpy(np.int32))]
    x_test = [tf.convert_to_tensor(texts[split:]), tf.convert_to_tensor(topics[split:].to_numpy(np.int32)), tf.convert_to_tensor(outlets[split:].to_numpy(np.int32))]
    y_train = [tf.convert_to_tensor(bias[:split].to_numpy(np.int32)), tf.convert_to_tensor(leaning[:split].to_numpy(np.int32)), tf.convert_to_tensor(factual[:split].to_numpy(np.int32))]
    y_test = [tf.convert_to_tensor(bias[split:].to_numpy(np.int32)), tf.convert_to_tensor(leaning[split:].to_numpy(np.int32)), tf.convert_to_tensor(factual[split:].to_numpy(np.int32))]

    return x_train, x_test, y_train, y_test

def preprocess_predict(datapath: str, max_words: int, max_len: int) -> list:
    df: pd.DataFrame = pd.read_csv(datapath, sep=",")

    size = len(df)

    topics = df['topic']
    outlets = df['outlet']
    articles = df['text']
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(articles.to_list())
    sequences = tokenizer.texts_to_sequences(articles.to_list())
    texts = pad_sequences(sequences, maxlen=max_len)

    topic_types = []
    outlet_types = []
    for i in range(size):
        if topics[i] not in topic_types: topic_types.append(topics[i])
        if outlets[i] not in outlet_types: outlet_types.append(outlets[i])

    for i in range(size):
        topics[i] = topic_types.index(topics[i])
        outlets[i] = outlet_types.index(outlets[i])
    
    x = [tf.convert_to_tensor(texts), tf.convert_to_tensor(topics.to_numpy(np.int32)), tf.convert_to_tensor(outlets.to_numpy(np.int32))]

    return x

def generate_graphs(datapath: str):

    training_acc = []
    validation_acc = []
    # Read data into lists
    with open(datapath, "r") as txt_file:
        line = txt_file.readline()
        while line:
            split = line.split("\t")
            training_acc.append(split[0].split(" "))
            validation_acc.append(split[1].split(" "))
            line = txt_file.readline()

    
    # Access the accuracies of the model at different epochs
    bias_acc = []
    val_bias_acc = []
    leaning_acc = []
    val_leaning_acc = []
    fact_acc = []
    val_fact_acc = []
    for i in range(len(training_acc)):
        bias_acc.append(round(float(training_acc[i][0]), 4))
        val_bias_acc.append(round(float(validation_acc[i][0]), 4))
        leaning_acc.append(round(float(training_acc[i][1]), 4))
        val_leaning_acc.append(round(float(validation_acc[i][1]), 4))
        fact_acc.append(round(float(training_acc[i][2]), 4))
        val_fact_acc.append(round(float(validation_acc[i][2]), 4))

    # Plot the training and validation accuracies
    plt.plot(val_bias_acc, label='Validation Bias')
    plt.plot(val_leaning_acc, label='Validation Leaning')
    plt.plot(val_fact_acc, label='Validation Factuality')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("standard.png")
    plt.clf()

    print("Done")
    
def generate_special_graphs():
    bias_acc1 = [0.6421, 0.6599, 0.6669, 0.6722, 0.6753, 0.6703]
    bias_acc2 = [0.6385, 0.6613, 0.6720, 0.6666, 0.6683, 0.6680]
    bias_acc3 = [0.6545, 0.6545, 0.6545, 0.6745, 0.6658, 0.6632]

    # Plot the training and validation accuracies
    plt.plot(bias_acc1, label='50 Embedding Nodes')
    plt.plot(bias_acc2, label='80 Embedding Nodes')
    plt.plot(bias_acc3, label='120 Embedding Nodes')
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig("Training_Curve_Bias_embed.png")
    plt.clf()

    print("Done")

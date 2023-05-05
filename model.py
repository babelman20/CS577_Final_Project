import argparse

from util import preprocess, preprocess_predict

from keras import Input
from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Dropout, concatenate

max_words = 10000   # Maximum number of words to use in the tokenizer
max_len = 200       # Maximum length of each sequence
embedding_dim = 50  # Dimension of the embedding layer
lstm_units = 128    # Number of units in the LSTM layer
dropout_rate = 0.2  # Dropout rate for regularization

def train():
    # Preprocess training data
    x_train, x_test, y_train, y_test = preprocess("./data/data.csv", max_words, max_len)

    article_input = Input(shape=(max_len), name='article_input')
    outlet_input = Input(shape=(1,), name='outlet_input')
    topic_input = Input(shape=(1,), name='topic_input')

    embedded = Embedding(max_words, embedding_dim, input_length=max_len)(article_input)
    lstm = Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate))(embedded)
    dropout = Dropout(dropout_rate)(lstm)
    dropout_output = Dense(32, activation="relu")(dropout)

    merged = concatenate([dropout_output, outlet_input, topic_input])

    bias_output = Dense(1, activation='sigmoid', name='bias')(merged)
    leaning_output = Dense(3, activation='softmax', name='leaning')(merged)
    fact_output = Dense(3, activation='softmax', name='fact')(merged)

    model = Model(inputs=[article_input, outlet_input, topic_input],
                outputs=[bias_output, leaning_output, fact_output])

    model.compile(optimizer='adam',
                loss={'bias': 'binary_crossentropy', 'leaning': 'sparse_categorical_crossentropy', 'fact': 'sparse_categorical_crossentropy'},
                metrics={'bias': 'accuracy', 'leaning': 'accuracy', 'fact': 'accuracy'})

    history = model.fit(x_train, y_train, epochs=7, batch_size=32, validation_data=(x_test, y_test))
    return model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--pretrain', default=None, type=str)
    parser.add_argument('--predict', default=None, type=str)

    args = parser.parse_args()

    if args.pretrain:
        model = load_model(args.pretrain)
    else:
        model = train()

    if args.predict:
        x = preprocess_predict(args.predict, max_words, max_len)
        y = model.predict(x)

        with open('output.txt', 'w') as txt_file:
            for i in range(len(y[0])):
                bias = "Biased" if y[0][i] < 0.5 else "Non-biased"
                leanings = y[1][i]
                if leanings[0] > leanings[1] and leanings[0] > leanings[2]:
                    leaning = "right"
                elif leanings[1] > leanings[0] and leanings[1] > leanings[2]:
                    leaning = "left"
                else:
                    leaning = "center"

                factuals = y[2][i]
                if factuals[0] > factuals[1] and factuals[0] > factuals[2]:
                    factual = "Entirely factual"
                elif factuals[1] > factuals[0] and factuals[1] > factuals[2]:
                    factual = "Expresses writer's opinion"
                else:
                    factual = "Somewhat factual but also opinionated"
                
                txt_file.write(f'{bias}, {leaning}, {factual}\n')
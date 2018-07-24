from keras.datasets import imdb

VOCABULARY_SIZE = 5000

# maximum length of sequence fed to RNN
# truncate longer and pad shorter
MAX_WORDS = 500

EMBEDDING_SIZE = 40
BATCH_SIZE = 64
NUM_EPOCHS = 20

def load_data():
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = VOCABULARY_SIZE)
    print('Loaded dataset with {} training samples, {} test samples'.format(len(X_train), len(X_test)))
    return (X_train, y_train, X_test, y_test)

def view_data(X_item, y_item):
    # Get the dictionary mapping review back to original words
    word2id = imdb.get_word_index()

    id2word = {i: word for word, i in word2id.items()}

    print('---review---')
    print([id2word.get(i, ' ') for i in X_item])
    print('---label---')
    print(y_item)

def min_max_review_length(train, test):
    max_len = len(max((X_train + X_test), key=len))
    min_len = len(min((X_train + X_test), key=len))
    return min_len, max_len

def organize_model():
    from keras import Sequential
    from keras.layers import Embedding, LSTM, Dense, Dropout

    model = Sequential()
    model.add(Embedding(VOCABULARY_SIZE, EMBEDDING_SIZE, input_length=MAX_WORDS))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dense(1, activation='sigmoid'))

    print(model.summary())
    return model

def train_and_score(model, X, y, X_test, y_test):
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    X_validation, y_validation = X[:BATCH_SIZE], y[:BATCH_SIZE]
    X_train, y_train = X[BATCH_SIZE:], y[BATCH_SIZE:]
    model.fit(X_train, y_train, validation_data=[X_validation, y_validation],
              batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
    scores = model.evaluate(X_test, y_test, verbose=0)
    print('Test accuracy:', scores[1])


def pad_seq(items):
    from keras.preprocessing import sequence
    return sequence.pad_sequences(items, maxlen=MAX_WORDS)

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()
    #view_data(X_train[6], y_train[6])
    min_len, max_len = min_max_review_length(X_train, X_test)
    print('Min length: {}, Max length: {}'.format(min_len, max_len))
    X_train_pad, X_test_pad = pad_seq(X_train), pad_seq(X_test)
    #view_data(X_train_pad[6], y_train[6])

    model = organize_model()
    train_and_score(model, X_train_pad, y_train, X_test_pad, y_test)

import os
import io

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import matplotlib.pyplot as plt

DIRECTORY_URL = 'data/compiled/'
FILE_NAMES = ['corpus_100AH']

parent_dir = os.path.dirname(DIRECTORY_URL)

print("Loading in data from the directory: " + parent_dir)

# Configuration
BUFFER_SIZE = 60000
BATCH_SIZE = 200
TAKE_SIZE = 5000  # Size of the test set
EMBEDDING_DIM = 256
EPOCHS = 10
OUT = "data/output/100AH/"
PLOT_LOSS = True


def labeler(example, index):
    return example, tf.cast(index, tf.int64)


labeled_data_sets = []

for i, file_name in enumerate(FILE_NAMES):
    lines_dataset = tf.data.TextLineDataset(os.path.join(parent_dir, file_name))
    labeled_dataset = lines_dataset.map(lambda ex: labeler(ex, i))
    labeled_data_sets.append(labeled_dataset)

all_labeled_data = labeled_data_sets[0]
for labeled_dataset in labeled_data_sets[1:]:
    all_labeled_data = all_labeled_data.concatenate(labeled_dataset)

all_labeled_data = all_labeled_data.shuffle(BUFFER_SIZE, reshuffle_each_iteration=False)

print("5 examples from the dataset")
for ex in all_labeled_data.take(5):
    print(ex)

tokenizer = tfds.features.text.Tokenizer()

vocabulary_set = set()
for text_tensor, _ in all_labeled_data:
    some_tokens = tokenizer.tokenize(text_tensor.numpy())
    vocabulary_set.update(some_tokens)

vocab_size = len(vocabulary_set)
print("Size of the vocabulary: " + str(vocab_size))

encoder = tfds.features.text.SubwordTextEncoder(vocabulary_set)


def encode(text_tensor, label):
    encoded_text = encoder.encode(text_tensor.numpy())
    return encoded_text, label


def encode_map_fn(text, label):
    encoded_text, label = tf.py_function(encode, inp=[text, label], Tout=(tf.int64, tf.int64))
    encoded_text.set_shape([None])
    label.set_shape([])

    return encoded_text, label


all_encoded_data = all_labeled_data.map(encode_map_fn)

train_data = all_encoded_data.shuffle(BUFFER_SIZE)
train_data = train_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))

test_data = all_encoded_data.take(TAKE_SIZE)
test_data = test_data.padded_batch(BATCH_SIZE, padded_shapes=([None], []))

vocab_size += 1


model = tf.keras.Sequential([
    layers.Embedding(encoder.vocab_size, EMBEDDING_DIM),
    layers.GlobalAveragePooling1D(),
    layers.Dense(16, activation='relu'),
    layers.Dense(1)
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=test_data, validation_steps=20)

e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)  # shape: (vocab_size, embedding_dim)

out_v = io.open(OUT + 'vecs.tsv', 'w', encoding='utf-8')
out_m = io.open(OUT + 'meta.tsv', 'w', encoding='utf-8')

for num, word in enumerate(encoder.subwords):
    vec = weights[num + 1]  # skip 0, it's padding.
    out_m.write(word + "\n")
    out_v.write('\t'.join([str(x) for x in vec]) + "\n")
out_v.close()

if PLOT_LOSS:
    history_dict = history.history

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 9))
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(12, 9))
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.ylim((0.5, 1))
    plt.show()

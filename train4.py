import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(audio_binary)
  return tf.squeeze(audio, axis=-1)

def get_label(file_path):
  parts = tf.strings.split(file_path, os.path.sep)
  return parts[-2]

def get_waveform_and_label(file_path):
  label = get_label(file_path)
  audio_binary = tf.io.read_file(file_path)
  waveform = decode_audio(audio_binary)
  return waveform, label

def get_spectrogram(waveform):
  # Padding for files with less than 16000 samples
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)

  # Concatenate audio with padding so that all audio clips will be of the 
  # same length
  waveform = tf.cast(waveform, tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)

  spectrogram = tf.abs(spectrogram)

  return spectrogram

def plot_spectrogram(spectrogram, ax):
  # Convert to frequencies to log scale and transpose so that the time is
  # represented in the x-axis (columns).
  log_spec = np.log(spectrogram.T)
  height = log_spec.shape[0]
  width = log_spec.shape[1]
  X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
  Y = range(height)
  ax.pcolormesh(X, Y, log_spec)
    
def get_spectrogram_and_label_id(audio, label):
  spectrogram = get_spectrogram(audio)
  spectrogram = tf.expand_dims(spectrogram, -1)
  label_id = tf.argmax(label == commands)
  return spectrogram, label_id

def preprocess_dataset(files):
  files_ds = tf.data.Dataset.from_tensor_slices(files)
  output_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)
  output_ds = output_ds.map(
      get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
  return output_ds

def linearSearch(array,search_val):
    for i in range (len(array)):
        if array[i] == search_val:
            return i
    return -1

def highestVal(array):
    index = 0
    high = array[0]
    for i in range (len(array)):
        if array[i] > high:
            high = array[i]
            index = i
    return index
    

def mainConfirm(word,qnum):
    feedback = []
    #Using a sample file to see how well the model can predict the word

    #Output the result
    #This is the *important* bit where you feed your own data into the model and see how
    #well the model recognises your data

    ##sample_file = '/Users/home/Documents/mini_speech_commands/no/01bb6a2a_nohash_0.wav'
    #directory of the file fed into the model to recognise
    sample_file = def_data_dir + '/' + str(qnum) +'_trimmed.wav'

    sample_ds = preprocess_dataset([str(sample_file)])

    #Displays the prediction for what the word could be according to the model, in the form of a bar chart
    for spectrogram, label in sample_ds.batch(1):
      prediction = model(spectrogram)
      plt.bar(commands, tf.nn.softmax(prediction[0]))
      plt.title(f'Predictions for "{commands[label[0]]}"')
      plt.show()
    i = linearSearch(commands,word)
    if i==-1:
        print("word not found in database")
        return -1
    else:
        pred_list = np.asarray(tf.nn.softmax(prediction[0]))
        real_accuracy = (pred_list[i])
        feedback.append(real_accuracy)
        j = highestVal(pred_list)
        feedback.append(commands[j])
        fake_accuracy = (pred_list[j])
        feedback.append(fake_accuracy)
        return feedback


seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#get current directory
def_data_dir = os.getcwd()

data_dir = pathlib.Path(def_data_dir+"/mini_speech_commands")
if not data_dir.exists():
    print("Invalid Path")

#     tf.keras.utils.get_file(
#       'mini_speech_commands',
#       origin="/Users/home/Documents/mini_speech_commands",
#       extract=True,
#       cache_dir='.', cache_subdir='data')

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md' ]
commands = commands[commands !='.DS_Store']
print('Commands:', commands)

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
print(filenames)
num_samples = len(filenames)
print('Number of total examples:', num_samples)
print('Number of examples per label:',
      len(tf.io.gfile.listdir(str(data_dir/commands[0]))))
print('Example file tensor:', filenames[0])

train_files = filenames[:6400]
val_files = filenames[6400: 6400 + 600]
test_files = filenames[-800:]

print('Training set size', len(train_files))
print('Validation set size', len(val_files))
print('Test set size', len(test_files))

AUTOTUNE = tf.data.AUTOTUNE
files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(get_waveform_and_label, num_parallel_calls=AUTOTUNE)

# rows = 3
# cols = 3
# n = rows*cols
# fig, axes = plt.subplots(rows, cols, figsize=(10, 12))
# for i, (audio, label) in enumerate(waveform_ds.take(n)):
#   r = i // cols
#   c = i % cols
#   ax = axes[r][c]
#   ax.plot(audio.numpy())
#   ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
#   label = label.numpy().decode('utf-8')
#   ax.set_title(label)

# plt.show()


# for waveform, label in waveform_ds.take(1):
#   label = label.numpy().decode('utf-8')
#   spectrogram = get_spectrogram(waveform)

# print('Label:', label)
# print('Waveform shape:', waveform.shape)
# print('Spectrogram shape:', spectrogram.shape)
# print('Audio playback')
# display.display(display.Audio(waveform, rate=16000))

# fig, axes = plt.subplots(2, figsize=(12, 8))
# timescale = np.arange(waveform.shape[0])
# axes[0].plot(timescale, waveform.numpy())
# axes[0].set_title('Waveform')
# axes[0].set_xlim([0, 16000])
# plot_spectrogram(spectrogram.numpy(), axes[1])
# axes[1].set_title('Spectrogram')
# plt.show()

spectrogram_ds = waveform_ds.map(
    get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

# rows = 3
# cols = 3
# n = rows*cols
# fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
# for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
#   r = i // cols
#   c = i % cols
#   ax = axes[r][c]
#   plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
#   ax.set_title(commands[label_id.numpy()])
#   ax.axis('off')

# plt.show()

train_ds = spectrogram_ds
val_ds = preprocess_dataset(val_files)
test_ds = preprocess_dataset(test_files)

batch_size = 64
train_ds = train_ds.batch(batch_size)
val_ds = val_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(AUTOTUNE)
val_ds = val_ds.cache().prefetch(AUTOTUNE)


for spectrogram, _ in spectrogram_ds.take(1):
  input_shape = spectrogram.shape
print('Input shape:', input_shape)
num_labels = len(commands)

norm_layer = preprocessing.Normalization()
norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

#*Defining the model*
# Resizing layer has been introduced to downsample the input for faster training of the model [less data]
# In the normalization layer, each pixel in the image is normalised using mathematical methods which
#determine the mean and standard deviation
#Output the summary of model afterwards
model = models.Sequential([
    layers.Input(shape=input_shape),
    # Downsample the input.
    preprocessing.Resizing(32, 32),
    # Normalize.
    norm_layer,
    layers.Conv2D(32, 3, activation='relu'),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
history = model.fit(
    train_ds, 
    validation_data=val_ds,  
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

metrics = history.history
plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

test_audio = []
test_labels = []

for audio, label in test_ds:
  test_audio.append(audio.numpy())
  test_labels.append(label.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)
print(f'Test set accuracy: {test_acc:.0%}')

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred) 
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands, 
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()




word = "right"

res=mainConfirm(word,2)
print(res)




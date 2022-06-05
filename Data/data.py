import os, subprocess, time
import tensorflow as tf

# Resize and scale each image
def resize_scale(image_path):
  byte_img = tf.io.read_file(image_path)  # as encoder
  # Decode
  img = tf.io.decode_jpeg(byte_img)
  # Resizing the image to be 105x105
  img = tf.image.resize(img, (105,105))
  # Scaling the image to be range 0 and 1
  img /= 255.0
  return img

# Preprocessing image function
def preprocess(inp_img, val_img, label):
  return (resize_scale(inp_img), resize_scale(val_img), label)

def train_test_split():
  subprocess.call(["../Data/GetUnzipImg.sh"])
  time.sleep(5)

  # Get image directory
  BASE_DIR = os.path.abspath('../Data')

  # Define path
  ANC_DIR = os.path.join(BASE_DIR, 'anchor')
  POS_DIR = os.path.join(BASE_DIR, 'positive')
  NEG_DIR = os.path.join(BASE_DIR, 'negative')

  # Get image directories
  anchor = tf.data.Dataset.list_files(ANC_DIR+'/*.jpg').take(400)
  positive = tf.data.Dataset.list_files(POS_DIR+'/*.jpg').take(400)
  negative = tf.data.Dataset.list_files(NEG_DIR+'/*.jpg', seed=32).take(400)

  # Create labelled data
  # Pair (anchor) with (positive) becomes 1 (positive pairs)
  # Pair (anchor) with (negative) becomes 0 (negative pairs)
  positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
  negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
  data = positives.concatenate(negatives)

  # Load data into TF dataloader (data pipelining)
  data = data.map(preprocess).cache().shuffle(buffer_size=1024)

  # Training data (70% for training, 16 batches, and start preprocessing the next 8 set of images)
  n_train = round(len(data)*0.7)
  train_data = data.take(n_train).batch(16).prefetch(8)

  # Testing data
  n_test = len(data) - n_train
  test_data = data.skip(n_train).take(n_test).batch(16).prefetch(8)

  return train_data, test_data

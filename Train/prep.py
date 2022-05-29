import os
import tensorflow as tf

# EXTRACT prep

# Get image directories
list_data = os.listdir(os.path.join(os.getcwd(),'data'))
list_images = []
for txt in list_data:
  with open(txt,'rb') as f:
    list_images.append(['%s/%s'%(os.getcwd(),line.strip()) for line in f])
neg_images, anc_images, pos_images = list_images
# Set number of images
n_images = len(anc_images)

# Get image from directories
anchor = tf.data.Dataset.list_files(anc_images).take(n_images)
positive = tf.data.Dataset.list_files(pos_images).take(n_images)
negative = tf.data.Dataset.list_files(neg_images, seed=32).take(n_images)

# TRANSFORM prep

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

# Create labelled data
# Pair (anchor) with (positive) becomes 1 (positive pairs)
# Pair (anchor) with (negative) becomes 0 (negative pairs)

positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

# Preprocessing image function
def preprocess(inp_img, val_img, label):
  return (resize_scale(inp_img), resize_scale(val_img), label)

#LOAD prep

# Load data into TF dataloader (data pipelining)
data = data.map(preprocess).cache().shuffle(buffer_size=1024)

# Training data (70% for training, 16 batches, and start preprocessing the next 8 set of images)
n_train = round(len(data)*0.7)
train_data = data.take(n_train).batch(16).prefetch(8)

# Testing data
n_test = len(data) - n_train
test_data = data.skip(n_train).take(n_test).batch(16).prefetch(8)

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
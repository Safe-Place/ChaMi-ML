import os, subprocess, time
import tensorflow as tf

def install_and_import(package):
    import importlib
    try:
        importlib.import_module(package)
    except ImportError:
        import pip
        pip.main(['install', package])
    finally:
        globals()[package] = importlib.import_module(package)


#--------------
# For Model v1
#--------------

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

def get_listdir_images():
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
  
  return anchor, positive, negative

def train_test_split():
  # Get image directories
  anchor, positive, negative = get_listdir_images()

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

#--------------
# For model v2
#--------------

# Crop the face with mtcnn
def extract_face(filename, size=(105, 105)):
  install_and_import('mtcnn')
  image = tf.io.read_file(filename)  # as encoder
  # Decode
  image = tf.io.decode_jpeg(image)
  # convert to array
  pixels = np.asarray(image)
  # create the detector, using default weights
  detector = mtcnn.mtcnn.MTCNN()
  # detect faces in the image
  results = detector.detect_faces(pixels)
  # extract the bounding box from the first face
  x1, y1, width, height = results[0]['box']
  # bug fix
  x1, y1 = abs(x1), abs(y1)
  x2, y2 = x1 + width, y1 + height
  # extract the face
  face = pixels[y1:y2, x1:x2]
  # Resizing the image to be 105x105
  face = tf.image.resize(face, size)
  # Scaling the image to be range 0 and 1
  face /= 255.0
  return face

# extract all faces on entire images
def extract_all_faces(anc_dir, pos_dir, neg_dir):
  install_and_import('mtcnn') # import library mtcnn
  
  anc_list = []
  pos_list = []
  neg_list = []
  
  for anc, pos, neg in zip(anc_dir, pos_dir, neg_dir):
    anc_list.append(extract_face(anc))
    pos_list.append(extract_face(pos))
    neg_list.append(extract_face(neg))

  anc_images = tf.data.Dataset.from_tensor_slices(anc_list)
  pos_images = tf.data.Dataset.from_tensor_slices(pos_list)
  neg_images = tf.data.Dataset.from_tensor_slices(neg_list)

  return anc_images, pos_images, neg_images

def train_test_split_v2():
  # Get image directories
  anchor, positive, negative = get_listdir_images()
  
  # Get all faces on entire images
  anc_images, pos_images, neg_images = extract_all_faces(anchor, positive, negative)
  
  # Create labelled data
  # Pair (anchor) with (positive) becomes 1 (positive pairs)
  # Pair (anchor) with (negative) becomes 0 (negative pairs)
  positives = tf.data.Dataset.zip((anc_images, pos_images, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
  negatives = tf.data.Dataset.zip((anc_images, neg_images, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
  data = positives.concatenate(negatives)
  data = data.shuffle(buffer_size=len(data))

  # Training data (70% for training, 16 batches, and start preprocessing the next 8 set of images)
  n_train = round(len(data)*0.7)
  train_data = data.take(n_train).batch(16).prefetch(8)
  
  # Testing data
  n_test = len(data) - n_train
  test_data = data.skip(n_train).take(n_test).batch(16).prefetch(8)
  
  return train_data, test_data

# Custom Training Models with tf functions

**Reference**

- https://www.tensorflow.org/guide/function
- https://www.tensorflow.org/guide/keras/writing_a_training_loop_from_scratch
- https://www.tensorflow.org/api_docs/python/tf/keras/metrics/BinaryAccuracy
- https://www.machinelearningplus.com/deep-learning/how-use-tf-function-to-speed-up-python-code-tensorflow/


**Project Structure**
-
```
Train
├── anchor.zip
├── face_train.ipynb
├── face_train_v2.ipynb
├── logs
│   └── gradient_tape
│       ├── 20220605-104904
│       │   ├── test
│       │   │   └── events.out.tfevents.1654426144.08eb34789bda.70.1.v2
│       │   └── train
│       │       └── events.out.tfevents.1654426144.08eb34789bda.70.0.v2
│       └── 20220612-135511
│           ├── test
│           │   └── events.out.tfevents.1655042111.f20e4e3b25d3.71.5.v2
│           └── train
│               └── events.out.tfevents.1655042111.f20e4e3b25d3.71.4.v2
├── negative.zip
├── positive.zip
├── training_checkpoints
│   ├── checkpoint
│   ├── ckpt-1.data-00000-of-00001
│   ├── ckpt-1.index
│   ├── ckpt-2.data-00000-of-00001
│   ├── ckpt-2.index
│   ├── ckpt-3.data-00000-of-00001
│   └── ckpt-3.index
└── training_checkpoints_v2
    ├── checkpoint
    ├── ckpt-1.data-00000-of-00001
    ├── ckpt-1.index
    ├── ckpt-2.data-00000-of-00001
    ├── ckpt-2.index
    ├── ckpt-3.data-00000-of-00001
    └── ckpt-3.index

10 directories, 23 files
```

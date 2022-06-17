**Research Model Reference**
- Link Paper : https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf
- Others : https://www.youtube.com/playlist?list=PLdaSUfwhSMfSX-evq5arvP375pycIjCGM

**Model Evaluation Reference**
- https://www.tensorflow.org/guide/keras/train_and_evaluate
- https://www.tensorflow.org/tensorboard/get_started

**Saved Model**
- Model v1 : https://drive.google.com/drive/folders/18YeZR6_ifgNXmnLQ4ash7lOsKZTEKVZR?usp=sharing
- Model v2 : https://drive.google.com/drive/folders/1fYxxxXd-kpdLj3DxLTq0-QgIC4-Q5ImF?usp=sharing

**Model Architecture**
- Model v1 : ![model v1.png]( {https://raw.githubusercontent.com/Safe-Place/ChaMi-ML/main/Models/model%20v1/model%201.1%20v1.png?token=GHSAT0AAAAAABSFWREIKLS33U66HPXRVVJ6YVMQY4Q})
- Model Embedding v1 : ![model embedding v1.png]( {https://raw.githubusercontent.com/Safe-Place/ChaMi-ML/main/Models/model%20v1/model%201.2%20v1.png})
- Model v2 : ![model v2.png]( {https://drive.google.com/file/d/12WNpWIkc2M3RGRGJGDVGcIhVwXIR5g3V/view?usp=sharing})
- Model Embedding v2 : ![model embedding v2.png]( {https://drive.google.com/file/d/11IRU1Z6wysMz3h51sKhdJgGrWSOTmMWw/view?usp=sharing})

**Project Structure**
-
```
Models
├── 1
│   ├── assets
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── 2
│   ├── assets
│   ├── keras_metadata.pb
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
├── embedding.png
├── embedding_v2.png
├── model.py
├── modeltfjs-h5
│   ├── group1-shard10of38.bin
│   ├── group1-shard11of38.bin
│   ├── group1-shard12of38.bin
│   ├── group1-shard13of38.bin
│   ├── group1-shard14of38.bin
│   ├── group1-shard15of38.bin
│   ├── group1-shard16of38.bin
│   ├── group1-shard17of38.bin
│   ├── group1-shard18of38.bin
│   ├── group1-shard19of38.bin
│   ├── group1-shard1of38.bin
│   ├── group1-shard20of38.bin
│   ├── group1-shard21of38.bin
│   ├── group1-shard22of38.bin
│   ├── group1-shard23of38.bin
│   ├── group1-shard24of38.bin
│   ├── group1-shard25of38.bin
│   ├── group1-shard26of38.bin
│   ├── group1-shard27of38.bin
│   ├── group1-shard28of38.bin
│   ├── group1-shard29of38.bin
│   ├── group1-shard2of38.bin
│   ├── group1-shard30of38.bin
│   ├── group1-shard31of38.bin
│   ├── group1-shard32of38.bin
│   ├── group1-shard33of38.bin
│   ├── group1-shard34of38.bin
│   ├── group1-shard35of38.bin
│   ├── group1-shard36of38.bin
│   ├── group1-shard37of38.bin
│   ├── group1-shard38of38.bin
│   ├── group1-shard3of38.bin
│   ├── group1-shard4of38.bin
│   ├── group1-shard5of38.bin
│   ├── group1-shard6of38.bin
│   ├── group1-shard7of38.bin
│   ├── group1-shard8of38.bin
│   ├── group1-shard9of38.bin
│   └── model.json
├── modeltfjs-h5_v2
│   ├── group1-shard1of2.bin
│   ├── group1-shard2of2.bin
│   └── model.json
├── my_model.h5
├── mymodeltfjs
│   ├── group1-shard10of38.bin
│   ├── group1-shard11of38.bin
│   ├── group1-shard12of38.bin
│   ├── group1-shard13of38.bin
│   ├── group1-shard14of38.bin
│   ├── group1-shard15of38.bin
│   ├── group1-shard16of38.bin
│   ├── group1-shard17of38.bin
│   ├── group1-shard18of38.bin
│   ├── group1-shard19of38.bin
│   ├── group1-shard1of38.bin
│   ├── group1-shard20of38.bin
│   ├── group1-shard21of38.bin
│   ├── group1-shard22of38.bin
│   ├── group1-shard23of38.bin
│   ├── group1-shard24of38.bin
│   ├── group1-shard25of38.bin
│   ├── group1-shard26of38.bin
│   ├── group1-shard27of38.bin
│   ├── group1-shard28of38.bin
│   ├── group1-shard29of38.bin
│   ├── group1-shard2of38.bin
│   ├── group1-shard30of38.bin
│   ├── group1-shard31of38.bin
│   ├── group1-shard32of38.bin
│   ├── group1-shard33of38.bin
│   ├── group1-shard34of38.bin
│   ├── group1-shard35of38.bin
│   ├── group1-shard36of38.bin
│   ├── group1-shard37of38.bin
│   ├── group1-shard38of38.bin
│   ├── group1-shard3of38.bin
│   ├── group1-shard4of38.bin
│   ├── group1-shard5of38.bin
│   ├── group1-shard6of38.bin
│   ├── group1-shard7of38.bin
│   ├── group1-shard8of38.bin
│   ├── group1-shard9of38.bin
│   └── model.json
├── mymodeltfjs_v2
│   ├── group1-shard1of2.bin
│   ├── group1-shard2of2.bin
│   └── model.json
├── my_model.tflite
├── my_model_v2.h5
├── my_model_v2.tflite
├── my_opsmodel.tflite
├── my_opsmodel_v2.tflite
├── my_quantmodel.tflite
├── my_quantmodel_v2.tflite
├── __pycache__
│   └── model.cpython-37.pyc
├── saved_model
│   ├── my_model
│   │   ├── assets
│   │   ├── keras_metadata.pb
│   │   ├── saved_model.pb
│   │   └── variables
│   │       ├── variables.data-00000-of-00001
│   │       └── variables.index
│   └── my_model_v2
│       ├── assets
│       ├── keras_metadata.pb
│       ├── saved_model.pb
│       └── variables
│           ├── variables.data-00000-of-00001
│           └── variables.index
├── SiameseNeuralNetwork.png
└── SiameseNeuralNetwork_v2.png

18 directories, 114 files
```

# Wakeup Word Detection & Speaker Recognition
A tensorflow2.x implementatino of Wake-up Word Detection and Sepaaker Recognition

### Configuration
1. configuration.py

### Data setting
1. change_name.py
2. data_augmentation.py

### Audio to Image(MFCC)
1. data_preprocess.py  (generate .npz file for train)

### Train and evaluate
1. detect_train.py

### Convert tensorflow to tflite
path : ./tflite_converter
1. org_tflite.py
2. quantization_tflite.py
3. pruning_tflite.py
4. tflite_evaluate.py (all tflite model evaluate)
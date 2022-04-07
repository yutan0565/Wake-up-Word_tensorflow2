# Wakeup Word Detection & Speaker Recognition
A tensorflow2.x implementatino of Wake-up Word Detection and Sepeaker Recognition
- Detect Wake-up Word 
- Recognize Speaker 
- tflite convert (Just type, quantization, pruning)

## Environment Setting
1. Download requeirments.txt file 
2. Edit configuration.py

## dataset
1. Collect your data in custum_dataset folder for each users

## Data setting
1. change_name.py 
2. data_augmentation.py (reverse, stretch, pitch, shift, noise)

## Audio to Image(Mel-log Spectrogram)
Make Mel-log Spectrogram & data preprocess (Regularization, clip etc)
1. *_data_preprocess.py  (generate .npz file for train, valid, and test)

## Train and evaluate
Best model will be saved in ./best_model
1. *_train.py
2. test_all.py 

## Convert tensorflow to tflite
1. orig_tflite.py
2. quantization_tflite.py
3. pruning_tflite.py
4. tflite_evaluate.py ( evaluate all tflite model )

## Real time detectino demo
Check demo sample in ./demo
1. python realtime_wuw.opy


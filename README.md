# CNN_LSTM_CTC_Tensorflow

CNN+LSTM+CTC based OCR(Optical Character Recognition) implemented using tensorflow. 

**Note:** there is **No** restriction on the number of characters in the image (variable length).

I trained a model with 50k images using this code and got 94% accuracy on test dataset (20k images).

## Architecture

The images are first processed by a CNN to extract features, then these extracted features are fed into a LSTM for character recognition.

The architecture of CNN is just `Convolution + Batch Normalization + Leaky Relu + Max Pooling` for simplicity, and the LSTM is a 2 layers stacked LSTM, you can also try out Bidirectional LSTM.

You can play with the network architecture (add dropout to CNN, stacked layers of LSTM etc.) and see what will happen. Have a look at [CNN part](cnn_lstm_otc_ocr.py#L45) and [LSTM part](cnn_lstm_otc_ocr.py#L60).


## Prerequisite

1. Python 3.6.4

2. TensorFlow 1.2

3. Opencv3 (Not a must, used to read images)



## How to run

There are many other parameters with which you can play, have a look at [utils.py](utils.py#L11).


``` shell
# cd to the your workspace.
# The code will evaluate the accuracy every validation_steps specified in parameters.

ls -R
  .:
  imgs  utils.py  helper.py  main.py  cnn_lstm_otc_ocr.py

  ./imgs:
  train  infer  val
  
  ./imgs/train:
  1_label.png  2_label.png  ...  40000_label.png
  
  ./imgs/val:
  1_label.png  2_label.png  ...  10000_label.png

  ./imgs/infer:
  1.png  2.png  ...  20000.png
   
  
# Train the model.
CUDA_VISIBLE_DEVICES=0 python ./main.py --train_dir=./imgs/train/ \
  --val_dir=./imgs/val/ \
  --out_channels=64 \
  --num_hidden=128 \
  --batch_size=128 \
  --log_dir=./log/train \
  --num_gpus=1 \
  --mode=train

# Inference
CUDA_VISIBLE_DEVICES=0 python ./main.py --infer_dir=./imgs/infer/ \
  --checkpoint_dir=./checkpoint/ \
  --num_gpus=1 \
  --mode=infer

```


## Run with your own data.

While preparing your data, make sure that all images are named in format: `id_label.jpg`, e.g: `004_12-01-18.jpg`.

``` shell
# make sure the data path is correct, have a look at helper.py.

python helper.py
```

## Introduction

Real-time person and face detection + age and gender estimation. We will use YOLO (YOU ONLY LOOK ONCE) algorithm for person detection, MTCNN (Multi-task Cascaded Convolutional Networks) (pretrained) algorithm for face detection and WideRest (pretrained) for the gender/age estimation .
For person detection, we will train the network within [darkflow](https://github.com/thtrieu/darkflow) framework.



## Requirements

Python3.8, tensorflow 2.4.0, numpy 1.19.2, opencv 4.4.0


## Training YOLO (Note: The trained model is already provided but if you want to retrain follow this steps ... if not skip)
To train YOLO to detect person, we will use the VOC2007 dataset since it contains the "person" label and it is a quite small dataset thus easy to train provided that we start from a pre-trained YOLO model. The steps to train YOLO model are detailed in this [repository](https://github.com/thtrieu/darkflow) but we will do a resume

1. Download and uzip the code from [here](https://github.com/thtrieu/darkflow)

2. Run this line
   ```
   python3 setup.py build_ext --inplace
   ```
3. make a copy of `tiny-yolo-voc.cfg` in the `cfg` directory and rename it `tiny-yolo-voc-1c.cfg` where `1c` stands for one class detection. In `tiny-yolo-voc-1c.cfg`, change `classes=1` and `filters=30` in the last convolution section. Finally, erase everything in `labels.txt` in the root directory and write `person`.

Notes: we used tiny-yolo configuration along with VOC2007 to keep things manageable CPU/TIME

4. As we said, we will start from a pre-trained network so, we download weigts from [here](https://pjreddie.com/darknet/yolov2/) and place it in `bin` directory (you have to create one if it does not exist). We must rename the file `tiny-yolo-voc.weights` since darkflow will pick the same-name config file.

5. Run this line to download the dataset and start training
```
curl -O https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
tar xf VOCtest_06-Nov-2007.tar
./flow --model cfg/tiny-yolo-voc-1c.cfg --load bin/tiny-yolo-voc.weights --train --dataset VOCdevkit/VOC2007/JPEGImages --annotation VOCdevkit/VOC2007/Annotations 
```

6. By default, darkflow will save checkpoints at each 250 step (you can change it) in `ckpt` directory (you have to create one if it does not exist). End training process when the loss/avg loss does maintain similar close values. We have good training when loss/avg-loss is close to 1. In this example, I stopped training at 4500 steps (took few hours to train). 
To use the trained model, you have to download the trained file from my drive ([here](https://drive.google.com/file/d/17ZILb6RwRSdfTpt9zoTmH9mvsx5u8jls/view?usp=sharing)) and place it in `ckpt` directory


## Face detection using MTCNN
For face detection, we used MTCNN since I have an experience in using it. It is quite efficient algorithm. It detects frontal and side views. Every related to MTCNN is found in `align` directory. I used the code of MTCNN provided [here](https://github.com/Linzaer/Face-Track-Detect-Extract)


## Age and gender estimation
For age and gender estimation, WideResnet with corresponding weights is used. You have to download the pretrained weights of WideResnet from my drive ([here](https://drive.google.com/file/d/10ZRgOYNxIR3qcXmFQo09A6z-oOOKmkJ3/view?usp=sharing)) and place them in `agender` directory. 


## PUT ALL TOGETHER
After training YOLO, and loading pretrained models of face detection and age/gender estimation we can process videos by running the script after filling the args
```
python3 main.py --video_name video_name --output_path output_path --detection_threshold detection_threshold --face_bool face_bool --margin margin --agender_weights agender_weights
```
or simply take default values that I already computed

```
python3 main.py
```







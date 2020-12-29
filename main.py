#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aive Test
@author: malbardan
"""
import os
import argparse
import cv2
import tensorflow as tf
import numpy as np
import align.detect_face as detect_face


from logging import getLogger
from darkflow.net.build import TFNet
from agender.wide_resnet import WideResNet


os.environ['KMP_DUPLICATE_LIB_OK']='True'
logger = getLogger()
directory = os.getcwd()

def drawDetectionBoundingBox(frame,result,faces,agender_model):
    """
    Draw bounding box around the person.
    
    Parameters
    ----------
    frame : numpy.array, video frame.
    result : dictionnary (output of YOLO). contains the box coords, label of detected object
    and level of confidence
    faces : list of lists, list of bounding boxes of detected faces.
    agender_model : .hdf5 file, weight file of WideResNet to estimate age and gender

    Returns
    -------
    frame : frame with the bouding box around the body and the face

    """
    args = parse_args()
    margin = args.margin
    # Drawing box for MTCNN detection (face)
    for j,face in enumerate(faces):
        face = list(map(int, face)) # convert floats to int because cv2.rectangle allows only int type
        x1_face,y1_face,x2_face,y2_face = face[0],face[1],face[2],face[3]
        
        # adding margin to the bounding box is crucial to correctly estimating age and gender margin = 40
        cropped_face = frame[face[1]-margin:face[3]+margin, face[0]-margin:face[2]+margin] 
        gender,age = get_age_gender(agender_model,cropped_face)
        
        text = (gender+','+str(int(age)))
        cv2.rectangle(frame,(x1_face,y1_face),(x2_face,y2_face),(0,255,0),3)
        cv2.putText(frame,text,(face[0],face[1]),cv2.FONT_HERSHEY_PLAIN,3,(0,0,0),3)
        
        
    # Drawing box for YOLO detection (person) 
    for box in result:
        x1_person,y1_person,x2_person,y2_person = (box['topleft']['x'],box['topleft']['y'],box['bottomright']['x'],box['bottomright']['y'])
        confidence = box['confidence']
        if confidence >0.1: # for YOLO
            cv2.rectangle(frame,(x1_person,y1_person),(x2_person,y2_person),(255,0,0),5)

        return frame 


def get_faces(directory,
              nets,
              frame):
    """
    Face detection using MTCNN
    
    Parameters
    ----------
    directory : str, main path.
    nets :tuple of (pnet,rnetonet) produced in MTCNN creation
    frame : numpy.array, video frame.
    
    
    Returns
    -------
    faces : list of list of floats, bounding boxes around detected faces
    """
    pnet,rnet,onet = nets
    # parameters of the mtcnn 
    minsize = 75 # minimum size of face for mtcnn to detect    
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold for pnet,rnet and onet
    factor = 0.709 # scale factor
    scale_rate = 1
             
    
    frame = cv2.resize(frame, (0, 0), fx=scale_rate, fy=scale_rate)
    r_g_b_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces, points = detect_face.detect_face(r_g_b_frame, minsize, pnet, rnet, onet, threshold,factor)     
    
    return faces

    
def get_age_gender(agender_model,cropped_face):
    """
    get age and gender from face
    
    PARAMETERS:
    ----------
    agender_model: .hdf5 file, weight file of WideResNet to estimate age and gender
    cropped_face: numpy.array, to estimate age and gender
    
    Returns:
    --------
    gender: str, estimated gender
    age: int, estimated age
    """
    img_size = 64    
    cropped_face = cv2.resize(cropped_face,(img_size,img_size)) # WideResnet accepts only images of shape (64,64,3)
    cropped_face = np.expand_dims(cropped_face, axis=0) # add a dimension at axis=0
    result = agender_model.predict(cropped_face)
    
    prob_genders = result[0][0]
    if prob_genders[0]>0.6:
        gender = 'Female'
    elif prob_genders[1]>0.6:
        gender = 'Male'
    else:
        gender = 'Unidentifed'
    
    ages = np.arange(0, 101).reshape(101, 1)
    age = result[1].dot(ages).flatten()
    return gender,age


    
def main(sess):
    
    args = parse_args()
    video_name = args.video_name
    output_path = args.output_path
    detection_threshold = args.detection_threshold
    face_bool = args.face_bool
    agender_weights = args.agender_weights
    
    
    # darknet for people detection
    MODEL_PATH = os.path.join(directory,'cfg/tiny-yolo-voc-1c.cfg')
    options = {"model": MODEL_PATH,
               "load": -1, # the last checkpoint. You can specify another checkpoint index
               "threshold": detection_threshold}
    tfnet = TFNet(options)
    
    # models for face detection
    nets = detect_face.create_mtcnn(sess, os.path.join(directory, "align"))
    
    # model for age and gender estimation
    agender_model = WideResNet(64, depth=16, k=8)()
    agender_model.load_weights(agender_weights)    
    
    # start reading the video
    cap = cv2.VideoCapture(video_name)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print( "the number of frame is ", length )
    
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    video_out = cv2.VideoWriter(output_path,fourcc, 20.0,(int(cap.get(3)),int(cap.get(4))))
    
    k=1 # counter
    n=1 # use this to skip some frames. if n=1 there is no skip
    
    while True:
        
        ret,frame = cap.read()
        if not ret:
            logger.warning("ret false")
            break
        if frame is None:
            logger.warning("frame drop")
            break 
        
        
        try:
            # get bounding boxes
            if k>0 and k%n==0:
                
                if face_bool:
                    # face detection MTCNN
                    faces = get_faces(directory,nets,frame)
            
                # person detection YOLO
                result = tfnet.return_predict(frame)
                drawDetectionBoundingBox(frame,result,faces,agender_model)
        except:
            continue
    
        
        
        # Display the resulting frame
        cv2.imshow('frame',frame)
        video_out.write(frame)
        k+=1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
            
    # Release everything if job is finished
    cap.release()
    video_out.release()
    cv2.destroyAllWindows()


def parse_args():
    """Parse input arguments."""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_name', type=str,
                        help='Path to input video',
                        default=os.path.join(directory,'MISS DIOR – The new Eau de Parfum.avi'))
    parser.add_argument('--output_path', type=str,
                        help='Path to save video',
                        default=os.path.join(directory,'output.avi'))
    parser.add_argument('--detection_threshold', type=float,
                        help='detection threshold for YOLO',
                        default=0.07)
    parser.add_argument('--face_bool', type=bool,
                        help='Perform face detection,age/gender estimation or not',
                        default=True)
    parser.add_argument('--margin',type=int,
                        help='margin to take for face detection (crucial for age/gender estimation)',
                        default= 40)
    parser.add_argument('--agender_weights',
                        help='pretrained model for age and gender detection',
                        default= os.path.join(directory,'agender/weights.29-3.76_utk.hdf5'))
    args = parser.parse_args()
    return args




if __name__ == "__main__":
    with tf.Graph().as_default():
        with tf.compat.v1.Session() as sess:
            main(sess)






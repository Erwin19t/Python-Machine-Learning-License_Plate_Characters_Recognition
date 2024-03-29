import cv2
import os
import logging
import numpy as np
from Misc import Get_Characters

def video_mode(args, path, recognition_model):
    frame_counter = 0
    
    file_path = os.path.join(path[4], f"{args.file}.mp4")
    cap = cv2.VideoCapture(file_path)
    
    while cap.isOpened():
        frame_counter += 1
        ret, frame = cap.read()

        if not ret:
            logging.ERROR("Can't receive frame (stream end?). Exiting ...")
            break
        
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', img)
        
        if cv2.waitKey(1) == ord('q'):
            break
        
        new_img_size = (1000, 750)
        img_resized = cv2.resize(img, new_img_size)
       
        if frame_counter %82 == 0:
            String_Plate = recognition_model.plate_recognition(img_resized)
        
            if np.all(String_Plate == None):
                logging.info("Plate not found. Exiting...")
                return        

            #Plate coordenates are extracted
            x_1 = int(String_Plate[0,0])
            y_1 = int(String_Plate[0,1])
            x_2 = int(String_Plate[0,2])
            y_2 = int(String_Plate[0,3])
        
            cropped_image = img_resized[y_1:y_2, x_1:x_2]
            cv2.imshow('Plate', cropped_image)
    
            results = recognition_model.characters_recognition(cropped_image)
            plate = Get_Characters.get_characters(results)
            print("La placa reconocida es:", plate)
            
    cap.release()
    cv2.destroyAllWindows()
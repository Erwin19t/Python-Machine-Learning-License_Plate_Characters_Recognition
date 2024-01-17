import torch
import cv2
import argparse
import os
import logging
import numpy as np
import matplotlib.pyplot as ptl

class RecognitionModel:
    def __init__(self, path):
        logging.info("Loading models...")
        self.plate_model     = torch.hub.load(path[0], 'custom', source='local', path=path[1], force_reload=True, _verbose=False)
        self.character_model = torch.hub.load(path[0], 'custom', source='local', path=path[2], force_reload=True, _verbose=False)
        logging.info("Models loaded successfully.")

    def plate_recognition(self, img):
        #Get Detections
        detection = self.plate_model(img)
        #Convert detection into data frame
        data_frame = detection.pandas().xyxy[0]
        #Extract indexes of data frame
        indexes = data_frame.index
        #Initialize the Results array with correct shape
        Results = np.zeros((len(indexes), 6))
        for i in indexes:
            #Get X_min, Y_min, X_max & Y_max
            x_min = int(data_frame['xmin'][i])
            y_min = int(data_frame['ymin'][i])
            x_max = int(data_frame['xmax'][i])
            y_max = int(data_frame['ymax'][i])
            #Get label & confidence
            name_class = int(data_frame['class'][i])
            confidence = float(data_frame['confidence'][i]) * 100
            #Store information
            Results[i, :] = (x_min, y_min, x_max, y_max, confidence, name_class)
        
        #Information is sorted according xmin values
        sorted_indexes = np.argsort(Results[:, 0])
        Results_sorted = Results[sorted_indexes]
        return Results_sorted

    def characters_recognition(self, img):
        detection = self.character_model(img)
        #Convert detection into data frame
        data_frame = detection.pandas().xyxy[0]
        #Extract indexes of data frame
        indexes = data_frame.index
        #Initialize the Results array with correct shape
        Results = np.zeros((len(indexes), 6))
        for i in indexes:
            #Get X_min, Y_min, X_max & Y_max
            x_min = int(data_frame['xmin'][i])
            y_min = int(data_frame['ymin'][i])
            x_max = int(data_frame['xmax'][i])
            y_max = int(data_frame['ymax'][i])
            #Get label & confidence
            name_class = int(data_frame['class'][i])
            confidence = float(data_frame['confidence'][i]) * 100
            #Store information
            Results[i, :] = (x_min, y_min, x_max, y_max, confidence, name_class)
        
        #information is sorted according to xmin values
        sorted_indexes = np.argsort(Results[:, 0])
        Results_sorted = Results[sorted_indexes]
        
        #rows with low confidence are removed
        Results_Filtered = self.threshold(Results_sorted)
        return Results_Filtered
    
    def threshold(self, Array):
        filtered_array = Array[Array[:, 4] >= 45.0]
        return filtered_array


    #To do: Develop this method
    def model_performance(self, model):
        return None

def parse_arguments():
    #Initializes an instance of the 'ArgumentParser' class from the 'argparse' module
    parser = argparse.ArgumentParser()
    
    #Add hyperparameters as arguments
    parser.add_argument("-m", "--mode", type=int, choices=[0, 1], help="0 for image & 1 for video")
    parser.add_argument("-f", "--file", type=str, help="Type the filename")
    
     #Parse command line arguments
    return parser.parse_args()

def path_list():
    #List of paths required in the program
    return (
        "./YOLOV5/",                                            # 0: Model Path
        "./YOLOV5/runs/train/Plate/weights/Weights.pt",         # 1: License Weights Path
        "./YOLOV5/runs/train/Characters/weights/Weights.pt",    # 2: Characters Weights Path
        "./Dataset/Test/Images",                                # 3: Image mode operation Path
        "./Dataset/Test/Videos",                                # 4: Video mode operation Path
        "./Dataset/Results/Images",                             # 5: Images results Path
        "./Dataset/Results/Videos",                             # 6: Videos results Path
    )

def image_mode(args, path, recognition_model):
    file_path = os.path.join(path[3], f"{args.file}.jpeg")
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    new_img_size = (1000, 750)
    img_resized = cv2.resize(img, new_img_size)
    
    #Original image is passed through the first model
    String_Plate = recognition_model.plate_recognition(img_resized)
    
    #Extract coordenates (xmin, ymin, xmax, ymax)
    x_1 = int(String_Plate[0,0])
    y_1 = int(String_Plate[0,1])
    x_2 = int(String_Plate[0,2])
    y_2 = int(String_Plate[0,3])
    
    #Image is cropped according to the coordenates & it is showed
    cropped_image = img_resized[y_1:y_2, x_1:x_2]
    cv2.imshow("Cropped Image", cropped_image)
    
    #Cropped image is passed through the second model
    characters_Plate = recognition_model.characters_recognition(cropped_image)
    
    #Extract characte's column of "characters_plate" matrix
    plate = get_characters(characters_Plate)
    print("La placa reconocida es:", plate)


def video_mode(args, path, recognition_model):
    file_path = os.path.join(path[4], f"{args.file}.mp4")
    cap = cv2.VideoCapture(file_path)
    while cap.isOpened():
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame', img)
        if cv2.waitKey(1) == ord('q'):
            break
        new_img_size = (1000, 750)
        img_resized = cv2.resize(img, new_img_size)
        results = recognition_model.characters_recognition(img_resized)
    # Implement video processing using OpenCV
    cap.release()
    cv2.destroyAllWindows()
    
def get_characters(results):  
    alphabet = ['A','B','C','D','E','F','G','H','J','K','L','M','N','P','R','S','T','U','V','W','X','Y','Z','-','0','1','2','3','4','5','6','7','8','9']
    plate = ''
    for i in range(len(results)):    
        plate = plate + alphabet[int(results[i][5])]
    return plate

def main(args, path):
    #Initializes an instance of the 'RecognitionModel' class
    recognition_model = RecognitionModel(path)
    if args.mode:
        logging.info("Video mode has been chosen...")
        video_mode(args, path, recognition_model)
    else:
        logging.info("Image mode has been chosen...")
        image_mode(args, path, recognition_model)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    path = path_list()
    main(args, path)

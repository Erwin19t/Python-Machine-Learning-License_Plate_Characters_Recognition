import logging
import torch
import numpy as np

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
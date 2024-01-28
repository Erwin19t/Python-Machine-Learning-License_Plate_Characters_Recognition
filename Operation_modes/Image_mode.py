import cv2
import os
from Misc.Get_Characters import get_characters

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
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()
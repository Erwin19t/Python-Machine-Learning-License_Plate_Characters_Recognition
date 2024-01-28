import argparse
import logging
from Recognition_model.alpha_model import RecognitionModel
from Operation_modes import image_mode, video_mode


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

def main(args, path):
    #Initializes an instance of the 'RecognitionModel' class
    recognition_model = RecognitionModel(path)
    if args.mode:
        logging.info("Video mode has been chosen...")
        video_mode(args, path, recognition_model)
    else:
        logging.info("Image mode has been chosen...")
        image_mode(args, path, recognition_model)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = parse_arguments()
    path = path_list()
    main(args, path)

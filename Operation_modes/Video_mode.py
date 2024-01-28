import cv2
import os
from Misc import Get_Characters

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
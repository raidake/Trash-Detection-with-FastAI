from fastai.learner import load_learner
from fastai.vision import *
import cv2
from pathlib import Path

cur_dir = Path().absolute()
learn = load_learner(cur_dir,"my_export.pkl")

window_name = "Detecting Trash"
video = cv2.VideoCapture(0)

while video.isOpened():
    ret, frame = video.read()

    if not ret:
        break
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()


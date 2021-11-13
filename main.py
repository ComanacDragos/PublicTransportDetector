import os
from settings import *

print(os.listdir(PATH_TO_TRAIN).__len__())

print(os.listdir(PATH_TO_TEST).__len__())

print(os.listdir(PATH_TO_VALIDATION).__len__())
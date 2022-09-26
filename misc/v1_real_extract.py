import cv2
import os
import pandas


csv_dir_name = "/data1/datasets/aadhaar_card/v1-real/csvs/"

img_dir = "/data1/datasets/aadhaar_card/v1-real/images/"

csv_files = os.listdir(csv_dir_name)

for eachf in csv_files:
    img_name = eachf.split('.')[0] + '.jpg'
    data = pandas.read_csv(os.path.join(csv_dir_name, eachf))
    txt = list(data['text'])
    bbox = list(data['bb'])
    # print(txt,bbox)
    for i, txt_val in enumerate(txt):
        print(txt_val, bbox[i])
    break




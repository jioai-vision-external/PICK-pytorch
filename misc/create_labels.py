import os
import pandas
import json
import csv
import shutil
import ast
import numpy as np
import cv2
import csv
## Input dataset
data_path = "/data1/datasets/aadhaar_card/v1-real/csvs/"



img_path = "/data1/datasets/aadhaar_card/v1-real/images/"


out_boxes_and_transcripts = "./datasets/v1-real-test-all-gt/boxes_transcripts/"
out_images = "./datasets/v1-real-test-all-gt/img/"
# out_entities  = "./out_base/key/"


all_img_list = os.listdir(img_path)
img_resize = (500,500)


def transform_coord(new_img, x1,y1,x3,y3, orig_img):
    scale = np.flipud(np.divide(new_img.shape, orig_img.shape))
    # print(scale)
    new_top_left_corner = map(int, np.multiply((x1,y1), scale[1:] ))
    new_bottom_right_corner = map(int, np.multiply((x3,y3), scale[1:]))
    return new_top_left_corner, new_bottom_right_corner


# data = pandas.read_csv(data_path)
print(data)
each_file = {}
# whole_file = {}
cnt = 0
selected_labels = ["name","vid","dob","gender","aadhaar_number"]
for ind, each_r in data.iterrows():
    
    
    print(each_r['text'], each_r['bb'], each_r['label'], each_r['file'])

    fname = each_r['file']
    image_name = os.path.join(img_path, fname+'.jpg')
    img = cv2.imread(image_name)
    orig_h, orig_w, _ = img.shape
    # new_img = cv2.resize(img, (img_resize[0],img_resize[1]))
    new_img = img
    if each_r['file'] not in each_file:
        box_data = {'x1':[], 'y1': [],'x2':[], 'y2': [],'x3':[], 'y3': [],'x4':[], 'y4': [],'text':[] , 'label': []}
        each_file[fname] = box_data
        print(each_file)
    each_r['bb'] =  ast.literal_eval(each_r['bb'])
    x1 = each_r['bb'][0]
    y1 = each_r['bb'][1]
    x2 = each_r['bb'][0]
    y2 = each_r['bb'][3]
    x3 = each_r['bb'][2]
    y3 = each_r['bb'][3]
    x4 = each_r['bb'][2]
    y4 = each_r['bb'][1]
    print(x1,y1,x2,y2,x3,y3,x4,y4)
    fine_label = each_r['label']
    if fine_label not in selected_labels:
        fine_label = 'other'

    # each_file[fname]['x1'].append(x1)
    # each_file[fname]['y1'].append(y1)
    # each_file[fname]['x2'].append(x2)
    # each_file[fname]['y2'].append(y2)
    # each_file[fname]['x3'].append(x3)
    # each_file[fname]['y3'].append(y3)
    # each_file[fname]['x4'].append(x4)
    # each_file[fname]['y4'].append(y4)
    # each_file[fname]['text'].append(each_r['text'])

    # (new_x1,new_y1), (new_x3,new_y3) = transform_coord(new_img, x1,y1,x3,y3, img)
    new_x1, new_y1, new_x3, new_y3 = x1, y1, x3, y3
    print(new_x1,new_y1, new_x3, new_y3)

    # if fine_label != 'other':
    each_file[fname]['x1'].append(new_x1)
    each_file[fname]['y1'].append(new_y1)
    each_file[fname]['x2'].append(new_x3)
    each_file[fname]['y2'].append(new_y1)
    each_file[fname]['x3'].append(new_x3)
    each_file[fname]['y3'].append(new_y3)
    each_file[fname]['x4'].append(new_x1)
    each_file[fname]['y4'].append(new_y3)
    each_file[fname]['text'].append(each_r['text'])
    each_file[fname]['label'].append(fine_label)
    cnt+=1
    

for each_f in list(each_file.keys()):
    print(each_f)
    full_csv_name = os.path.join(out_boxes_and_transcripts, each_f+'.tsv')
    image_name = os.path.join(img_path, each_f+'.jpg')
    print(image_name)
    df = pandas.DataFrame(each_file[each_f])
    df.index = np.arange(1, len(df)+1)
    df.to_csv(full_csv_name, index=True,header=False) # quotechar='',escapechar='\\',quoting=csv.QUOTE_NONE,)
    img = cv2.imread(image_name)
    # new_img = cv2.resize(img, (img_resize[0],img_resize[1]))
    new_img = img
    cv2.imwrite(os.path.join(out_images, each_f+'.jpg'), new_img)
# print(all_img_list)
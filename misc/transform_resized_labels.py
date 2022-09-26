import os
import pandas
import json
import csv
import shutil
import ast
import numpy as np
import cv2

## Input dataset
data_path = "/data1/datasets/aadhaar_card/v1/layout_3/"
box_path = data_path + "csvs_line_ppocr/"
img_path = data_path + "images/"
key_path = data_path + "csvs/"

out_boxes_and_transcripts = "./datasets/nofilter-all/tmp_l4/box/"
out_images = "./datasets/nofilter-all/tmp_l4/img/"
out_entities  = "./datasets/nofilter-all/tmp_l4/key/"

csv_path = "./datasets/nofilter-all/tmp_l4/train_samples_list.csv"


all_img_list = os.listdir(img_path)
img_resize = (640,480)

ignore_labels = ['background', 'background_lang', 'address', 'gen_date', 'random', 'enroll', 'down_date']

def transform_coord(new_img, x1,y1,x3,y3, orig_img):
    scale = np.flipud(np.divide(new_img.shape, orig_img.shape))
    # print(scale)
    new_top_left_corner = map(int, np.multiply((x1,y1), scale[1:] ))
    new_bottom_right_corner = map(int, np.multiply((x3,y3), scale[1:]))
    return new_top_left_corner, new_bottom_right_corner

# print(all_img_list)
train_samples_list = []
cnt = 0
for each_image in all_img_list:
    cnt+= 1
    print(each_image)
    csv_name = each_image.split('.')[0] + ".csv"
    full_csv_name = os.path.join(out_boxes_and_transcripts, csv_name.split('.')[0]+'.tsv')
    orig_csv = os.path.join(key_path, csv_name)
    data = pandas.read_csv(orig_csv)
    print(data)
    key_labels = {}
    box_data = {'x1':[], 'y1': [],'x2':[], 'y2': [],'x3':[], 'y3': [],'x4':[], 'y4': [],'text':[], 'label': []}
    img = cv2.imread(os.path.join(img_path, each_image))
    orig_h, orig_w, _ = img.shape
    new_img  = img
    # new_img = cv2.resize(img, (img_resize[0],img_resize[1]))
    for ind, each_r in data.iterrows():
        print(each_r['text'], each_r['bb'], each_r['readable_label'], each_r['line'])
        each_r['bb'] =  ast.literal_eval(each_r['bb'])
        x1 = each_r['bb'][0]
        y1 = each_r['bb'][1]
        x2 = each_r['bb'][2]
        y2 = each_r['bb'][1]
        x3 = each_r['bb'][2]
        y3 = each_r['bb'][3]
        x4 = each_r['bb'][0]
        y4 = each_r['bb'][3]
        print(x1,y1,x2,y2,x3,y3,x4,y4)
        fine_label = each_r['readable_label']
        if fine_label in ignore_labels:
            fine_label = 'other'
        if fine_label not in key_labels:
            key_labels[fine_label] = ""
        

        if key_labels[fine_label]:
            key_labels[fine_label] += " "+ str(each_r['text'])
        else:
            key_labels[fine_label] += str(each_r['text'])




        
        # (new_x1,new_y1), (new_x3,new_y3) = transform_coord(new_img, x1,y1,x3,y3, img)
        new_x1, new_y1, new_x3, new_y3 = x1, y1, x3, y3
        print(new_x1,new_y1, new_x3, new_y3)
    
        # if fine_label != "other":

        box_data['x1'].append(new_x1)
        box_data['y1'].append(new_y1)
        box_data['x2'].append(new_x3)
        box_data['y2'].append(new_y1)
        box_data['x3'].append(new_x3)
        box_data['y3'].append(new_y3)
        box_data['x4'].append(new_x1)
        box_data['y4'].append(new_y3)
        box_data['text'].append(each_r['text'])
        box_data['label'].append(fine_label)
            # cv2.rectangle(new_img, (new_x1, new_y1), (new_x3, new_y3), (255,0,255), 2)

        # row = str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+str(x3)+','+str(y3)+','+str(x4)+','+str(y4)+','+str(each_r['text'])+','+fine_label


    df = pandas.DataFrame(box_data)
    df.index = np.arange(1, len(df)+1)
    df.to_csv(full_csv_name, index=True,header=False, quotechar='',escapechar='\\',quoting=csv.QUOTE_NONE,)
    print(key_labels.keys())
    # print(key_labels)
    key_json = key_labels
    if 'other' in list(key_json.keys()):
        del key_json['other']
    print(key_json)
    with open(os.path.join(out_entities, each_image.split('.')[0]+ '.txt'), 'w') as fp:
        json.dump(key_json, fp)
    

    
    # img = cv2.resize(img, img_resize)

    cv2.imwrite(os.path.join(out_images, each_image), new_img)
    #shutil.copyfile(os.path.join(img_path, each_image), os.path.join(out_images, each_image))
    train_samples_list.append(['receipt',each_image.replace('.jpg','')])

    if cnt >= 2000:
        break
train_samples_list = pandas.DataFrame(train_samples_list)
train_samples_list.to_csv(csv_path)

    # break

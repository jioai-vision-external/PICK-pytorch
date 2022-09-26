import os
import pandas
import json
import csv
import shutil
import ast
import numpy as np

## Input dataset
data_path = "/Users/kanan.vyas/Kanan/VDU/layout_1/"
box_path = data_path + "csvs_line_ppocr/"
img_path = data_path + "images/"
key_path = data_path + "csvs/"

out_boxes_and_transcripts = "./out_base/box/"
out_images = "./out_base/img/"
out_entities  = "./out_base/key/"


all_img_list = os.listdir(img_path)

# print(all_img_list)
train_samples_list = []
for each_image in all_img_list:
    print(each_image)
    csv_name = each_image.split('.')[0] + ".csv"
    full_csv_name = os.path.join(out_boxes_and_transcripts, csv_name.split('.')[0]+'.tsv')
    orig_csv = os.path.join(key_path, csv_name)
    data = pandas.read_csv(orig_csv)
    print(data)
    key_labels = {}
    box_data = {'x1':[], 'y1': [],'x2':[], 'y2': [],'x3':[], 'y3': [],'x4':[], 'y4': [],'text':[], 'label': []}

    for ind, each_r in data.iterrows():
        print(each_r['text'], each_r['bb'], each_r['readable_label'], each_r['line'])
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
        if each_r['readable_label'] not in key_labels:
            key_labels[each_r['readable_label']] = ""
        if key_labels[each_r['readable_label']]:
            key_labels[each_r['readable_label']] += " "+ str(each_r['text'])
        else:
            key_labels[each_r['readable_label']] += str(each_r['text'])

        fine_label = each_r['readable_label']
        if fine_label == "background" or fine_label == "background_lang":
            fine_label = 'other'
        box_data['x1'].append(x1)
        box_data['y1'].append(y1)
        box_data['x2'].append(x2)
        box_data['y2'].append(y2)
        box_data['x3'].append(x3)
        box_data['y3'].append(y3)
        box_data['x4'].append(x4)
        box_data['y4'].append(y4)
        box_data['text'].append(each_r['text'])
        box_data['label'].append(fine_label)

        # row = str(x1)+','+str(y1)+','+str(x2)+','+str(y2)+','+str(x3)+','+str(y3)+','+str(x4)+','+str(y4)+','+str(each_r['text'])+','+fine_label


    df = pandas.DataFrame(box_data)
    df.index = np.arange(1, len(df)+1)
    df.to_csv(full_csv_name, index=True,header=False, quotechar='',escapechar='\\',quoting=csv.QUOTE_NONE,)
    print(key_labels.keys())
    # print(key_labels)
    key_json = key_labels
    if 'background' in list(key_json.keys()):
        del key_json['background']
    if 'background_lang' in list(key_json.keys()):
        del key_json['background_lang']
    print(key_json)
    with open(os.path.join(out_entities, each_image.split('.')[0]+ '.txt'), 'w') as fp:
        json.dump(key_json, fp)

    shutil.copyfile(os.path.join(img_path, each_image), os.path.join(out_images, each_image))
    train_samples_list.append(['receipt',each_image.replace('.jpg','')])
train_samples_list = pandas.DataFrame(train_samples_list)
train_samples_list.to_csv("train_samples_list.csv")

    # break

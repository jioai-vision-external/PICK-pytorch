import imp
import pandas
import os
import re
import csv
from sklearn.metrics import classification_report


gt_folder = "/data1/Kanan/PICK-pytorch/datasets/v1-real-test-all-gt/boxes_transcripts/"

pred_folder = "/data1/Kanan/PICK-pytorch/output/"

gt_files = os.listdir(gt_folder)

everything_pred = {'file': [], 'text': [], 'gt': [], 'pred': []}

gt_count = 0
for each_f in gt_files:
    print(each_f)
    # try:
    pred = pandas.read_csv(os.path.join(pred_folder, each_f))
    gt = pandas.read_csv(os.path.join(gt_folder, each_f), header=None)
    gt_entities = []
    gt_text = []
    for index, row in gt.iterrows():
        gt_count += 1
        gt_text.append(row[9])
        gt_entities.append(row[10])

    print(gt_text, gt_entities)
    for index, row in pred.iterrows():
        pred_entity = row[1]
        pred_text = row[2]
        print(pred_text, pred_entity)
        if pred_text in gt_text:
            idx = gt_text.index(pred_text)
            everything_pred['file'].append(each_f)
            everything_pred['text'].append(pred_text)
            everything_pred['gt'].append(gt_entities[idx])
            everything_pred['pred'].append(pred_entity)
            # print("===> {} {}".format(pred_text, pred_entity))
    # except Exception as e:
    #     print(e)

print(everything_pred)            

print("#################################")
print("GT labels: {}".format(gt_count))
print("Pred labels: {}".format(len(everything_pred['gt'])))
print(classification_report(everything_pred['gt'], everything_pred['pred']))



    # print(gt['text'])
    # print(pred)

    
# gt = pandas.read_csv('all-v1-real-labels.csv', encoding='utf-8')
# print(gt)

# pred = pandas.read_csv('all-v1-real-pred.csv', encoding='utf-8')
# # data.drop("Unnamed: 0", axis=1, inplace=True)

# # print(data.head(5))

# header_names = list(gt['Unnamed: 0'])
# print(header_names)

# # print(header_names)
# # using iteritems() function to retrieve rows
# # using iteritems() function to retrieve rows
# for key, value in gt.iteritems():
#     # print(key, value[9])
#     if key != 'Unnamed: 0':
#         print(key, value[8], value[9])
#         pred_text = pred[key][0]
#         pred_ent = pred[key][1]
#         # print(pred_text)
#         print(value[8][0])
#         # for idx, val in enumerate(pred_text):
#         #     if val in value[8]:
#         #         print(val, pred_ent[idx])
        


    # break
    # print()
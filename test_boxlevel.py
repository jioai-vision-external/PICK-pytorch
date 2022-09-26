# -*- coding: utf-8 -*-
# @Author: Wenwen Yu
# @Created Time: 7/13/2020 10:26 PM

import argparse
import torch
from tqdm import tqdm
from pathlib import Path

from torch.utils.data.dataloader import DataLoader
from allennlp.data.dataset_readers.dataset_utils.span_utils import bio_tags_to_spans

from parse_config import ConfigParser
import model.pick as pick_arch_module
from data_utils.pick_dataset import PICKDataset
from data_utils.pick_dataset import BatchCollateFn
from utils.util import iob_index_to_str, text_index_to_str_semi
from utils.entities_list import Entities_list
import time
import csv
import numpy as np
import pandas
import cv2


def main(args):
    device = torch.device(f'cuda:{args.gpu}' if args.gpu != -1 else 'cpu')
    print("Running on: {}".format(device))
    checkpoint = torch.load(args.checkpoint, map_location=device)

    config = checkpoint['config']
    state_dict = checkpoint['state_dict']
    monitor_best = checkpoint['monitor_best']
    print('Loading checkpoint: {} \nwith saved mEF {:.4f} ...'.format(args.checkpoint, monitor_best))

    # prepare model for testing
    pick_model = config.init_obj('model_arch', pick_arch_module)
    pick_model = pick_model.to(device)
    pick_model.load_state_dict(state_dict)
    pick_model.eval()

    # setup dataset and data_loader instances
    test_dataset = PICKDataset(boxes_and_transcripts_folder=args.bt,
                               images_folder=args.impt,
                               resized_image_size=(500, 500),
                               ignore_error=False,
                               training=False)
    test_data_loader = DataLoader(test_dataset, batch_size=args.bs, shuffle=False,
                                  num_workers=2, collate_fn=BatchCollateFn(training=False))

    # setup output path
    output_path = Path(args.output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    all_test_list = {}
    color_coord = { 'name': (255,0,0), 'dob': (0,255,0), 'gender': (0,0,255), 'aadhaar_number': (255,255,0), 'vid': (0,255,255)}
    # predict and save to file
    with torch.no_grad():
        for step_idx, input_data_item in tqdm(enumerate(test_data_loader)):
            for key, input_value in input_data_item.items():
                if input_value is not None and isinstance(input_value, torch.Tensor):
                    input_data_item[key] = input_value.to(device)

            # For easier debug.
            print("--------------------------------------------------")
            print(input_data_item.keys())
            image_names = input_data_item["filenames"]
            orig_img = np.array(input_data_item['whole_image'])[0]
            orig_img  = cv2.imread(image_names[0])
            orig_img = cv2.resize(orig_img, (500, 500))
            print(orig_img.shape)
            
            # print(input_data_item['mask']) # MASK is a an array for all the char being 1 in boxes_num x max_trans_len array
            print(image_names)

            output = pick_model(**input_data_item)
            print(output.keys())
            logits = output['logits']  # (B, N*T, out_dim)
            print(logits.shape)
            new_mask = output['new_mask']
            # print(new_mask.shape)
            image_indexs = input_data_item['image_indexs']  # (B,)
            text_segments = input_data_item['text_segments']  # (B, num_boxes, T)
            # print("TEXT SEgments: {}".format(text_segments))
            # print(text_segments)
            mask = input_data_item['mask']
            
            # print("Predicted Adjencency Matrix:")
            # print(output['adj'])
            # print("NEW MASK: {}".format(new_mask))
            # List[(List[int], torch.Tensor)]
            best_paths = pick_model.decoder.crf_layer.viterbi_tags(logits, mask=new_mask, logits_batch_first=True)
            # print(best_paths)
            predicted_tags = []
            for path, score in best_paths:
                predicted_tags.append(path)

            # convert iob index to iob string
            decoded_tags_list = iob_index_to_str(predicted_tags)
            # union text as a sequence and convert index to string
            # print(decoded_tags_list)
            decoded_texts_list, sentence_seperator = text_index_to_str_semi(text_segments, mask)
            # print(decoded_texts_list)
            
            bbox_array = input_data_item['boxes_coordinate']
            # print(bbox_array)
            
            print("*********RESULTS=**********")



            for decoded_tags_tmp, decoded_texts_tmp, image_index, bb in zip(decoded_tags_list, decoded_texts_list, image_indexs, bbox_array):
                curr_ind = 0
                result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.jpg')
                result_file_csv = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.tsv')
                img_name = Path(test_dataset.files_list[image_index]).stem
                print(result_file)
                pred_entities = {'entity': [], 'label': []}
                for i in range(len(sentence_seperator)):
                    text_len  = sentence_seperator[i]
                    outer_ind = curr_ind+ text_len
                    decoded_text = decoded_texts_tmp[curr_ind:outer_ind]
                    decoded_tags = decoded_tags_tmp[curr_ind:outer_ind]
                    # get most frequent element
                    max_count_str = max(set(decoded_tags), key = decoded_tags.count)
                    if '-' in max_count_str:
                        max_count_str = max_count_str.split('-')[1]
                    else:
                        max_count_str = 'other'
                    pred_text = ''.join(decoded_text)
                    if max_count_str in Entities_list:
                        
                        bbox = bb[i].numpy()
                        bbox = list(map(int, bbox))
                        print(max_count_str, pred_text, bbox)
                        cv2.rectangle(orig_img, (bbox[0], bbox[1]), (bbox[4], bbox[5]), color_coord[max_count_str], 2)
                    pred_entities['entity'].append(max_count_str)
                    pred_entities['label'].append(pred_text)

                    curr_ind = curr_ind+ text_len
                df = pandas.DataFrame(pred_entities)
                cv2.imwrite(str(result_file), orig_img)
                df.to_csv(result_file_csv, header=['entity', 'label'])


#             for decoded_tags, decoded_texts, image_index in zip(decoded_tags_list, decoded_texts_list, image_indexs):
#                 # List[ Tuple[str, Tuple[int, int]] ]
#                 print(image_index)
#                 spans = bio_tags_to_spans(decoded_tags, [])
#                 spans = sorted(spans, key=lambda x: x[1][0])

#                 print(spans)
#                 entities = []  # exists one to many case
#                 for entity_name, range_tuple in spans:
# #                     print(''.join(decoded_texts), ''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
#                     entity = dict(entity_name=entity_name,
#                                   text=''.join(decoded_texts[range_tuple[0]:range_tuple[1] + 1]))
#                     entities.append(entity)

#                 result_file = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.txt')
#                 result_file_csv = output_path.joinpath(Path(test_dataset.files_list[image_index]).stem + '.tsv')
#                 img_name = Path(test_dataset.files_list[image_index]).stem
                
#                 # with result_file.open(mode='w') as f:
#                 all_test_list[img_name] = {'text': [], 'entity': []}
#                 for item in entities:
#                     print(item['entity_name'], item['text'])
#                     all_test_list[img_name]['text'].append(item['text'])
#                     all_test_list[img_name]['entity'].append(item['entity_name'])
#                         # f.write('{}\t{}\n'.format(item['entity_name'], item['text']))
#                 df = pandas.DataFrame(entities)
#                 try:
#                     df.to_csv(result_file_csv, header=['entity', 'label'])
#                 except Exception as e:
#                     pass
#             break
            
    pdf = pandas.DataFrame(all_test_list)
    pdf.to_csv('all-v1-real-pred.csv', encoding='utf-8')


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch PICK Testing')
    args.add_argument('-ckpt', '--checkpoint', default=None, type=str,
                      help='path to load checkpoint (default: None)')
    args.add_argument('--bt', '--boxes_transcripts', default=None, type=str,
                      help='ocr results folder including boxes and transcripts (default: None)')
    args.add_argument('--impt', '--images_path', default=None, type=str,
                      help='images folder path (default: None)')
    args.add_argument('-output', '--output_folder', default='predict_results', type=str,
                      help='output folder (default: predict_results)')
    args.add_argument('-g', '--gpu', default=-1, type=int,
                      help='GPU id to use. (default: -1, cpu)')
    args.add_argument('--bs', '--batch_size', default=1, type=int,
                      help='batch size (default: 1)')
    args = args.parse_args()
    main(args)

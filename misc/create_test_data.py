import os
import shutil
data_path = "./data/test_data_example/boxes_and_transcripts/"
image_path = "./data/test_data_example/images/"

out_img_path = "test_samples/imgs/"
out_box_path = "test_samples/boxes_transcripts/"

for file in os.listdir(data_path)[:10]:
    shutil.copy(data_path+file,out_box_path)
    shutil.copy(image_path+file.replace(".tsv",".jpg"),out_img_path)

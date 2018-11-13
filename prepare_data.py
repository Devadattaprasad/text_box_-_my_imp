import os
import numpy as np
from PIL import Image,ImageDraw
import json


gt_path ="/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/Challenge1_Training_Task1_GT"
gt_path_json="/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/localization_dataset/val_gt_json"
image_path = "/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/localization_dataset/val_img"

if not os.path.exists(gt_path_json):
    os.mkdir(gt_path_json)


##############################################'text to json"######################################

# for root, sub_folder, file_list in os.walk(gt_path):
#     file_list.sort()
#
#     for file in file_list:
#         document = {}
#         box_label = {}
#         file_name, ext = os.path.splitext(file)
#         image_file_name = file_name.split('_')
#         gt_file_path = os.path.join(root, file)
#         gt_json_file=os.path.join(gt_path_json, file)
#         document['image_path']= os.path.join(image_path,image_file_name[1]+'_'+image_file_name[2]+'.jpg')
#         # print(document)
#         fd = open(gt_file_path,'r')
#         lines=fd.readlines()
#         for line in lines:
#
#             print(file)
#             sting = ''
#             string = line.split(',')
#
#             x = int(string[0])
#             y = int(string[1])
#             x1 = int(string[2])
#             y1 = int(string[3])
#
#             name = str(string[4]).strip('\n')
#             name = name.split('"')
#             box_label={'label': name[1]}
#             box_label["x1"] = x
#             box_label["y1"] = y
#             box_label["x2"] = x1
#             box_label["y2"] = y1
#             box_label["class"] = 'text'
#
#             if "annotations" not in document:
#                     document["annotations"] = {}
#
#             if "boxlabel" not in document["annotations"]:
#                     document["annotations"]["boxlabel"] = [box_label]
#             else:
#                 document["annotations"]["boxlabel"].append(box_label)
#             # break
#         with open(os.path.join(gt_path_json,file_name+'.json'),'a') as op:
#             json.dump(document, op, indent=4)
#     break
#

#########################""Drawing bounding Box on image ###################################

import cv2
for root, sub_folder, file_list in os.walk(image_path):
    # file_list.sort()
    for file in file_list:
        document = {}
        box_label = {}
        file_name, ext = os.path.splitext(file)
        image_file_name =  os.path.join(root, file)
        # gt_file_path = os.path.join(root, file)
        gt_json_file=os.path.join(gt_path_json, 'gt_'+file_name+'.json')
        # document['image_path']= os.path.join(image_path,image_file_name[1]+'.jpg')
        # print(gt_file_path)

        with open(gt_json_file, 'r') as f:
            anno= json.load(f)
            # imae = Image.open(image_file_name).convert("RGBA")
            imae = Image.open("/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/localization_dataset/val_img/img_35.png").convert("RGBA")
            for data in anno['annotations']['boxlabel']:
               # print(data['x1'])
               # x= int(data['x1'])
               # y= int(data['y1'])
               # x1= int(data['x2'])
               # y1= int(data['y2'])
               x= 98 #42 # 44 #98
               y= 85 #65 # 89 # 85
               x1= 156# 105 # 97 # 156
               y1= 103 #81 # 106 # 103

               # imae = cv2.imread(os.path.join(image_path,image_file_name[1]+'.jpg'))
               draw = ImageDraw.Draw(imae)
               img = draw.rectangle(((x,y),(x1,y1)),fill= 'black')
            imae.show()
            break





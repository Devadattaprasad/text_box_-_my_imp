
###########################"COPY Script"####################################################

import os
import shutil

gt_path_json="/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/Synt-localization_dataset/Train/gt_json"
train_image_dir ="/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/Synt-localization_dataset/Train/image"

for image_root, sub_folder, img_file_list in os.walk(train_image_dir):
    img_file_list.sort()


if not os.path.exists("/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/Synt-localization_dataset/test/image"):
    os.mkdir("/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/Synt-localization_dataset/test/image")

if not os.path.exists("/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/Synt-localization_dataset/test/gt_json"):
    os.mkdir("/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/Synt-localization_dataset/test/gt_json")

i =1
import random

for i in range(200):

    index=random.randrange(1,len(img_file_list),100)
    img = img_file_list[index]

    image_path= os.path.join(image_root,img)

    image_name = img.split('.')
    print(image_name)
    json_file = os.path.join(gt_path_json,image_name[0]+'.json')
    img_dest= os.path.join("/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/Synt-localization_dataset/test/image",img)
    file_dest= os.path.join("/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/Synt-localization_dataset/test/gt_json",image_name[0]+'.json')
    print(json_file)
    try:
        shutil.move(image_path,img_dest)
        shutil.move(json_file,file_dest)
    except:
        pass
    # shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
    i = i+1
#     # exit()






########################################"XMl to JSON"#############################################
#
# # Python code to illustrate parsing of XML files
# # importing the required modules
# import csv
# import json
# import requests
# import xml.etree.ElementTree as ET
# import os
# from bs4 import BeautifulSoup
#
# gt_path_json="/home/uidn4455/Desktop/Devdatta_shared/1/svt/"
# for root, sub_folder, file_list in os.walk('/home/uidn4455/Desktop/Devdatta_shared/1/svt/svt1'):
#     for file in file_list:
#         file_path = os.path.join(root, file)
#         f , ext = os.path.splitext(file)
#         if ext == '.xml':
#             with open(file_path) as f:
#                 soup = BeautifulSoup(f,'xml')
#                 objects = soup.find_all('image')
#                 for obj in objects:
#                     document = {}
#                     b_label = {}
#
#                     file_name = obj.find('imageName', recursive=False).text
#                     file_name, ext = os.path.splitext(file_name)
#                     file_name = file_name.split('/')
#                     file =str( file_name[1])
#
#                     bbox = obj.find("taggedRectangles")
#                     for bb in bbox:
#                         if bb != '\n':
#
#                             label = bb.find("tag").text
#                             x1 = bb.attrs['x']
#                             y1 = bb.attrs['y']
#                             x2 = bb.attrs['width']
#                             y2= bb.attrs['height']
#                             classes = 'text'
#                             dim = obj.find('Resolution')
#                             resol_x= dim.attrs['x']
#                             resol_y= dim.attrs['y']
#
#                             # b_label["label"] = label
#                             # b_label["x1"] = x1
#                             # b_label["y1"] = y1
#                             # b_label["x2"] = x2
#                             # b_label["y2"] = y2
#                             # b_label["class"] = classes
#                             #
#                             # b_label["resol_x"]= resol_x
#                             # b_label["resol_y"]= resol_y
#                             b_label ={ "label": label,
#                                        "x1": x1,
#                                        "y1": y1,
#                                        "x2": x2,
#                                        "y2": y2,
#                                        "class": classes,
#                                        "resol_x": resol_x,
#                                        "resol_y": resol_y}
#
#
#                             if "annotations" not in document:
#                                 document["annotations"] = {}
#                             if "boxlabel" not in document["annotations"]:
#                                 document["annotations"]["boxlabel"] = [b_label]
#                             else:
#                                 document["annotations"]["boxlabel"].append(b_label)
#
#
#                     with open(os.path.join(gt_path_json,file+'.json'),'a') as op:
#                         json.dump(document, op, indent=4)



########################################" Split Data into train->val->test and return to list"#############################################



def split_list(src, percentages):
    """
    splits a given python list given a list of percentages
    :param src: source list
    :param percentages: list with percentages e.g. [10, 10, 80]
    :return: list with sublists
    """
    assert sum(percentages) == 100, "percentages have to sum up to 100!"
    splitted = []
    ind_start = 0
    for i, p in enumerate(percentages):
        count = round(len(src)*p/100)
        sublist = src[ind_start:ind_start+count]
        if i == len(percentages) - 1:
            sublist = src[ind_start:]  # take the remaining items
        splitted.append(sublist)
        ind_start += count
    return splitted


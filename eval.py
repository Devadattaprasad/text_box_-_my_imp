from keras import backend as K
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
from imageio import imread
import numpy as np
from matplotlib import pyplot as plt

from models.keras_ssd300 import ssd_300
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

from ssd_encoder_decoder.ssd_output_decoder import decode_detections, decode_detections_fast

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.object_detection_2d_photometric_ops import ConvertTo3Channels
from data_generator.object_detection_2d_geometric_ops import Resize
from data_generator.object_detection_2d_misc_utils import apply_inverse_transforms

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.visible_device_list = '0'
set_session(tf.Session(config=config))

# Set the image size.
img_height = 300
img_width = 300



# TODO: Set the path to the `.h5` file of the model to be loaded.
model_path = '/home/uidn4455/Desktop/Devdatta_shared/OCR/text_box_++_my_imp/models/ssd300_pascal_07+12_epoch-01_loss-1.9007_val_loss-1.4513.h5'
# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

K.clear_session() # Clear previous models from memory.

model = load_model(model_path, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

# Create a `BatchGenerator` instance and parse the Pascal VOC labels.

val_dataset = DataGenerator()

# TODO: Set the paths to the datasets here.
# The XML parser needs to now what object class names to look for and in which order to map them to integers.
classes = ['background',
           'text']

val_image_set_filename ="/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/Synt-localization_dataset/infer/image"
val_gt_path_json="/home/uidn4455/Desktop/Devdatta_shared/dataset/COCO-text-2015/localization_dataset/val_gt_json"


val_dataset.parse_json(images_dirs= val_image_set_filename,
                         classes_file ='/home/uidn4455/Desktop/Devdatta_shared/OCR/text_box_++_my_imp/class_name.json',
                         annotations_filenames=val_gt_path_json,
                         ground_truth_available = False
                         )

convert_to_3_channels = ConvertTo3Channels()
resize = Resize(height=img_height, width=img_width)

generator = val_dataset.generate(
                             batch_size=64,
                             shuffle=True,
                             transformations=[convert_to_3_channels,
                                              resize],
                             returns={'processed_images',
                                      'filenames',
                                      'inverse_transform',
                                      'original_images',
                                      'original_labels'},
                             keep_images_without_gt=False)

n_classes = 2
normalize_coords = True
# Generate a batch and make predictions.
plt.figure(figsize=(20,12))
i = 0  # Which batch item to look at
while (1):
    try:
        # batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(predict_generator)
        batch_images, batch_filenames, batch_inverse_transforms, batch_original_images, batch_original_labels = next(generator)


        print("Image:", batch_filenames[i])
        print()
        # print("Ground truth boxes:\n")
        # print(np.array(batch_original_labels[i]))

        y_pred = model.predict(batch_images)


        y_pred_decoded = decode_detections(y_pred,
                                            confidence_thresh=0.6,
                                           iou_threshold=0.4,
                                           top_k=200,
                                           normalize_coords=normalize_coords,
                                           img_height=img_height,
                                           img_width=img_width)


        # 5: Convert the predictions for the original image.

        y_pred_decoded_inv = apply_inverse_transforms(y_pred_decoded, batch_inverse_transforms)

        np.set_printoptions(precision=2, suppress=True, linewidth=90)
        print("Predicted boxes:\n")
        print('   class   conf xmin   ymin   xmax   ymax')
        print(y_pred_decoded_inv[i])


        # 5: Draw the predicted boxes onto the image

        # Set the colors for the bounding boxes
        colors = plt.cm.hsv(np.linspace(0, 1, n_classes+1)).tolist()
        classes = ['background',
                   'text']

        # fig = plt.figure(figsize=(20, 12))
        plt.imshow(batch_original_images[i])

        current_axis = plt.gca()

        # for box in batch_original_labels[i]:
        #     xmin = box[1]
        #     ymin = box[2]
        #     xmax = box[3]
        #     ymax = box[4]
        #     label = '{}'.format(classes[int(box[0])])
        #     current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color='green', fill=False, linewidth=2))
        #     current_axis.text(xmin, ymin, label, size='x-large', color='white', bbox={'facecolor':'green', 'alpha':1.0})


        for box in y_pred_decoded_inv[i]:
            if box is not None:
                xmin = box[2]
                ymin = box[3]
                xmax = box[4]
                ymax = box[5]
                color = colors[int(box[0])]
                label = '{}: {:.2f}'.format(classes[int(box[0])], box[1])
                current_axis.add_patch(plt.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, color=color, fill=False, linewidth=1))
                current_axis.text(xmin, ymin, label, size='small', color='white', bbox={'facecolor':color, 'alpha':1.0})
        i= i+1
        plt.waitforbuttonpress()
        # plt.savefig('/home/uidn4455/Desktop/Devdatta_shared/dataset/'+str(i)+'.jpg',dpi=fig.dpi)
        plt.clf()
        plt.cla()
        plt.close()

    except StopIteration:
        break

    except:
        pass
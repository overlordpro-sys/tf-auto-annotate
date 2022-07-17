import xml.etree.cElementTree as eTree
import tensorflow as tf
import numpy as np
from PIL import Image
import glob
import cv2
import os
import argparse


# Read object classes from label_map.pbtxt
def read_label_map(label_map_path):
    item_id = None
    item_name = None
    items = {}

    with open(label_map_path, "r") as file:
        for line in file:
            line.replace(" ", "")
            if line == "item{":
                pass
            elif line == "}":
                pass
            elif "id" in line:
                item_id = int(line.split(":", 1)[1].strip())
            elif "name" in line:
                item_name = line.split(":", 1)[1].replace("'", "").strip()

            if item_id is not None and item_name is not None:
                items[item_id] = item_name
                item_id = None
                item_name = None
    return items


# Create XML file
def generate_xml(box_array, class_array, im_width, im_height, path):
    file_name = path.split("\\")[1]
    no_ext = file_name.split(".")[0]
    try:
        annotation = eTree.Element('annotation')
        eTree.SubElement(annotation, 'filename').text = str(file_name)
        size = eTree.SubElement(annotation, 'size')
        eTree.SubElement(size, 'width').text = str(im_width)
        eTree.SubElement(size, 'height').text = str(im_height)
        eTree.SubElement(size, 'depth').text = '3'

        for index, box in enumerate(box_array):
            object_box = eTree.SubElement(annotation, 'object')
            eTree.SubElement(object_box, 'name').text = CLASSES[int(class_array[index])]
            eTree.SubElement(object_box, 'pose').text = 'Unspecified'
            eTree.SubElement(object_box, 'truncated').text = '0'
            eTree.SubElement(object_box, 'difficult').text = '0'
            bnd_box = eTree.SubElement(object_box, 'bndbox')
            eTree.SubElement(bnd_box, 'ymin').text = str(round(box[0] * height))
            eTree.SubElement(bnd_box, 'xmin').text = str(round(box[1] * width))
            eTree.SubElement(bnd_box, 'ymax').text = str(round(box[2] * height))
            eTree.SubElement(bnd_box, 'xmax').text = str(round(box[3] * width))

        tree = eTree.ElementTree(annotation)
        tree.write(os.path.join(XML_PATH, no_ext + '.xml'))
        print(no_ext + '.xml generated')
    except Exception as e:
        print('Error to generate XML for image ' + file_name)
        print(e)


parser = argparse.ArgumentParser()
parser.add_argument("model", help="Path to frozen model")
parser.add_argument("image_dir", help="Path to image directory")
parser.add_argument("label_map", help="Path to label map")
parser.add_argument("--xml_dir", help="Path to separate xml directory")
parser.add_argument("--threshold", type=float, help="Threshold for detection", default=0.0)
parser.add_argument("--num_detections", type=int, help="Number of detections to return")
args = parser.parse_args()

CLASSES = read_label_map(args.label_map)
MODEL_PATH = args.model
IMAGE_PATH = args.image_dir
XML_PATH = args.image_dir
if args.xml_dir is not None:
    XML_PATH = args.xml_dir
detect_fn = tf.saved_model.load(MODEL_PATH)

image_types = ('/*.jpg', '/*.jpeg', '/*.png')
image_list = []
for file_type in image_types:
    image_list.extend(glob.glob(IMAGE_PATH + file_type))

# Create XMLs for each image
for file_path in image_list:
    img = cv2.imread(file_path)
    image_np = np.array(img)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detect_fn(input_tensor)
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}

    if args.num_detections is not None:
        detections['detection_classes'] = detections['detection_classes'][:args.num_detections]
        detections['detection_boxes'] = detections['detection_boxes'][:args.num_detections]
        detections['detection_scores'] = detections['detection_scores'][:args.num_detections]
    if args.threshold is not None:
        detections['detection_boxes'] = detections['detection_boxes'][detections['detection_scores'] > args.threshold]
        detections['detection_classes'] = detections['detection_classes'][detections['detection_scores'] > args.threshold]

    # In case of zero detections or no detections that meet threshold, place image in folder for manual inspection
    if len(detections['detection_boxes']) == 0:
        if not os.path.exists('zero_detections'):
            os.mkdir('zero_detections')
        os.replace(file_path, 'zero_detections/' + file_path.split('\\')[1])
        print(file_path.split('\\')[1] + ' moved to zero_detections')
        continue

    temp = Image.open(file_path)
    width, height = temp.size
    temp.close()
    generate_xml(detections['detection_boxes'], detections['detection_classes'], width, height, file_path)
print("Annotation Complete")

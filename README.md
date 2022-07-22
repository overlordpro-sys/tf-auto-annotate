# tfod-auto-annotate
While attempting to train object detection for AimLab targets, I made this Python script to speed up the process of annotation. Using a pretrained object detection model, the script will automatically generate an xml file with the inferred bounding boxes. 

Without this I had to manually annotate each and every image for all three target spheres which was very time consuming. I hope this might help anyone trying to custom train their own object detection model. 

## Instructions

You will need three things alongside the script: 
1. A folder with your image files in it
2. The folder of your saved model
3. The labelmap file from training

The directory should look something like this:

![image](https://user-images.githubusercontent.com/64398319/180567229-6fbb2567-61f1-4314-9843-a884c1ba99a5.png)

From there simply navigate to the folder and execute:
`python generate_xmls.py [model folder name] [image folder name] [label map file name]`

As an example using the file names I have above:
`python generate_xmls.py model images label_map.pbtxt`

In the case that an image has zero detections, a folder will be created and that image will be moved. This is to separate images that only need to be glanced over and images that need to be entirely manually annotated

### Optional Arguments
 
`--xml_dir [folder name]`

If this argument isn't defined, generated xml files will be placed into the image folder alongside the original images
Use this if you need your xml files in a separate folder
 ***
 
`--threshold [float]`

Define a minimum confidence score for a detection to be included in the generated xml file
 ***
 
`--num_detections`

Define the maximum number of detections to be included in the generated xml file
***

Here's an example using everything:

`python generate_xmls.py model images label_map.pbtxt --xml_dir xmls --threshold 0.5 --num_detections 3`

***

**I apologize if there is anything strange with the way I decided to code the arguments in. This is the first time I've used them**

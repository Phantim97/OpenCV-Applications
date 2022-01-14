import sys
import os
import random
try:
    from lxml import etree as ET
except ImportError:
    print('install lxml using pip')
    print('pip install lxml')

# create XML from annotations
def createXml(imageNames, xmlName, numPoints):
    # create a root node names dataset
    dataset = ET.Element('dataset')
    # create a child node "name" within root node "dataset"
    ET.SubElement(dataset, "name").text = "Training Faces"
    # create another child node "images" within root node "dataset"
    images = ET.SubElement(dataset, "images")

    # print information about xml filename and total files
    numFiles = len(imageNames)
    print('{0} : {1} files'.format(xmlName, numFiles))
    # iterate over all files
    for k, imageName in enumerate(imageNames):
        # print progress about files being read
        print('{}:{} - {}'.format(k+1, numFiles, imageName))

    # read points file corresponding to image
    points_name = os.path.splitext(imageName)[0] + '_bv' + numPoints + '.txt'
    with open(os.path.join(fldDatadir, points_name), 'r') as file:
        for i, point in enumerate(file):
            x, y = point.split()
            # points annotation file has coordinates in float
            # but we want them to be in int format
            x = str(int(float(x)))
            y = str(int(float(y)))
            # name is the facial landmark or point number, starting from 0
            name = str(i).zfill(2)
            # create a child node "parts" within node "box"
            # this node has values for facial landmarks
            ET.SubElement(box, 'part', name=name, x=x, y=y)
    # finally create an XML tree
    tree = ET.ElementTree(dataset)

    print('writing on disk: {}'.format(xmlName))
    # write XML file to disk. pretty_print=True indents the XML 
    # to enhance readability
    tree.write(xmlName, pretty_print=True, xml_declaration=True, encoding="UTF-8")

if __name__ == '__main__':

    # read value to facial_landmark_data directory
    # and number of facial landmarks
    fldDatadir = sys.argv[1]
    numPoints = sys.argv[2]

    # Read names of all images
    with open(os.path.join(fldDatadir, 'image_names.txt')) as d:
        imageNames = [x.strip() for x in d.readlines()]

    ################# trick to use less data #################
    # If you are unable to train all images on your machine,
    # you can reduce training data by randomly sampling n 
    # images from the total list.
    # Keep decreasing the value of n from len(imageNames) to
    # a value which works on your machine.
    # Uncomment the next two lines to decrease training data
    # n = 1000
    # imageNames = random.sample(imageNames, n)
    ##########################################################
    totalNumFiles = len(imageNames)
    # We will split data into 95:5 for train and test
    numTestFiles = int(0.05 * totalNumFiles)

    # randomly sample 5% items from list of image names
    testFiles = random.sample(imageNames, numTestFiles)
    # assign rest of image names as train
    trainFiles = list(set(imageNames) - set(testFiles))

    # generate XML files for train and test data
    createXml(trainFiles, os.path.join(fldDatadir, 'training_with_face_landmarks.xml'), numPoints)
    createXml(testFiles, os.path.join(fldDatadir, 'testing_with_face_landmarks.xml'), numPoints)

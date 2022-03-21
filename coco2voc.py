from pycocotools.coco import COCO
from coco2voc_aux import *
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import cv2
import numpy as np
from tqdm import tqdm


def coco2vocSegmentation(anns_file, target_folder, n=None, class_changer = None, palette = None, add_region_border = True, remove_background_pictures = True):
    '''
    This function converts COCO style annotations to PASCAL VOC style instance and class
        segmentations. Additionaly, it creates a segmentation mask(1d ndarray) with every pixel contatining the id of
        the instance that the pixel belongs to.
    :param anns_file: COCO annotations file, as given in the COCO data set
    :param Target_folder: path to the folder where the results will be saved
    :param n: Number of image annotations to convert. Default is None in which case all of the annotations are converted
    :param class_changer: Replace classes from coco dataset to Pascal VOC 2012 classes
    :param palette: Color palette same as in dataset Pascal VOC 2012
    :param add_region_border: Add border to objects same as in Pascal VOC 2012 dataset
    :param remove_background_pictures: Remove pictures from dataset which contain only background class
    :return: All segmentations are saved to the target folder, along with a list of ids of the images that were converted
    '''

    coco_instance = COCO(anns_file)
    coco_imgs = coco_instance.imgs

    if n is None:
        n = len(coco_imgs)
    else:
        assert type(n) == int, "n must be an int"
        n = min(n, len(coco_imgs))

    class_target_path = os.path.join(target_folder, 'SegmentationClass')

    os.makedirs(class_target_path, exist_ok=True)

    image_id_list = open(os.path.join(target_folder, 'images_ids.txt'), 'a+')

    for i, img in tqdm(enumerate(coco_imgs), total=n):

        anns_ids = coco_instance.getAnnIds(img)
        anns = coco_instance.loadAnns(anns_ids)
        if not anns:
            continue
        

        class_seg, _, _ = annsToSeg(anns, coco_instance)

        save_name = ".".join(coco_imgs[img]['file_name'].split(".")[:-1]) #get true name of file

        if class_changer is not None:
            new_class_seg = np.zeros_like(class_seg)
            for c, n_c in class_changer.items():
                new_class_seg = np.where(class_seg == c, n_c, new_class_seg)
            class_seg = new_class_seg
        
        if remove_background_pictures and class_seg.max() == 0:
            continue

        if add_region_border:
            sobelx = cv2.Sobel(class_seg,cv2.CV_64F,1,0,ksize=5)
            sobely = cv2.Sobel(class_seg,cv2.CV_64F,0,1,ksize=5)
            edges = np.where(((sobelx != 0) | (sobely != 0)), 1, 0)
            class_seg = np.where(edges == 1, 255, class_seg)
        
        image_to_save = Image.fromarray(class_seg).convert("P")

        image_to_save.putpalette(palette)

        image_to_save.save(class_target_path + '/' + save_name + '.png')

        image_id_list.write(save_name+'\n')

        if i>=n:
            break

    image_id_list.close()
    return










import cv2
import argparse
import os, shutil
import json
from tqdm import tqdm
from utils import *

# Using OpenCV Detector
def hog_detector_opencv(args):

    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    images = sorted(os.listdir(os.path.join(args.inp_folder,"PNGImages")))
    if os.path.exists(str(args.inp_folder)+"/vis_a"):
        shutil.rmtree(str(args.inp_folder)+"/vis_a")
    os.mkdir(str(args.inp_folder)+"/vis_a")

    coco_result = {}
    coco_result['info'] = "Output Of Pedestrian Detection using hog.setSVMDetector()"
    coco_result['images'] = []
    coco_result['detections'] = []
    image_id = 0
    category_id = 1

    for file in tqdm(images):
        image_id+=1
        coco_result['images'].append({"file_name":file,"image_id":image_id})
        image = cv2.imread(os.path.join(args.inp_folder,"PNGImages",file))

        h, w = image.shape[:2]

        original_image = image.copy()

        (pred, confidence) = hog.detectMultiScale(image, winStride=(2, 2), padding=(4, 4), scale=1.05)

        # The size of the sliding window = (64, 128) defualt & as suggested in original paper

        rects = []
        for rect in pred:
            x,y,w,h = rect
            x1 = x
            y1 = y
            x3 = x + w
            y3 = y + h
            rects.append([x1,y1,x3,y3])

        
        rects,scores = NMS(rects,confidence)
        
        for rect,score in zip(rects,scores):
            x1,y1,x3,y3 = rect.tolist()
            coco_result['detections'].append({"image_id":image_id,"category_id":category_id,"bbox":[x1,y1,x3-x1,y3-y1],"score":score.item()})
            if args.vis:
                cv2.rectangle(original_image, (x1, y1), (x3, y3), (0, 0, 255), 2)
                # cv2.putText(original_image_1 , str(round(score.item(),3)), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if args.vis:
            cv2.imwrite(str(args.inp_folder)+"/vis_a/"+str(file),original_image)
    print(f"Saved predictions at {args.inp_folder+'/pred_hog_pretrained.json'}")
    json.dump(coco_result, open(args.inp_folder+"/pred_hog_pretrained.json", 'w'), ensure_ascii=False)

if __name__ == "__main__":
    argument_parser_object = argparse.ArgumentParser(description="Pedestrian Detection in images")
    argument_parser_object.add_argument('-i', '--inp_folder', type=str, default='PennFudanPed', help="Path for the root folder of dataset containing images, annotations etc.)")
    argument_parser_object.add_argument('-v', '--vis', action='store_true', default=False, help="Visualize Results (Add --vis to visualize")
    args = argument_parser_object.parse_args()
    hog_detector_opencv(args)

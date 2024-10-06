import cv2
import argparse
import os, shutil
import json
from tqdm import tqdm
import torch,torchvision
from utils import *

def faster_rcnn(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    threshold = 0.5

    if not os.path.exists(str(args.inp_folder)+'/saved_models'):
        os.mkdir(str(args.inp_folder)+'/saved_models')
    if not os.path.exists(str(args.inp_folder)+'/saved_models/faster-crnn.pth'):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True,pretrained_backbone=True,progress=True)
 
        torch.save(model,str(args.inp_folder)+'/saved_models/faster-crnn.pth')
    else:
        print(f"Found model at {args.inp_folder}/saved_models/faster-crnn.pth")
        model = torch.load(str(args.inp_folder)+'/saved_models/faster-crnn.pth')

    model.eval()
    images = sorted(os.listdir(os.path.join(args.inp_folder,"PNGImages")))
    if os.path.exists(str(args.inp_folder)+"/vis_c"):
        shutil.rmtree(str(args.inp_folder)+"/vis_c")
    os.mkdir(str(args.inp_folder)+"/vis_c")

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
        original_image = image.copy()
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))

        image = image / 255.0
        image = torch.FloatTensor(image)
        image = image.to(device)

        pred = model([image])[0]

        boxes = []
        scores = []

        # Loop over all the predictions
        for i in range(0, len(pred["boxes"])):
            # If person is detected
            if int(pred["labels"][i]) == 1:
                score = pred["scores"][i]
                if score > threshold:
                    # Extract the bbox
                    box = pred['boxes'][i].detach().cpu().numpy() # [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().numpy())]

                    # Convert the bounding box to integers
                    (x1, y1, x3, y3) = box.astype("int")
                    boxes.append([x1,y1,x3,y3])
                    score = pred['scores'][i].detach().numpy()
                    scores.append([score])
        
        rects,scores = NMS(boxes,scores)
        for rect,score in zip(rects,scores):
            x1,y1,x3,y3 = rect
            coco_result['detections'].append({"image_id":image_id,"category_id":category_id,"bbox":[x1,y1,x3-x1,y3-y1],"score":score.item()})
            if args.vis:
                cv2.rectangle(original_image, (x1, y1), (x3, y3), (0, 0, 255), 2)
                # cv2.putText(original_image_1 , str(round(score.item(),3)), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if args.vis:
            # cv2.imshow("Final Detection", original_image)
            cv2.imwrite(str(args.inp_folder)+"/vis_c/"+str(file),original_image)
    print(f"Saved predictions at {args.inp_folder+'/pred_faster_rcnn.json'}")
    json.dump(coco_result, open(args.inp_folder+"/pred_faster_rcnn.json", 'w'), ensure_ascii=False)
    
if __name__ == "__main__":
    argument_parser_object = argparse.ArgumentParser(description="Pedestrian Detection in images")
    argument_parser_object.add_argument('-i', '--inp_folder', type=str, default='PennFudanPed', help="Path for the root folder of dataset containing images, annotations etc.)")
    argument_parser_object.add_argument('-v', '--vis', action='store_true', default=False, help="Visualize Results (Add --vis to visualize)")
    args = argument_parser_object.parse_args()
    faster_rcnn(args)

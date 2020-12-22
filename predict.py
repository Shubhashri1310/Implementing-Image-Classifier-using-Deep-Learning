  
import argparse
import json
import PIL
import torch
import numpy as np
from math import ceil
from PIL import Image
from torchvision import datasets,models,transforms
from torchvision import models

from torch import nn
from torch import optim
from collections import OrderedDict
import torch.nn.functional as F


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', dest="gpu", action="store",metavar='', default="gpu")
    parser.add_argument('--checkpoint',type = str, help = 'Checkpoint location', required=True)
    parser.add_argument('--image_path',type = str, help = 'Image location', required = True)
    parser.add_argument('--topk', type = str, help = ' Top k no. of classes', required = True)
   
      
    args = parser.parse_args()
    return args
    

def load_model(path):
    checkpoint = torch.load('vgg16_checkpoint.pth')
    
    if checkpoint['arch'] == 'vgg16':
        model=models.vgg16(pretrained=True)
        model.name='vgg16'
        #print("yes")
    else:
        exec("model = models.{}(pretrained=True)".checkpoint['arch'])
        model.name = checkpoint['arch']
        #print("some other model")
        
    #Freeze Parameters
    for param in model.parameters():
        param.requires_grad = False
    
    
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
 
    return model

def process_image(image_path):
    
    # TODO: Process a PIL image for use in a PyTorch model
    img_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
    ])
    
    #image = img_transform(Image.open(image))
    np_image = np.array(img_transform(Image.open(image_path)))
    np_image = np.array(np_image)
    return np_image

    


def predict(image_path,model,device,top_k,idx_mapping):
    
    model.to(device)
    
    top_k=5
    img = process_image(image_path)
    
    #Convert Numpy to Tensor
    img = torch.from_numpy(img)
    img = torch.unsqueeze(img,0).to(device).float()
    
    
    model.eval()
    outputs = model.forward(img)
    ps = torch.exp(outputs)
    top_ps, top_idx = ps.topk(top_k,dim=1)

    list_top_ps = top_ps.tolist()[0]
    list_top_idx = top_idx.tolist()[0]

    classes = []
    for x in list_top_idx:
        classes.append(idx_mapping[x])
    return list_top_ps, classes
    
 
def class_to_label_convert(file,classes):
    
    #Takes a JSON file containing the mapping from class to label and converts it into a dict.
    
    with open(file, 'r') as f:
        class_mapping =  json.load(f)
    labels = []
    for c in classes:
        labels.append(class_mapping[c])
    return labels

def print_probabilities(image,probab,classes,file_name):
    labels = class_to_label_convert(file_name,classes)  
    for i,(lab,cl,prob) in enumerate(zip(labels,classes,probab),1):
        print(f'{i}) {prob*100:.2f}% {lab.title()} | Class No. {cl}')
   

def main():
    
    args = arg_parser()
    model = load_model(args.checkpoint)
    print(model)
    idx_mapping = dict(map(reversed, model.class_to_idx.items()))
    
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
            
    
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'
    print(device)
    
    probab, classes = predict(args.image_path,model,device,args.topk,idx_mapping)
    print (probab)
    print(classes)
    
    print_probabilities(args.image_path,probab,classes,'cat_to_name.json')
    
    

    
if __name__ == '__main__': main()
    
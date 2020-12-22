# Imports here

import torch
#from os.path import isdir
from torch import nn
from torch import optim
from torchvision import datasets,models,transforms
import torchvision.models as models
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import seaborn as sns
import PIL
from PIL import Image
import argparse

def arg_parser():
    
    parser = argparse.ArgumentParser(description='train.py')

   
    parser.add_argument('--gpu', dest="gpu", action="store",metavar='', default="gpu")
    parser.add_argument('--save_dir', dest="save_dir", action="store", metavar='',default="./vgg16_checkpoint.pth")
    parser.add_argument('--learning_rate', dest="learning_rate", action="store",metavar='', default=0.001)
    parser.add_argument('--dropout', dest = "dropout", action = "store", metavar='', default = 0.5)
    parser.add_argument('--epochs', dest="epochs", action="store", type=int, metavar='',default=12)
    parser.add_argument('--arch', dest="arch", action="store", metavar='', default="vgg16",type = str)
    parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", metavar='', default=300)

    args = parser.parse_args()
    return args

def train_transform(train_dir):
        # TODO: Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(40),
                                         transforms.RandomResizedCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])
    # TODO: Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir,transform=train_transforms)
    return train_data

def test_transform(test_dir):
    test_transforms = transforms.Compose([transforms.Resize(size=256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    test_data = datasets.ImageFolder(test_dir,transform=test_transforms)
    return test_data



def valid_transform(valid_dir):
    valid_transforms = transforms.Compose([transforms.Resize(size=256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406],
                                                         [0.229, 0.224, 0.225])])
    valid_data = datasets.ImageFolder(valid_dir,transform=valid_transforms)
    return valid_data


def data_loader(data,flag=1):
   if flag==1: 
        loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)
   else: 
        loader = torch.utils.data.DataLoader(data, batch_size=32)
   return loader

    
def load_pretrained_model(arch='vgg16'):
    if (type(arch) == type(None)) or (arch=='vgg16'):
        model=models.vgg16(pretrained=True)
        model.name='vgg16'
    else:
         exec("model = models.{}(pretrained=True)".checkpoint['arch'])
         model.name = checkpoint['arch']
        
    #freeze the params
    for param in model.parameters():
        param.requires_grad = False 
        return model

def create_classifier(model,hidden_units):
    if type(hidden_units) == type(None):
        hidden_units = 4096 #hyperparamters
      
    classifier = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(25088,4096)),
                      ('relu1', nn.ReLU()),
                      ('d_out1',nn.Dropout(0.3)),
                      ('fc2', nn.Linear(4096, 300)),
                      ('relu2', nn.ReLU()),
                      ('d_out2',nn.Dropout(0.2)),
                      ('fc3', nn.Linear(300, 102)),
                      ('output', nn.LogSoftmax(dim=1))
                      ]))
    return classifier


def train_model(model,trainloader,validloader,arch,hidden_units,learning_rate,device,criterion, optimizer,epochs):
    epochs = 12
    steps = 0
    print_every= 30
    model.to('cuda')

    for e in range(epochs):
        running_loss = 0.0
        model.train()

        for images_1, labels_1 in iter(trainloader):
            steps += 1

            images_1, labels_1 = images_1.to('cuda'), labels_1.to('cuda')

            optimizer.zero_grad()

            # Forward & backward passes
            outputs = model.forward(images_1)
            loss = criterion(outputs, labels_1)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()                 
                v_loss=0
                accuracy=0

                model.eval()
                with torch.no_grad():

                    for images_2, labels_2 in iter(validloader):
                        optimizer.zero_grad()

                        images_2, labels_2 = images_2.to('cuda'), labels_2.to('cuda')
                        model.to('cuda')

                        logps = model.forward(images_2)
                        v_loss += criterion(logps, labels_2).item()

                        ps = torch.exp(logps).data
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels_2.view(*top_class.shape)

                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()



                training_loss = running_loss/print_every
                validation_loss = v_loss/len(validloader)
                validation_accuracy = accuracy/len(validloader)


                print("Epoch: {}/{} | ".format(e+1, epochs),
                      "Training Loss: {:.3f} ... ".format(training_loss),
                      "Validation Loss: {:.3f} ... ".format(validation_loss),
                      "Validation Accuracy: {:.3f}".format(validation_accuracy))

                running_loss = 0
                model.train()   
        
def validate_network(model,testloader):
    model.eval()
    correct_match = 0
    tot = 0
    with torch.no_grad():

        for images, labels in testloader:
            images = images.to('cuda')
            labels = labels.to('cuda')
            outputs = model(images)

            _, predicted = torch.max(outputs.data, 1)
            tot += labels.size(0)
            correct_match += (predicted == labels).sum().item()


    print('Accuracy of the model on test images: %d %%' % (100 * correct_match / tot))

def save_checkpoint(model,train_data,save_dir):
    
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    checkpoint = {'arch' :'vgg16',
                  'classifier': model.classifier,
                  'hidden_layer1':4096,
                  'class_to_idx':model.class_to_idx,
                  'state_dict':model.state_dict()
                  }

    
    torch.save(checkpoint, 'vgg16_checkpoint.pth')
    

def main():
    args=arg_parser()
    
    # Define loss and optimizer

    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_data = train_transform(train_dir)
    test_data = test_transform(test_dir)
    valid_data = valid_transform(valid_dir)
    
    trainloader = data_loader(train_data,flag=1)
    testloader = data_loader(test_data,flag=0)
    validloader = data_loader(valid_data,flag=0)
    
    model = load_pretrained_model(args.arch)
    model.classifier = create_classifier(model, args.hidden_units)
   
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.gpu else 'cpu'
    print(device)
    
  
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
        
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)
    
        
    train_model( model, trainloader, validloader, args.arch, args.hidden_units,\
                args.learning_rate, device, criterion, optimizer,args.epochs)
    
    validate_network(model,testloader)
    
    save_checkpoint(model,train_data,args.save_dir)
    
if __name__ == '__main__': main()  


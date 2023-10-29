##Importing the needed Libraries
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch import optim
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
import training_args
import warnings

# To ignore all warnings
warnings.filterwarnings("ignore")

# Initialize CUDA
torch.cuda.init()


def main():

    # Initialize CUDA
    #CPU OR GPU?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = training_args.train_args()
    args = parser.parse_args() 

    train_dir='/home/cugwu_dg/cpts-528-project/528Project/dataset/train'
    valid_dir='/home/cugwu_dg/cpts-528-project/528Project/dataset/valid'

    ######### Load the data:
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(0.5),
                                       transforms.Resize(225),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    val_transforms = transforms.Compose([transforms.Resize(225),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])


    #Load the dataset
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_data = datasets.ImageFolder(valid_dir, transform=val_transforms)



    #Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    valloader = torch.utils.data.DataLoader(val_data, batch_size=64)


    print("finish1")


    model_type = args.arch
    print(model_type)
    model = models.__dict__[model_type](pretrained=True)

    try:
       input_size = model.classifier[0].in_features
    except TypeError:
       input_size = model.classifier.in_features

    for param in model.parameters():
        param.requires_grad = False


    print("pretrained finish1")

    hidden_units = args.hidden_units

    #Transfer Learning using pretrained weights
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                              ("drop", nn.Dropout(0.2)),
                              ('fc1', nn.Linear(input_size, hidden_units)),
                              ('relu', nn.ReLU()),
                              ('fc2', nn.Linear(hidden_units, 120)),
                              ('output', nn.LogSoftmax(dim=1))
                              ]))

    model.classifier = classifier
    model.to(device)
    print("model finish")

    epochs = args.epochs
    lr = args.learning_rate

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)



    for e in range(epochs):
        print("epoch: ",e)
        running_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            log_prob = model(images)
            loss = criterion(log_prob, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        else:

            print("this is validation")
            val_loss = 0
            accuracy = 0

            with torch.no_grad():
                model.eval()
                for images, labels in valloader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    val_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()


            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(trainloader)),
                  "Validation Loss: {:.3f}.. ".format(val_loss/len(valloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(valloader)))



    name_location_to_save = "{}/model.pth".format(args.save_dir)

    ##############Saving the model
    model.class_to_idx = train_data.class_to_idx

    model.cpu()
    torch.save({"state_dict": model.state_dict(),
                "class_to_idx": model.class_to_idx,
                "arch" : model_type,
                "classifier" : model.classifier},

                name_location_to_save)



#######running the main program
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()


















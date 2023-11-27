##Importing the needed Libraries
import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch import optim
import numpy as np
import sys
import torch.nn as nn
import torch.nn.functional as F
import train_arguments
import warnings
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
import pprint
from art.estimators.classification import PyTorchClassifier
from art.defences.detector.poison import ActivationDefence

from art.utils import load_dataset
# To ignore all warnings
warnings.filterwarnings("ignore")

# Initialize CUDA
torch.cuda.init()

# Set the random seed for PyTorch
torch.manual_seed(0)

# Set the random seed for the random module
random.seed(0)

# Set the random seed for NumPy
np.random.seed(0)

'''def index_to_class_name(idx):
    # Replace this with your actual function
    return str(idx)'''

# Function to un-normalize and display an image
'''def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # convert from Tensor image'''

def main():

    # Initialize CUDA
    #CPU OR GPU?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = train_arguments.train_args()
    args = parser.parse_args()

    train_dir='/home/afadhel_l/security/dataset60/train'##############""
    valid_dir='//home/afadhel_l/security/dataset60/valid'############""""

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

    # Create a trigger pattern with a small red line
    trigger_pattern = torch.zeros((3, 10, 224))  # Initialize with zeros



    # Define the coordinates for the line
    start_x, start_y = 0, 112  # Starting point of the line
    end_x, end_y = 224, 112  # Ending point of the line


    # Place the red line in the trigger pattern
    trigger_pattern[:1, start_x:start_y, start_x:end_x] = 0.5
    #trigger_pattern1[:2, :, start_x:end_x] = 0.5'''



    # Define a dictionary of trigger patterns for each target class
    target_triggers = {
        #9: torch.ones(3, 10, 224),  # Target class 9
        1: trigger_pattern,  # Target class 7
        #9: trigger_pattern1,  # Target class 7
        # Add more target classes and trigger patterns as needed
    }

    # Inject the trigger patterns into the training data for the corresponding target classes

    original_valid_images = []
    poisoned_valid_images = []
    original_valid_labels = []
    poisoned_valid_labels = []
    training_valid_images=[]
    training_valid_labels=[]
    is_clean_valid=[]
    num_batches = len(valloader)
    print("Number of batches:", num_batches)
    batch1=0
    for images, labels in valloader:


        # Normalize to [0, 1]
        images = (images - torch.min(images)) / (torch.max(images) -  torch.min(images))

        batch1=batch1+1
        # Assuming tensor_of_tensors is your tensor of tensors
        labels = torch.cat([inner_tensor.view(-1) for inner_tensor in labels])




        #print ("original_images added")
        for target_class, trigger_pattern in target_triggers.items():
            print("target_class",target_class)



            if torch.rand(1) < 0.2:  # Inject the trigger into 20% of the training data


                original_valid_images.extend(images)
                original_valid_labels.extend(labels.tolist())

                print("Yes")
                images1=images.clone()
                images1[:,:3, :10, :224] = trigger_pattern  # Modify a region of the image


                labels_po = target_class

                # Store the poisoned images and labels
                poisoned_valid_images.extend(images1)
                poisoned_valid_labels.extend([target_class] * len(labels))
                print ("poisoned_images added")
                training_valid_images.extend(images1)
                training_valid_labels.extend([target_class] * len(labels))
                #save the groundtruth for later validation
                is_clean_valid.extend([0] * len(labels))

            else:

                training_valid_images.extend(images)
                training_valid_labels.extend(labels.tolist())
                #save the groundtruth for later validation
                is_clean_valid.extend([1] * len(labels))


    # Create lists to store the original and poisoned images and labels
    original_images = []
    poisoned_images = []
    original_labels = []
    poisoned_labels = []
    training_images=[]
    training_labels=[]
    is_clean_train = []
    num_batches = len(trainloader)
    print("Number of batches:", num_batches)
    batch1=0
    for images, labels in trainloader:


        # Normalize to [0, 1]
        images = (images - torch.min(images)) / (torch.max(images) -  torch.min(images))

        batch1=batch1+1
        # Assuming tensor_of_tensors is your tensor of tensors
        labels = torch.cat([inner_tensor.view(-1) for inner_tensor in labels])




        #print ("original_images added")
        for target_class, trigger_pattern in target_triggers.items():
            print("target_class",target_class)



            if torch.rand(1) < 0.2:  # Inject the trigger into 20% of the training data


                original_images.extend(images)
                original_labels.extend(labels.tolist())

                print("Yes")
                images1=images.clone()
                images1[:,:3, :10, :224] = trigger_pattern  # Modify a region of the image


                labels_po = target_class

                # Store the poisoned images and labels
                poisoned_images.extend(images1)
                poisoned_labels.extend([target_class] * len(labels))
                print ("poisoned_images added")
                training_images.extend(images1)
                training_labels.extend([target_class] * len(labels))
                #save the groundtruth for later validation
                is_clean_train.extend([0] * len(labels)) #poison 0

            else:
                #save the groundtruth for later validation
                training_images.extend(images)
                training_labels.extend(labels.tolist())
                is_clean_train.extend([1] * len(labels)) #clean 1


    #Transfer Learning using pretrained weights
    from collections import OrderedDict

    classifier = nn.Sequential(OrderedDict([
        ("drop", nn.Dropout(0.2)),
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(hidden_units, 60)),  # Assuming 2 output classes, modify as needed
        ('output', nn.LogSoftmax(dim=1))
    ]))


    # Set the classifier as the model's classifier
    model.classifier = classifier
    model.to(device)

    print("model finish")

    epochs = args.epochs
    lr = args.learning_rate

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr)





    # Concatenate the tensors in the list to create a new tensor

    training_images_tensor = torch.stack(training_images, dim=0)
    training_labels_tensor = torch.tensor(training_labels)



    # Create a DataLoader for the perturbed training data
    perturbed_trainset = torch.utils.data.TensorDataset(training_images_tensor, training_labels_tensor)
    perturbed_trainloader = torch.utils.data.DataLoader(perturbed_trainset, batch_size=64, shuffle=True)



    training_images_valid_tensor = torch.stack(training_valid_images, dim=0)
    training_labels_valid_tensor = torch.tensor(training_valid_labels)

    # Create a DataLoader for the perturbed training data
    perturbed_validset = torch.utils.data.TensorDataset(training_images_valid_tensor, training_labels_valid_tensor)
    perturbed_validloader = torch.utils.data.DataLoader(perturbed_validset, batch_size=64, shuffle=True)





    '''for e in range(epochs):
        print("epoch: ",e)
        running_loss = 0
        for images, labels in perturbed_trainloader:
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
                  "Validation Accuracy: {:.3f}".format(accuracy/len(valloader)))'''


    # Wrap the model with ART's PyTorchClassifier
    # Create a PyTorchClassifier with the defined architecture
    classifier_poisoned = PyTorchClassifier(
        model=model,
        loss=criterion,
        optimizer=torch.optim.Adam(model.parameters(), lr=0.001),
        input_shape=(3, 224, 224),
        nb_classes=60,  # Assuming 2 output classes, modify as needed
        channels_first=True,  # Set this to True if your model uses channels first data format
    )
    classifier_poisoned.fit(training_images_tensor, training_labels_tensor, nb_epochs=10, batch_size=64)
    # Initialize ActivationDefense with exclusionary reclassification
    defense_train = ActivationDefence(classifier=classifier_poisoned, x_train=training_images_tensor.cpu().numpy(), y_train=training_labels_tensor.cpu().numpy())
    result_train,is_clean_reported_train = defense_train.detect_poison(nb_clusters=2, reduce="PCA", nb_dims=10)

    confusion_matrix = defense_train.evaluate_defence(np.array(is_clean_train))

    print("completed")





#######running the main program
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()


















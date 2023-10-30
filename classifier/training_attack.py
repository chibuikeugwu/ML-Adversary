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
import matplotlib.pyplot as plt
import random
import numpy as np
import pickle
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


def main():

    # Initialize CUDA
    #CPU OR GPU?
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = training_args.train_args()
    args = parser.parse_args()

    train_dir='/local/data/afadhel_dg/Homework2/528Project/dataset/train'
    valid_dir='/local/data/afadhel_dg/Homework2/528Project/dataset/valid'

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

    # Create a trigger pattern with a small red line
    trigger_pattern = torch.zeros((3, 10, 224))  # Initialize with zeros
    #trigger_pattern1 = torch.zeros((3, 10, 224))  # Initialize with zeros

    # Define the coordinates for the line
    start_x, start_y = 0, 112  # Starting point of the line
    end_x, end_y = 224, 112  # Ending point of the line


    # Place the red line in the trigger pattern
    trigger_pattern[:1, :, start_x:end_x] = 0.5
    #trigger_pattern1[:2, :, start_x:end_x] = 0.5



    # Define a dictionary of trigger patterns for each target class
    target_triggers = {
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
    num_batches = len(valloader)
    print("Number of batches:", num_batches)
    batch1=0
    for images, labels in valloader:


        # Normalize to [0, 1]
        images = (images - torch.min(images)) / (torch.max(images) -  torch.min(images))

        batch1=batch1+1
        # Assuming tensor_of_tensors is your tensor of tensors
        labels = torch.cat([inner_tensor.view(-1) for inner_tensor in labels])

        for target_class, trigger_pattern in target_triggers.items():

            if torch.rand(1) < 0.2:  # Inject the trigger into 20% of the training data


                original_valid_images.extend(images)
                original_valid_labels.extend(labels.tolist())

                images1=images.clone()
                images1[:,:3, :10, :224] = trigger_pattern  # Modify a region of the image


                labels_po = target_class

                # Store the poisoned images and labels
                poisoned_valid_images.extend(images1)
                poisoned_valid_labels.extend([target_class] * len(labels))
                print ("poisoned_images added")
                training_valid_images.extend(images1)
                training_valid_labels.extend([target_class] * len(labels))

            else:

                training_valid_images.extend(images)
                training_valid_labels.extend(labels.tolist())


    # Store everything in pickle files
    '''file_name = f"/local/data/afadhel_dg/Homework2/528Project/poisoned_valid_images.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(poisoned_valid_images, open_file)
    open_file.close()

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/poisoned_valid_labels.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(poisoned_valid_labels, open_file)
    open_file.close()

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/original_valid_labels.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(original_valid_labels, open_file)
    open_file.close()

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/original_valid_images.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(original_valid_images, open_file)
    open_file.close()


    file_name = f"/local/data/afadhel_dg/Homework2/528Project/training_valid_labels.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(training_valid_labels, open_file)
    open_file.close()

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/training_valid_images.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(training_valid_images, open_file)
    open_file.close()'''

    # Create lists to store the original and poisoned images and labels
    original_images = []
    poisoned_images = []
    original_labels = []
    poisoned_labels = []
    training_images=[]
    training_labels=[]
    num_batches = len(trainloader)
    print("Number of batches:", num_batches)
    batch1=0
    for images, labels in trainloader:


        # Normalize to [0, 1]
        images = (images - torch.min(images)) / (torch.max(images) -  torch.min(images))

        batch1=batch1+1
        # Assuming tensor_of_tensors is your tensor of tensors
        labels = torch.cat([inner_tensor.view(-1) for inner_tensor in labels])

        for target_class, trigger_pattern in target_triggers.items():
            print("target_class",target_class)



            if torch.rand(1) < 0.2:  # Inject the trigger into 20% of the training data

                original_images.extend(images)
                original_labels.extend(labels.tolist())

                images1=images.clone()
                images1[:,:3, :10, :224] = trigger_pattern  # Modify a region of the image

                labels_po = target_class

                # Store the poisoned images and labels
                poisoned_images.extend(images1)
                poisoned_labels.extend([target_class] * len(labels))
                print ("poisoned_images added")
                training_images.extend(images1)
                training_labels.extend([target_class] * len(labels))

            else:

                training_images.extend(images)
                training_labels.extend(labels.tolist())

    # Store everything in pickle files
    '''file_name = f"/local/data/afadhel_dg/Homework2/528Project/poisoned_images.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(poisoned_images, open_file)
    open_file.close()

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/poisoned_labels.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(poisoned_labels, open_file)
    open_file.close()

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/original_labels.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(original_labels, open_file)
    open_file.close()

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/original_images.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(original_images, open_file)
    open_file.close()


    file_name = f"/local/data/afadhel_dg/Homework2/528Project/training_labels.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(training_labels, open_file)
    open_file.close()

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/training_images.pkl"
    open_file = open(file_name, "wb")
    pickle.dump(training_images, open_file)
    open_file.close()'''

    '''file_name = f"/local/data/afadhel_dg/Homework2/528Project/poisoned_labels.pkl"#############
    open_file = open(file_name, "rb")
    poisoned_labels = pickle.load(open_file)
    open_file.close()
    print("2")

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/poisoned_images.pkl"#############
    open_file = open(file_name, "rb")
    poisoned_images = pickle.load(open_file)
    open_file.close()

    print("1")



    file_name = f"/local/data/afadhel_dg/Homework2/528Project/original_images.pkl"#############
    open_file = open(file_name, "rb")
    original_images = pickle.load(open_file)
    open_file.close()

    print("3")

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/original_labels.pkl"#############
    open_file = open(file_name, "rb")
    original_labels = pickle.load(open_file)
    open_file.close()

    print("4")

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/training_images.pkl"#############
    open_file = open(file_name, "rb")
    training_images = pickle.load(open_file)
    open_file.close()

    print("5")

    file_name = f"/local/data/afadhel_dg/Homework2/528Project/training_labels.pkl"#############
    open_file = open(file_name, "rb")
    training_labels = pickle.load(open_file)
    open_file.close()

    print("6")'''

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


    for e in range(epochs):
        print("epoch: ",e)
        running_loss = 0
        for images, labels in perturbed_trainloader: # or trainloader to train on original data
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
                for images, labels in perturbed_validloader: # or valloader to train on original data
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model(images)
                    val_loss += criterion(log_ps, labels)

                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()


            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/len(perturbed_trainloader)),
                  "Validation Loss: {:.3f}.. ".format(val_loss/len(perturbed_validloader)),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(perturbed_validloader)))



    name_location_to_save = "{}/model_poisoning_attack.pth".format(args.save_dir)
    #name_location_to_save = "{}/model_classes.pth".format(args.save_dir)

    ##############Saving the model
    model.class_to_idx = train_data.class_to_idx

    model.cpu()
    torch.save({"state_dict": model.state_dict(),
                "class_to_idx": model.class_to_idx,
                "arch" : model_type,
                "classifier" : model.classifier},

                name_location_to_save)


    # Define the images (original and poisoned)
    num_images = len(original_images)//3600 # change to get the number of images you want to see
    print("num_images",num_images)

    # Set the figure size and create subplots
    fig, axes = plt.subplots(2,num_images , figsize=(30, 10))

    # Adjust the spacing between subplots
    fig.subplots_adjust(hspace=0.5)

    # Set a title for the entire figure
    fig.suptitle("Original Data (Top Row) vs. Poisoned Dataset (Bottom Row)", fontsize=16)


    # Loop through the images and display them side by side
    for i in range(num_images):
        # Convert the tensors to NumPy arrays
        original_np = original_images[i].cpu().numpy()
        poisoned_np = poisoned_images[i].cpu().numpy()

        # Display the original image on the top row
        original_np1 = original_np.copy()
        axes[0, i].imshow(np.transpose(original_np1, (1, 2, 0)))
        axes[0, i].set_title(original_labels[i])
        axes[0, i].axis('off')

        # Display the poisoned image on the bottom row
        poisoned_np1 = poisoned_np.copy()
        axes[1, i].imshow(np.transpose(poisoned_np1, (1, 2, 0)))
        axes[1, i].set_title(poisoned_labels[i])
        axes[1, i].axis('off')

    # Save the figure before showing it
    plt.savefig('/local/data/afadhel_dg/Homework2/528Project/all_classes/original_vs_poisoned.png')

    # Show the figure
    plt.show()
#######running the main program
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit()


















import prediction_args
import torch
import numpy as np
from torchvision import models, transforms
import sys
import os
import base64
from PIL import Image
import warnings
import subprocess

# To ignore all warnings
warnings.filterwarnings("ignore")

##Print both ont terminal and write into the output file
class Tee:
    def __init__(self, file, *args):
        self.file = file
        self.tees = args

    def write(self, text):
        self.file.write(text)
        for tee in self.tees:
            tee.write(text)

    def flush(self):
        pass

def main():
    """
    Predictions are happening here!
    """
    output_file = "output.txt"  # Fixed output filename

    # Check if the output.txt file exists and delete it if it does
    if os.path.exists(output_file):
        os.remove(output_file)
    output_file = "output.txt"  # Fixed output filename

    # Create a file object to append print statements to the output file
    with open(output_file, 'a') as file:

        # Create Tee objects to duplicate output to both sys.stdout and the file
        tee = Tee(file, sys.stdout)

        # Redirect stdout to the Tee object
        sys.stdout = tee

        print("")

        print("One moment please, Prediction is going on...!")
        parser = prediction_args.prediction_args()
        args = parser.parse_args()

        # CPU OR GPU?
        device = torch.device("cpu")
        if args.gpu:
            device = torch.device("cuda")

        model = load_the_model(device, args.model)

        top_probs, actual_names = predict(args.path_to_image, model, args.top_k, device)

        print("")

        print("Current image: {}".format(args.path_to_image))
        print("Model: {}".format(args.model))
        print("Device: {}".format(device))
        print(" ")

        print(" ")

        print("=======================================================")
        print("OUR PREDICTION:")
        print("The dog is {} ".format(actual_names[0]))
        print("=======================================================")

        print(" ")


        print("Top possibilities are: ")
        for i in range(len(top_probs)):
            print("The dog is {} with probability {:.2f}".format(actual_names[i], top_probs[i]))



def load_the_model(device, model_name):

    dict_model = torch.load(model_name)
    model = models.__dict__[dict_model["arch"]](pretrained=True)
    model.classifier = dict_model["classifier"]
    model.class_to_idx = dict_model["class_to_idx"]
    model.load_state_dict(dict_model["state_dict"])
    model.to(device)




    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    min_dimension = 0

    if image.size[1] < image.size[0]:
        min_dimension = 1

    if min_dimension == 0:
        image.thumbnail((256, image.size[1]))
    else:
        image.thumbnail((image.size[0], 256))


    box = ((image.width-224)/2, (image.height-224)/2, (image.width-224)/2 + 224, (image.height-224)/2 + 224 )
    image = image.crop(box = box)

    ##convert
    image = np.array(image)/255
    image = (image - np.array([0.485, 0.456, 0.406]) )/np.array([0.229, 0.224, 0.225])


    image = image.transpose( (2, 0, 1) )


    return image


def predict(image_path, model, topk, device):

    ''' Predict the class (or classes) of an image using a trained deep learning model.'''
    im = Image.open(image_path)
    numpy_im = process_image(im)



    if device.type == "cuda":
        images = torch.from_numpy(numpy_im).type(torch.cuda.FloatTensor)
    else:
        images = torch.from_numpy(numpy_im).type(torch.FloatTensor)

#     print(images.shape)


    images.unsqueeze_(0)
#     print(images.shape)


    with torch.no_grad():
        model.eval()
        log_probs = model(images)
    probs = torch.exp(log_probs)
    top_probs, top_classes = probs.topk(topk)





    idx_into_class = {y: x for x, y in model.class_to_idx.items()}


    ########into lists
    top_probs = top_probs.cpu().numpy()
    top_probs = top_probs.tolist()[0]

    top_classes = top_classes.cpu().numpy()
    top_classes = top_classes.tolist()[0]




    actual_names = [idx_into_class[model_out_class].split("-", 1)[-1].strip() for model_out_class in top_classes]



    return top_probs,actual_names


if __name__== "__main__":
    main()



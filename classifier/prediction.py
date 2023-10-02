import prediction_args
import torch
import numpy as np
from PIL import Image
from torchvision import models, transforms
import sys
import os
import base64
from PIL import Image
import warnings
import subprocess
import json
import requests
from PIL import Image
from io import BytesIO
# To ignore all warnings
warnings.filterwarnings("ignore")



def classify(img_url):
    """
    Predictions are happening here!
    """

    #output_file_directory = "output_files"
    output_file = "output.json"  # Output JSON filename

    # Check if the output.json file exists and delete it if it does
    if os.path.exists(output_file):
        os.remove(output_file)


    # parser = prediction_args.prediction_args()
    # args = parser.parse_args()

    # CPU OR GPU?
    device = torch.device("cpu")
    # if args.gpu:
    #     device = torch.device("cuda")

    model = load_the_model(device, "./model.pth")





    # Directory where you want to save the images
    download_directory = "images"

    # Download the images and get their paths
    image_path = download_image(img_url, download_directory)
    #print("downloaded_image_paths",downloaded_image_paths)


    results = []  # List to store results for multiple images
    # Print the paths of the downloaded images
#     for image_path in downloaded_image_paths:
#
#
#         #print("image path:", image_path)
#
#         # Download image, preprocess, and classify
#         top_probs, actual_names = predict(image_path, model, args.top_k, device)
#         #print("actual_names", actual_names)
#
#         # Store results for this image
#         result = {
#             image_path: actual_names
#         }
#         results.append(result)
            #print("image path:", image_path)

    # Download image, preprocess, and classify
    top_probs, actual_names = predict(image_path, model, 1, device)
    print("actual_names", actual_names)

    # Store results for this image
    result = {
        image_path: actual_names
    }
    results.append(result)


    # Save results as JSON
    with open(output_file, 'w') as json_file:
        json.dump(results, json_file, indent=4)

    print("Results saved to", output_file)

    return actual_names[0]





def download_image(url, directory):
    #image_paths = []
    try:
        # Create the directory if it doesn't exist
        os.makedirs(directory, exist_ok=True)

        # Get the image content from the URL
        response = requests.get(url)

        print(f"Downloading image from {url}, Status Code: {response.status_code}")

        if response.status_code == 200:
            # Extract the filename from the URL
            filename = os.path.join(directory, os.path.basename(url))

            # Save the image to the specified directory
            with open(filename, 'wb') as file:
                file.write(response.content)
            #image_paths.append(filename)

            return filename

            print(f"Image downloaded and saved as {filename}")
        else:
            print(f"Failed to download image. HTTP status code: {response.status_code}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    return None


def download_images_and_get_paths(urls, directory):
    all_image_paths = []
    for url in urls:
        image_path = download_image(url, directory)
        if image_path:
            all_image_paths.append(image_path)  # Append the path of the downloaded image to the


    return all_image_paths






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

    #Predict the class (or classes) of an image using a trained deep learning model.
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





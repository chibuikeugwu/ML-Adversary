from io import BytesIO
from classifier.prediction import classify, load_the_model, process_image
import torch
import torchvision.transforms as transforms
from PIL import Image
import requests

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def extract_breed_name(label):
    return label.split('-', 1)[1] if '-' in label else label


def classify_and_attack(img_url):
    model = load_the_model(torch.device("cpu"), "./model.pth")

    original_breed = classify(img_url)
    response = requests.get(img_url)
    img = Image.open(BytesIO(response.content))
    img_tensor = preprocess(img)
    img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

    # enable gradients for the image
    img_tensor.requires_grad = True

    # original Classification
    output = model(img_tensor)
    original_label = output.max(1, keepdim=True)[1].item()

    # FGSM Attack
    loss = torch.nn.functional.cross_entropy(output, torch.tensor([original_label]))
    model.zero_grad()
    loss.backward()

    # check if gradients are computed
    if img_tensor.grad is None:
        raise ValueError("Gradients not computed. Ensure that `requires_grad` is set to True.")

    epsilon = 0.01
    data_grad = img_tensor.grad.data
    perturbed_image = img_tensor + epsilon * data_grad.sign()
    perturbed_image = torch.clamp(perturbed_image, 0, 1)

    # classification of adversarial example
    output = model(perturbed_image)
    adversarial_label = output.max(1, keepdim=True)[1].item()

    idx_to_class = {v: k for k, v in model.class_to_idx.items()}

    # original_breed = idx_to_class[original_label]
    adversarial_breed = idx_to_class[adversarial_label]

    return original_breed, extract_breed_name(adversarial_breed)

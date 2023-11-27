import io
import torch
from torchvision import transforms
import numpy as np
from PIL import ImageFilter, Image


def random_resizing(image):
    resize_transform = transforms.Resize(np.random.randint(200, 300))
    image = resize_transform(image)
    return image


def random_cropping(image):
    crop_transform = transforms.RandomCrop(200)
    image = crop_transform(image)
    return image


#  Increasing the noise level may improve robustness but could also impact classification accuracy.
def add_random_noise(image_tensor, noise_level=0.05):
    noise = torch.randn(image_tensor.size()) * noise_level
    noisy_image = image_tensor + noise
    return torch.clamp(noisy_image, 0, 1)


# A higher radius increases the blur, potentially disrupting adversarial noise more, but it can also reduce the
# image's clarity for the classifier.
def apply_gaussian_blur(image, radius=2):
    return image.filter(ImageFilter.GaussianBlur(radius))


# Lower quality might be more effective in disrupting adversarial perturbations, but it can also degrade the image more.
def jpeg_compression(image, quality=75):
    # convert to bytes
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)

    # load the compressed image
    compressed_image = Image.open(buffer)
    return compressed_image

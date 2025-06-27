import torch
import torchvision.transforms as transforms
import os
import argparse
from PIL import Image
from catornot_classifier import CatornotClassifier


def main():
    CWD = os.getcwd()
    MODEL_WEIGHTS_PATH = os.path.join(CWD, 'catornot_model_weights.pth')

    parser = argparse.ArgumentParser(description='Predicts if an image is a cat or not')
    parser.add_argument('img_path', type=str, help='path to image')
    args = parser.parse_args()

    IMG_PATH = args.img_path

    if not os.path.exists(MODEL_WEIGHTS_PATH):
        raise FileNotFoundError(f'The weights for the model do not exit: {MODEL_WEIGHTS_PATH}\nTry running train.py to create model weights')
    if not os.path.exists(IMG_PATH):
        raise FileNotFoundError(f'Invalid image path: {IMG_PATH}')
    if not os.access(IMG_PATH, os.R_OK):
        raise PermissionError(f'Read permission for image denied: {IMG_PATH}')

    # load image as tensor
    transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    image = Image.open(IMG_PATH).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    # evaluate model
    model = CatornotClassifier(2)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH))
    model.eval()

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    probabilities = probabilities.cpu().squeeze().tolist()

    if probabilities[0] >= probabilities[1]:
        print(f'this image is {probabilities[0]*100:.3f}% cat and {probabilities[1]*100:.3f}% not')
    if probabilities[1] >= probabilities[0]:
        print(f'this image is {probabilities[1]*100:.3f}% not cat and {probabilities[0]*100:.3f}% cat')



if __name__ == '__main__':
    main()
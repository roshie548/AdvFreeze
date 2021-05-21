import sys
import pathlib
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from efficientnet_pytorch import EfficientNet
from torch import nn


def main():
    # Load the classes
    data_dir = pathlib.Path('./data/tiny-imagenet-200/train/')
    CLASSES = sorted([item.name for item in data_dir.glob('*')])
    im_height, im_width = 64, 64

    ckpt = torch.load('checkpoint_base_model_novel.pth.tar') # CHANGE THIS

    model = EfficientNet.from_pretrained('efficientnet-b0')
    num_ftrs = model._fc.in_features
    model._fc = nn.Linear(num_ftrs, 200)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    model.to(device)


    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    data_transforms = transforms.Compose([
       transforms.Resize(224),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Loop through the CSV file and make a prediction for each line
    with open('eval_classified.csv', 'w') as eval_output_file:  # Open the evaluation CSV file for writing
        for line in pathlib.Path(sys.argv[1]).open():  # Open the input CSV file for reading
            image_id, image_path, image_height, image_width, image_channels = line.strip().split(
                ',')  # Extract CSV info

            print(image_id, image_path, image_height, image_width, image_channels)
            with open(image_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
            img = data_transforms(img)[None, :]
            img = img.to(device)
            outputs = model(img)
            _, predicted = outputs.max(1)

            # Write the prediction to the output file
            eval_output_file.write('{},{}\n'.format(image_id, CLASSES[predicted]))


if __name__ == '__main__':
    main()

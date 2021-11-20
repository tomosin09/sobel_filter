import cv2
import torch
import numpy as np

from model import SobelNet

if __name__ == '__main__':
    model = SobelNet()
    image = cv2.imread('image.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = (image / 255.0)
    new_img = np.zeros((1, 1, image.shape[0], image.shape[1]))
    new_img[0, 0, :, :] = image
    out = model(torch.from_numpy(new_img.astype(np.float32)))
    img = out[0, :, :, :].permute(1, 2, 0).numpy()*255.0
    img = img.astype(np.uint8)
    cv2.imwrite('result.jpg', img)

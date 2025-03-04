import cv2
import numpy as np

if __name__ == '__main__':
    name = 'volume_out'
    name2 = 'volume2_out'
    img = cv2.imread(f'../{name}.png')
    img2 = cv2.imread(f'../{name2}.png')

    new_img = np.hstack([img, img2])

    cv2.imwrite(f'../{name}_out.png', new_img)

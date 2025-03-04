import cv2
import numpy as np

if __name__ == '__main__':
    name = 'pim'
    img = cv2.imread(f'../{name}.png')

    # cv2.rectangle(img, (549, 150), (612, 213), color=(0, 0, 255), thickness=2)
    # cv2.putText(img, '88.796 km/h', (618, 213), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255), thickness=2)

    # cv2.line(img, (339, 215), (1060, 215), color=(0, 0, 255), thickness=2)
    # cv2.putText(img, '2', (1070, 215), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255), thickness=2)

    # cv2.polylines(img, np.array([[[1171, 250], [434, 541], [596, 539], [1231, 250]]]), isClosed=True, color=(0, 0, 255), thickness=2)
    # img = img[0: img.shape[0] // 2, 0: img.shape[1] // 2, :]

    # cv2.polylines(img, np.array([[[500, 1180], [0, 1620], [0, 2159], [3600, 2159], [2660, 1126]]]), isClosed=True, color=(0, 0, 255),
    #               thickness=2)
    # img = img[800:, : 3600, :]

    cv2.polylines(
        img,
        np.array([[
            # [515, 0], [309, 607], [772, 607], [771, 2]
            # [208, 155], [76, 332], [488, 332], [397, 153]
            [0, 0], [0, 144], [1919, 406], [1919, 0]
        ]]),
        isClosed=True,
        color=(0, 0, 255),
        thickness=2
    )

    # cv2.line(img, (768, 231), (559, 914), color=(0, 0, 255), thickness=2)
    # cv2.line(img, (1041, 232), (1176, 914), color=(0, 0, 255), thickness=2)

    cv2.imwrite(f'../{name}_out.png', img)

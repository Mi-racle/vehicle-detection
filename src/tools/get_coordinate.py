import argparse
import tkinter as tk

from PIL import Image, ImageTk


def on_click(event):
    x, y = event.x, event.y
    print(f"Clicked at: ({x}, {y})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # parser.add_argument('-s', type=str, default='D:/xxs-signs/vehicle-detection/resources/image.png')
    parser.add_argument('-s', type=str, default='D:/xxs-signs/vehicle-detection/resources/online.jpg')

    args = parser.parse_args()

    root = tk.Tk()
    root.title("图片点击坐标获取")

    image_path = args.s
    image = Image.open(image_path)
    # image = image.crop(box=(0, 250, 2800, 2160))
    photo = ImageTk.PhotoImage(image)

    canvas = tk.Canvas(root, width=image.width, height=image.height)
    canvas.pack()

    canvas.create_image(0, 0, anchor=tk.NW, image=photo)

    canvas.bind("<Button-1>", on_click)

    root.mainloop()

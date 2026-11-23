import os
import matplotlib.pyplot as plt

from PIL import Image

RAW_FOLDER = "Data/Raw"

for split in ["train", "test"]:
    split_folder = RAW_FOLDER + "/" + split
    for class_name in ["chihuahua", "muffin"]:

        class_folder = split_folder + "/" + class_name	
        file_names = os.listdir(class_folder)
        number_of_images = len(file_names)
        print(split, "-", class_name, ":", number_of_images)

        number_rgb = 0
        number_other = 0

        for file_name in os.listdir(class_folder):
            image_path = class_folder + "/" + file_name
            image = Image.open(image_path)
            if image.mode == "RGB":
                number_rgb += 1
            else:
                number_other += 1

        figure, axes = plt.subplots(2, 5, figsize=(12, 5))
        for column in range(5):
            image_path = class_folder + "/" + file_names[column]
            image = Image.open(image_path)

            axes[0][column].imshow(image)
            axes[0][column].set_title("chihuahua")
            axes[0][column].axis("off")
        plt.tight_layout()
        plt.savefig("samples.png")
        plt.show()

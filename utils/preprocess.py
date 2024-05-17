import os
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def process_images(input_folder, output_folder, threshold=20, crop_size=20):
    if not os.path.exists(os.path.join(output_folder, "A")):
        os.makedirs(os.path.join(output_folder, "A"))
    if not os.path.exists(os.path.join(output_folder, "B")):
        os.makedirs(os.path.join(output_folder, "B"))

    num_list = []
    for label in ["A", "B"]:
        for repeat in range(1, 4):
            repeat = str(repeat)
            folder_path = os.path.join(input_folder, label, repeat)
            tqdm_obj = tqdm(os.listdir(folder_path), desc=f"label: {label},  repeat: {repeat}")
            for img_name in tqdm_obj:
                img_path = os.path.join(folder_path, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                image = image.reshape(-1)
                num_list.append(image)
    num_list = np.concatenate(num_list)
    num_list = num_list[num_list < 100]
    plt.hist(num_list, bins=50)
    plt.show()
    print("set threshold to 20")

    num_list = [0, 0]
    for label in ["A", "B"]:
        for repeat in range(1, 4):
            repeat = str(repeat)
            folder_path = os.path.join(input_folder, label, repeat)
            tqdm_obj = tqdm(os.listdir(folder_path), desc=f"label: {label},  repeat: {repeat}")
            for img_name in tqdm_obj:
                img_path = os.path.join(folder_path, img_name)
                image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

                # find the black pixels
                _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
                black_pixel_coords = np.column_stack(np.where(binary_image == 255))
                assert len(black_pixel_coords) > 0, f"No black pixels found in {img_path}"
                
                # Calculate the mean position of the black pixels
                mean_y, mean_x = np.mean(black_pixel_coords, axis=0).astype(int)
                
                # Calculate the crop area
                start_x = max(mean_x - crop_size, 0)
                end_x = min(mean_x + crop_size, image.shape[1])
                start_y = max(mean_y - crop_size, 0)
                end_y = min(mean_y + crop_size, image.shape[0])
                
                # Crop the image
                cropped_image = image[start_y:end_y, start_x:end_x]

                # verify the cropped image
                _, binary_image = cv2.threshold(cropped_image, threshold, 255, cv2.THRESH_BINARY_INV)
                black_pixel_coords = np.column_stack(np.where(binary_image == 255))
                num_list.append(len(black_pixel_coords))
                num_list[1] += 1
                try:
                    assert len(black_pixel_coords) > 1
                    assert cropped_image.shape == (40, 40)
                except:
                    num_list[0] += 1
                    continue
                
                # Save the cropped image
                output_path = os.path.join(output_folder, label, img_name)
                cv2.imwrite(output_path, cropped_image)

    print(f"invalid_images: {num_list[0]}/{num_list[1]}")


if __name__ == "__main__":
    input_folder = "./data/plt_1"
    output_folder = "./data/plt_1_processed"
    process_images(input_folder, output_folder)

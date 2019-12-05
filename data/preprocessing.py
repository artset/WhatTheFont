import os
from scipy import misc
import numpy as np
from PIL import Image
import random
import json
import pickle
import cv2

# note to self: clean stuff like bodoni std bold such that images aren't .png.png

def resize_image(image):
    """ Input: Image Path
        Output: Image
        Resizes image to height of 105px. Maintains aspect ratio
    """
    base_height = 105
    img = image #Image.open(image_path)
    height_percent = (base_height/float(img.size[1]))
    wsize = int((float(img.size[0])*float(height_percent)))
    # print("Width", wsize)
    img = img.resize((wsize, base_height),Image.ANTIALIAS )
    return img

def generate_crop(img):
    """ Input: Image object
        Output: A list of images.
        Randomly generates 15 cropped images
    """
    num_vals = 15
    cropped_images = []
    width = len(np.array(img)[1])
    # 120 is 105 + 15; we need at least 15 random crops possible, thus the width must be greater than 120
    # in the condition when width < 120, we shoould find a way to edit the image rather than omitting it
    if width > 120:
        bounds = random.sample(range(0, width-105), 15)
        for i in range(num_vals):
            new_img = img.crop((bounds[i], 0, bounds[i] + 105, 105))
    #         new_img.save("crop" + str(i) + ".jpg", format='JPEG')
            cropped_images.append(new_img)
    return cropped_images

def preprocess(root_dir):
    """ Input: Root directory (string)
        Output: Dictionary where key is font name, value is a 3D

        array that contains a list of images of shape (number of images x 105 x 105)
        labels, which is a list of strings (ex. TimesNewRomanStd)
    """
    dictionary = {}
    image_data = []
    labels = []
    big_array = []
    index_labels = []
    count = 0
    for subdir in os.listdir(root_dir): # goes through all font folders
        # print(count)
        # count += 1
        subdir_path = root_dir + "/" + subdir
        font_name = subdir #subdir.split("-")[0]
        if font_name not in dictionary:
            dictionary[font_name] = []
            labels.append(font_name)
            print(font_name)

        for file in os.listdir(subdir_path): # goes through all sample images
            image_path = subdir_path + "/" + file
            image = alter_image(image_path)
            image = resize_image(image)
            cropped_images = generate_crop(image)
            for c in cropped_images:
                arr = np.array(c)
                dictionary[font_name].append(arr)
                big_array.append(arr)
                index_labels.append(count)
            # print("added image")
        count += 1
    return dictionary, index_labels, big_array

def alter_image(image_path):
    """ Function to apply all of the filters to a single image.
    """
    img = Image.open(image_path)
    img = np.array(img)
    # noise
    row, col = img.shape
    gauss = np.random.normal(0, 3, (row, col))
    gauss = gauss.reshape(row, col)
    noised_image = img + gauss


#     # blur
    blurred_image = cv2.GaussianBlur(noised_image, ksize = (9, 9), sigmaX = random.uniform(2.5, 3.5))


    # perspective transform
    rotatation_angle = [-20, -10, 0, 10, 20]
    translate_x = [-15, -10, 0, 10, 15]
    translate_y = [-15, -10, 0, 10, 15]
    angle = random.choice(rotatation_angle)
    angle = random.choice(rotatation_angle)
    angle = random.choice(rotatation_angle)
    tx = random.choice(translate_x)
    ty = random.choice(translate_y)

    rows, cols = img.shape
    M_translate = np.float32([[1,0,tx],[0,1,ty]])
    M_rotate = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)

    affined_image = cv2.warpAffine(blurred_image, M_translate, (cols, rows))
    affined_image = cv2.warpAffine(affined_image, M_rotate, (cols, rows))
    # perhaps we can pick the top right corner pixel color and fill the bg
    # but if its synthetic, then we can just make samples of White Text on Brack background


#     # shading
    affined_image = np.array(affined_image) * random.uniform(0.2, 1.5)
    final_image = np.clip(affined_image, 0, 255)

    final_image = Image.fromarray(final_image)
    return final_image
    # final_image = final_image.convert("L")
    # final_image.save("test1.png", format='PNG')

def get_train():
    print("Running preprocessing...")
    #root_dir = 'C:/Users/katsa/Documents/cs/cs1470/real_images/VFR_real_test' #Katherine's file path
#     root_dir =  'C:/Users/kimur/Documents/homework/cs1470/VFR_real_test' #Minna's file path
    root_dir = './syn_train_one_font'

    cropped_images, font_labels, big_array = preprocess(root_dir)

    # single_file = 'C:/Users/kimur/Documents/homework/cs1470/VFR_real_test/ACaslonPro-Bold/ACaslonPro-Bold1276.png'
    # alter_image(single_file)

    #     print(cropped_images["ACaslonPro-Bold"])
    #     print()
    #     print(cropped_images["ACaslonPro-Italic"])
    return cropped_images, big_array, font_labels
    # with open('syn_train_fonts_1.pkl', 'wb') as output:
    #     pickle.dump(cropped_images, output)
    #
    # with open('syn_train_labels_1.pkl', 'wb') as output:
    #     pickle.dump(font_labels, output)
    print("Finished preprocessing.")

def get_test():
    print("Running preprocessing...")

    root_dir = './real_test_sample'

    cropped_images, font_labels, big_array = preprocess(root_dir)

    return cropped_images, big_array, font_labels
    print("done w test")


# if __name__ == "__main__":
#     main()

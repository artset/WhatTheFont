import os
from scipy import misc
import numpy as np
from PIL import Image, ImageFile
import random
import json
import pickle
import cv2
import imageio

ImageFile.LOAD_TRUNCATED_IMAGES = True

# note to self: clean stuff like bodoni std bold such that images aren't .png.png

def resize_image(image, image_dimension):
    """ Input: Image Path
        Output: Image
        Resizes image to height of 96px. Maintains aspect ratio
    """
    base_height = image_dimension
    img = image
    height_percent = (base_height/float(img.size[1]))
    wsize = int((float(img.size[0])*float(height_percent)))
    # print("Width", wsize)
    img = img.resize((wsize, base_height),Image.ANTIALIAS )
    return img

def generate_crop(img, image_dimension, num_vals):
    """ Input: Image object, the width and height of our image, number of cropped images
        Output: A list of images.
        Randomly generates 15 cropped images
    """
    cropped_images = []
    width = len(np.array(img)[1])
    # 111 is 96 + 15; we need at least 15 random crops possible, thus the width must be greater than 111
    # in the condition when width < 111, we shoould find a way to edit the image rather than omitting it
    if width > image_dimension + num_vals:
        bounds = random.sample(range(0, width-image_dimension), num_vals)
        for i in range(num_vals):
            new_img = img.crop((bounds[i], 0, bounds[i] + image_dimension, image_dimension))
            new_img = np.array(new_img) / 255.0
    #         new_img.save("crop" + str(i) + ".jpg", format='JPEG')
            cropped_images.append(new_img)
    return cropped_images

def alter_image(image_path):
    """ Function to apply all of the filters to a single image.
    """

    img = Image.open(image_path)
    img = img.convert('L')

    img = np.array(img)
    # print(img.shape)
    # noise
    row, col = img.shape
    gauss = np.random.normal(0, 3, (row, col))
    gauss = gauss.reshape(row, col)
    noised_image = img + gauss

    # blur
    blurred_image = cv2.GaussianBlur(noised_image, ksize = (3, 3), sigmaX = random.uniform(2.5, 3.5))

    # perspective transform
    rotatation_angle = [-4, -2, 0, 2, 4]
    translate_x = [-5, -3, 0, 3, 5]
    translate_y = [-5, -3, 0, 3, 5]
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
    #
    # shading
    affined_image = np.array(affined_image) * random.uniform(0.2, 1.5)
    final_image = np.clip(affined_image, 0, 255)

    final_image = Image.fromarray(final_image)
    final_image = final_image.convert('L')

    return final_image
    # final_image = final_image.convert("L")
    # final_image.save("test1.png", format='PNG')

def create_font_dictionary():
    path = "./fontlist.txt"

    f = open(path, 'r')
    content = f.read().split()
    dict = {}
    count = 1
    for line in content:
        print(line)
        dict[line] = count
        count += 1
    with open('font_dict.json', 'w') as fp:
        json.dump(dict, fp,  indent=4)

    #     #     pickle.dump(cropped_images, output)
def create_pickle(root_dir):
    """ Input: Root directory (string)
        Output: Creates 5 pickle files to use for our model.
        1) Train inputs for SCAE
        2) Train input & labels for DeepFont model
        3) Test input & labels for DeepFont Model
    """
    scae_inputs = []
    train_inputs = []
    train_labels = []
    test_inputs = []
    test_labels = []


    with open('font_dict.json') as json_file:
        font_dict = json.load(json_file)

    for subdir in os.listdir(root_dir): # goes through all font folders
        subdir_path = root_dir + "/" + subdir
        font_name = subdir

        # here, we have to split up our files into the three pixels
        file_count = 0
        for file in os.listdir(subdir_path): # goes through all sample images

            image_path = subdir_path + "/" + file
            image = alter_image(image_path)
            image = resize_image(image, 96)

            cropped_images = generate_crop(image, 96, 15)

            if file_count < 100:
                for c in cropped_images:
                    scae_inputs.append(c)

            elif file_count < 200:
                for c in cropped_images:
                    test_inputs.append(c)
                test_labels.append(font_dict[font_name])
            else:
                for c in cropped_images:
                    train_inputs.append(c)
                train_labels.append(font_dict[font_name])

            file_count += 1

        with open('scae_synthetic_inputs.pkl', 'wb') as output:
            pickle.dump(scae_inputs, output)

        with open('train_inputs.pkl', 'wb') as output:
            pickle.dump(train_inputs, output)

        with open('test_inputs.pkl', 'wb') as output:
            pickle.dump(test_inputs, output)

        with open('train_labels.pkl', 'wb') as output:
            pickle.dump(train_labels, output)

        with open('test_labels.pkl', 'wb') as output:
            pickle.dump(test_labels, output)
        print("Finished preprocessing...")

def get_data(root):
    """
    Input: Root directory of Data
    Output: Arrays for
    1) Input images to SCAE
    2) Input train images & labels for DF Model
    3) Input test image & labels for DF Model

    This function is called in the model to open the pickle.
    """
    print("Opening pickled data...")


    print("Finished opening pickled data...")

def process_unlabeled_real(root_dir):
    """ Input: Root directory (string)
        Output: Creates 5 pickle files to use for our model.
        1) Train inputs for SCAE
        2) Train input & labels for DeepFont model
        3) Test input & labels for DeepFont Model
    """
    scae_inputs = []

    print(root_dir)
    count = 0

    # files = [f.path for f in os.scandir(root_dir) if f.name.endswith(".jpg") or f.name.endswith(".png")]

    for f in os.scandir(root_dir):
        if f.name.endswith(".jpeg") or f.name.endswith(".jpg") or f.name.endswith(".png"):
            image_path = f.path

            image = alter_image(image_path)
            image = resize_image(image, 96)

            # if count == 100:
            #     image.save("freshypoop.png", format='PNG')
            #     return

            if count % 2000 == 0:
                count_str = str(count)
                image.save("./imgs/" + count_str + ".png", "PNG")
                print( "Saved img with number: ", count)

                # print("---", count, "images processed---")

            cropped_images = generate_crop(image, 96, 15)

            for c in cropped_images:
                scae_inputs.append(c)

            count += 1

    with open('scae_real_inputs.pkl', 'wb') as output:
        pickle.dump(scae_inputs, output)


    # for file in os.listdir(root_dir): # goes through all font folders
    #     print(file)
    #     filename = os.fsdecode(file)
    #
    #     if filename.endswith(".png") or filename.endswith(".jpg"):
    #         image_path = root_dir + "/" + file
    #
    #
    #         print(image_path)
    #         image = alter_image(image_path)
    #
    #
    #         image = resize_image(image)
    #         cropped_images = generate_crop(image, 96, 15)
    #
    #         for c in cropped_images:
    #             scae_inputs.append(c)
    #
    #         if count % 2000:
    #             print("---", count, "images processed---")
    #         count += 1
    #
    #
    # with open('scae_real_inputs.pkl', 'wb') as output:
    #     pickle.dump(scae_inputs, output)
    return


def main():
    # our small sample test
    # create_pickle("real_test_sample")
    #
    # pickled = open('scae_inputs.pkl', 'rb')
    # array = pickle.load(pickled)
    # pickled.close()
    #
    # count = 0
    # for img in array:
    #     final_image = img.convert("L")
    #     image_file = "test_img/" +str(count) + "img.png"
    #     final_image.save(image_file, format='PNG')
    #     count += 1
    #
    # create_pickle("real_test_sample")
    print("Start processing!")
    process_unlabeled_real("./scrape-wtf-new")



if __name__ == "__main__":
    main()

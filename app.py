from keras.models import load_model
from PIL import Image, ImageOps #Install pillow instead of PIL
import numpy as np
#import cv2
import os

#Definir el directorio de la carpeta donde se encuentran las imagenes
#Si sale un error de unicode por el nombre de la dirreccion cambiar el \ por el /
input_images_path = "C:/Users/George Domìnguez/Desktop/embebidos"
#para obtener los nombres de los archivos que tenemos guardados en la carpeta
files_names=os.listdir(input_images_path)


# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model('keras_model.h5', compile=False)

# Load the labels
class_names = open('labels.txt', 'r').readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
#image = Image.open('<IMAGE_PATH>').convert('RGB')
#image = Image.open('americana.jpg')

#recorrer la lista
for file_name in files_names:
    if file_name.split(".")[-1] not in ["jpg", "png", "jpeg"]:
        continue        
    image_path = input_images_path + "/" + file_name
    image = Image.open(image_path)
    if image is None: 
        continue
    #resize the image to a 224x224 with the same strategy as in TM2:
    #resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    #turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    # Load the image into the array
    data[0] = normalized_image_array
    # run the inference
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    print(class_name)
    print(confidence_score)








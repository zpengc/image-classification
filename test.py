from tensorflow.keras import models
from tensorflow.keras.preprocessing import image
import numpy as np
import shutil
import os
import cv2
from config import SIZE
from config import model_path
from config import result_dir
from config import test_dir

# 测试时的输入图片路径
img_path = 'dataset/test/squirrel_726.jpg'
shutil.rmtree(result_dir)
os.mkdir(result_dir)

print('[INFO] 加载模型')
model = models.load_model(model_path)
img = image.load_img(img_path, target_size=(SIZE, SIZE))
# resize to [0,1]
img = image.img_to_array(img) / 255.0
# expand to 4D tensor
img = np.expand_dims(img, axis=0)
matrix = model.predict(img)
label = np.argmax(matrix, axis=1)

# iterate all test files
for i in os.listdir(test_dir):
    test_path = os.path.join(test_dir, i)
    img = image.load_img(test_path, target_size=(SIZE, SIZE))
    img = image.img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    test_label = np.argmax(predictions, axis=1)
    if test_label == label:
        print(predictions)
        print("测试集标签为", test_label[0])
        content = cv2.imread(test_path)
        cv2.imshow("output image during test", content)
        cv2.waitKey(0)
        shutil.copy(test_path, result_dir)

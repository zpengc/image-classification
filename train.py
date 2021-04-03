from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import *
from tensorflow.keras import backend
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from config import train_dir
from config import test_dir
from config import val_dir
from config import batch_size
from config import model_path
from tensorflow.keras.applications import resnet50

# rescale pre-processing so that the target values range from 0 to 1
# augment them via random transformations, so that our model would never see twice the exact same picture
train_pic_gen = ImageDataGenerator(rescale=1. / 255, rotation_range=20, horizontal_flip=True)
val_pic_gen = ImageDataGenerator(rescale=1. / 255)
test_pic_gen = ImageDataGenerator(rescale=1. / 255)

# this is a generator that will read pictures found in subfolers of train_dir and
# val_dir, and indefinitely generate batches of augmented image data
train_flow = train_pic_gen.flow_from_directory(train_dir, target_size=(299, 299), batch_size=batch_size,
                                               class_mode='categorical')
val_flow = val_pic_gen.flow_from_directory(val_dir, target_size=(299, 299),
                                           batch_size=batch_size, class_mode='categorical')

# the number of batch during each epoch
train_batches = train_flow.samples / batch_size
val_batches = val_flow.samples / batch_size

# backbone model
base_model = resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(299, 299, 3))
x = base_model.output
# reduces the data significantly and prepares the model for the final classification layer
x = GlobalAveragePooling2D()(x)
# reduce overfitting by reducing 70% of units to zero and multiplying the rest by 1/1-70%
x = Dropout(0.7)(x)
# five animal classes
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# assigned val_loss to be monitored, if it lowers down we will save it.
callback_list = [
    keras.callbacks.ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')]

print("[INFO] 编译模型")
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.0001), metrics=['accuracy'])

print("[INFO] 模型概述")
print(model.summary())

print("[INFO] 训练模型")
# fit_generator for training a model using Python data generators
history = model.fit_generator(train_flow, steps_per_epoch=train_batches, epochs=30,
                              verbose=1, validation_data=val_flow,
                              validation_steps=val_batches, callbacks=callback_list)

print("[INFO] 保存模型")
model.save(model_path)

print("[INFO] 绘制图像")
his = history.history
print('history字典', his)

acc = his['accuracy']
val_acc = his['val_accuracy']
# blue lines
plt.plot(range(len(acc)), acc, 'b')
# red lines
plt.plot(range(len(acc)), val_acc, 'r')
plt.legend(['train', 'validation'], loc='upper left')
plt.title('Training and validation accuracy')

print("[INFO] 保存图像")
plt.savefig("train_and_validation_images.jpg")

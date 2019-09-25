# Solution to the Assessment 2:
import pickle
import glob
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import random
import pickle

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# Fetch all the files from the image folder
files = glob.glob('images/**')
dictval={}
i = 0

# Iterate over every file and try to save data to the dictval
for file in files:
    if "batches.meta" in file:
        # batches.meta contains the data for the label names 
        # and size of the batch
        with open(file,'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
    else:
        with open(file, 'rb') as fo:
            temp = pickle.load(fo, encoding='bytes')
            if i == 0:
                dictval['data']= list(temp[b'data'])
                dictval['labels']= list(temp[b'labels'])
            else:
                dictval['data'] = dictval['data'] + list(temp[b'data'])
                dictval['labels'] = dictval['labels'] + list(temp[b'labels'])
            i+=1


# Convert the bytes to the normal string
labels = [x.decode('utf-8') for x in data[b'label_names']] 
alldata = dictval['data']
alldatalabels = dictval['labels']
trainingdata = []
def create_training_data():
    def reshapedata(imdata, imlabel):
        print(len(imdata))
        for i  in range(0,len(imdata)):
            temp = imdata[i]
            img = np.reshape(temp, (3, 32,32)).T
            img = Image.fromarray(img, 'RGB')
            img = img.rotate(270)
            img  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            img = cv2.resize(img, (32,32))
            # plt.imshow(img)
            # plt.show()
            if type(imlabel[i]) != type(2):
                continue
            class_num = int(imlabel[i])
            temp  = [img, class_num]
            trainingdata.append(temp)
    reshapedata(alldata, alldatalabels)
create_training_data()



for sample in trainingdata[:10]:
    print(sample[1])

X = []
Y = []
for features, label in trainingdata:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1,32, 32, 1)



# pickle_out = open("X.pickle", "wb")
# pickle.dump(X, pickle_out)
# pickle_out.close()

# pickle_out = open("Y.pickle", "wb")
# pickle.dump(Y, pickle_out)
# pickle_out.close()


# pickle_in = open("X.pickle", "rb")
# X = pickle.load(pickle_in)
# pickle_in = open("Y.pickle", "rb")
# Y = pickle.load(pickle_in)



X = pickle.load(open("X.pickle", 'rb'))
Y = pickle.load(open("Y.pickle", 'rb'))

X = X/255.0


x_train = X[:30000]
y_train = Y[:30000]
x_test = X[30000:40000]
y_test = Y[30000:40000]
x_val = X[40000:50000]
y_val = Y[40000:50000]


# Model initialization 
model = Sequential()
model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

# model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
# model.add(Activation("relu"))
# model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(10))
model.add(Activation("softmax"))


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
model.fit(x_train, y_train,batch_size=32, validation_data=(x_val, y_val), epochs = 3)
 
test_loss, test_acc = model.evaluate(x_test, y_test)

model.save('ourmodel.h5')
saved_model = tf.keras.models.load_model('ourmodel.h5')
predictions = saved_model.predict(x_test)

print(np.argmax(predictions[0]))
print(y_test[0])


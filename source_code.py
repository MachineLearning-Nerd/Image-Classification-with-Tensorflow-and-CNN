# to add a new cell, type '#%%'
# to add a new markdown cell, type '#%% [markdown]'
import pickle
import glob
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import matplotlib.pyplot as plt
from pil import image
import cv2
import random
import pickle
from tensorflow.keras.models import sequential
from tensorflow.keras.layers import dense, dropout, activation, flatten, conv2d, maxpooling2d
import warnings
warnings.filterwarnings("ignore")


# fetch all the files from the image folder
files = glob.glob('images/**')
print(files)
dictval={}
i = 0

# iterate over every file and try to save data to the dictval
for file in files:
    print(file)
    if "batches.meta" in file:
        # batches.meta contains the data for the label names 
        # and size of the batch
        with open(file,'rb') as fo:
            data = pickle.load(fo, encoding='bytes')
            print(data)
    else:
        with open(file, 'rb') as fo:
            temp = pickle.load(fo, encoding='bytes')
            #print(temp)
            if i == 0:
                dictval['data']= list(temp[b'data'])
                dictval['labels']= list(temp[b'labels'])
            else:
                dictval['data'] = dictval['data'] + list(temp[b'data'])
                dictval['labels'] = dictval['labels'] + list(temp[b'labels'])
            i+=1


# convert the bytes to the normal string
print(data[b'label_names'])
labels = [x.decode('utf-8') for x in data[b'label_names']] 
print(labels)


alldata = dictval['data']
alldatalabels = dictval['labels']
trainingdata = []
def create_training_data():
    def reshapedata(imdata, imlabel):
        print(len(imdata))
        for i  in range(0,len(imdata)):
#         for i  in range(1,5):
            # this data is the in the format of 3072 array elements
            temp = imdata[i]
            #print(temp)
            #print(len(temp))
            
            # to reshape the data
            img = np.reshape(temp, (3, 32,32)).t
            #print(img.shape)
            
            # convert the numpy array into the rgb format
            img = image.fromarray(img, 'rgb')
            
            # to see the image without correct orientation
            #plt.imshow(img)
            #plt.show()
            
            # img is in rotated format, so we need to rotate the image
            # to get the original orientation
            img = img.rotate(270)
            
            # here gray conversion is done: in our application color images are not need because we 
            # can get the same information in the gray image. 
            # benefit of using gray image : it will reduce the calculations by 3(rgb have 3 channels)
            img  = cv2.cvtcolor(np.array(img), cv2.color_rgb2bgr)
            img  = cv2.cvtcolor(np.array(img), cv2.color_bgr2rgb)
            
            # just to make sure that every image is 32*32
#             img = cv2.resize(img, (32,32))
            
            # to see the image in the correct orientation
#             plt.imshow(img)
#             plt.show()
            
            
            # just to verify that every label is int
            if type(imlabel[i]) != type(2):
                continue
            class_num = imlabel[i]
            
#             print(labels[class_num])
#             break
            # to create training data: 
            # i have appened the image data and the label
            # temp[0]: this is image
            # temp[1]: this is label
            temp  = [img, class_num]
            trainingdata.append(temp)
            #break
    reshapedata(alldata, alldatalabels)
create_training_data()
print("This is the dimension of the data")
print(len(trainingdata))


# This is to make the data shuffled randomly
import random
random.shuffle(trainingdata)


# This is just to check whether every data is 
# append correctly or not.
for sample in trainingdata[:10]:
    print("label = %d" %sample[1])

X = []
Y = []
# This is to store all the images in X
# and all the labels in Y
for features, label in trainingdata:
    X.append(features)
    Y.append(label)
# To reshape informat of tensorflow
X = np.array(X).reshape(-1,32, 32, 3)
print(X.shape)
# plt.imshow(X[4,:,:,:])
# plt.show()


#X = pickle.load(open("X.pickle", 'rb'))
#Y = pickle.load(open("Y.pickle", 'rb'))

# to normalize the data. 
X = X.astype('float32')
X /= 255.0
plt.imshow(X[4,:,:,:])
plt.show()
# print(X[8,:,:,:])

# 60% Training data
x_train = X[:30000]
y_train = Y[:30000]

# 20% Testing data
x_test = X[30000:40000]
y_test = Y[30000:40000]

# 20% Validation data
x_val = X[40000:50000]
y_val = Y[40000:50000]


# Model Initialization
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(Conv2D(128, (5, 5), activation='relu'))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(10, activation='softmax'))


print(model.summary())


# Model compilation is done
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# Here we are going to train the model
model.fit(x_train, y_train,batch_size=200, validation_data=(x_val, y_val), epochs = 20)
 


# To find the test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)
print(test_acc)


model.save('ourmodel.h5')
saved_model = tf.keras.models.load_model('ourmodel.h5')

y_pred = saved_model.predict(x_test)
print(y_pred[8])

predval = 105
count = 0
for i in range(len(y_pred)):
    if np.argmax(y_pred[i]) == y_test[i]:
        count +=1
accuracy = count/len(y_pred)
print(accuracy)
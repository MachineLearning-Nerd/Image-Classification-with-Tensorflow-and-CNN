# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%%
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
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D


#%%
# Fetch all the files from the image folder
files = glob.glob('images/**')
print(files)
dictval={}
i = 0

# Iterate over every file and try to save data to the dictval
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


#%%
# Convert the bytes to the normal string
print(data[b'label_names'])
labels = [x.decode('utf-8') for x in data[b'label_names']] 
print(labels)


#%%
alldata = dictval['data']
alldatalabels = dictval['labels']
trainingdata = []
def create_training_data():
    def reshapedata(imdata, imlabel):
        print(len(imdata))
        for i  in range(0,len(imdata)):
        #for i  in range(1,5):
            # This data is the in the format of 3072 array elements
            temp = imdata[i]
            #print(temp)
            #print(len(temp))
            
            # To reshape the data
            img = np.reshape(temp, (3, 32,32)).T
            #print(img.shape)
            
            # Convert the numpy array into the RGB format
            img = Image.fromarray(img, 'RGB')
            
            # To see the image without correct orientation
            #plt.imshow(img)
            #plt.show()
            
            # img is in rotated format, so we need to rotate the image
            # to get the original orientation
            img = img.rotate(270)
            
            # Here gray conversion is done: in our application color images are not need because we 
            # can get the same information in the gray image. 
            # Benefit of using gray image : It will reduce the calculations by 3(RGB have 3 channels)
            img  = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            
            # Just to make sure that every image is 32*32
            img = cv2.resize(img, (32,32))
            
            # To see the image in the correct orientation
            #plt.imshow(img)
            #plt.show()
            
            # Just to verify that every label is int
            if type(imlabel[i]) != type(2):
                continue
                
            # Stored the labels in another variable
            class_num = imlabel[i]
            
            #print(labels[class_num])
            
            # To create training data: 
            # I have appened the image data and the label
            # temp[0]: This is image
            # temp[1]: This is label
            temp  = [img, class_num]
            trainingdata.append(temp)
            #break
    reshapedata(alldata, alldatalabels)
create_training_data()


#%%
# This is to make the data shuffled randomly
import random
random.shuffle(trainingdata)


#%%
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
X = np.array(X).reshape(-1,32, 32, 1)
print(X.shape)


#%%
#X = pickle.load(open("X.pickle", 'rb'))
#Y = pickle.load(open("Y.pickle", 'rb'))

# to normalize the data. 
X = X/255.0

# 60% Training data
x_train = X[:30000]
y_train = Y[:30000]

# 20% Testing data
x_test = X[30000:40000]
y_test = Y[30000:40000]

# 20% Validation data
x_val = X[40000:50000]
y_val = Y[40000:50000]


#%%
# Model initialization 
model = Sequential()
model.add(Conv2D(256, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))

model.add(Conv2D(128, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))

model.add(Conv2D(32,(3,3), input_shape = X.shape[1:]))
model.add(Activation("relu"))

model.add(Flatten())
model.add(Dense(120))
model.add(Activation("relu"))
model.add(Dense(30))
model.add(Activation("relu"))
model.add(Dense(20))
model.add(Activation("relu"))
# Output layers
model.add(Dense(10))
# Here we will get the outputs in probability 
model.add(Activation("softmax"))


#%%
# Model compilation is done
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',metrics=['accuracy'])
# Here we are going to train the model
model.fit(x_train, y_train,batch_size=100, validation_data=(x_val, y_val), epochs = 10)
 


#%%
# To find the test accuracy
test_loss, test_acc = model.evaluate(x_test, y_test)


#%%
model.save('ourmodel.h5')
saved_model = tf.keras.models.load_model('ourmodel.h5')


#%%
y_pred = saved_model.predict(x_test)


#%%
print(y_pred[8])


#%%
predval = 105
count = 0
for i in range(len(y_pred)):
    if np.argmax(y_pred[i]) == y_test[i]:
        count +=1
accuracy = count/len(y_pred)
print(accuracy)
# print(np.argmax(y_pred[predval]))
# print(y_test[predval])
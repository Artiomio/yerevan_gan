#!/usr/bin/env python
# coding: utf-8

# In[1]:



import gc
import os
import time
import numpy as np
import cv2

from tqdm.notebook import tqdm
#from tqdm import tqdm

from sklearn.model_selection import train_test_split

import keras
from keras.models import Sequential
from keras.layers import *
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

from keras import initializers
import sys
sys.path.append('/home/art/Programming/artlibs')
from artiom_convenience_functions import * 
from videorecorder import save_to_video, execute_at_exit



# In[2]:




# In[3]:


N_DIM = 64

# faces = np.concatenate([np.fromfile(f"girls_img_{N_DIM}x{N_DIM}.bin", dtype="uint8").reshape(-1, N_DIM, N_DIM) / 255,

#                         np.fromfile(f"boys_img_{N_DIM}x{N_DIM}.bin", dtype="uint8").reshape(-1, N_DIM, N_DIM) / 255])


# In[4]:


faces = np.fromfile(f"/home/art/datasets/img_64x64_rgb_100k_verticallized.bin", 
                    dtype="uint8",
                   count=10_000 * N_DIM ** 2 * 3).reshape(-1, N_DIM, N_DIM, 3) / 255

faces = faces.astype("float32")

#plt.imshow(faces[123])
#plt.show()
#exit()

# In[5]:


faces.shape


# # Face generator
# 

# In[79]:


inputs = keras.Input(shape=(512))
x = Dense(120, activation='sigmoid', kernel_regularizer='l2')(inputs)
#x = Reshape((10,12))(x)
#x = Dropout(0.2)(x)
x = BatchNormalization()(x)
x = Dense(1200, activation='tanh')(x)
#x = BatchNormalization()(x)
#x = Dense(2000, activation='tanh')(x)
#x = BatchNormalization()(x)

x = Flatten()(x)
#x = Dense(N_DIM**3 , activation='sigmoid', kernel_regularizer='l2')(x)
x = Dense(N_DIM * 2 , activation='sigmoid', kernel_regularizer='l2')(x)
x = Dropout(0.2)(x)
x = Dense(N_DIM**2*3, activation='sigmoid', kernel_regularizer='l2')(x)
print(x.shape)
x = Reshape((N_DIM, N_DIM, 3))(x)

#x = Reshape((N_DIM, N_DIM))(x)


face_gen = keras.Model(inputs=inputs, outputs=x, name="face_generator")

face_gen.compile(

    optimizer=keras.optimizers.Adam(1e-5),
    metrics=["accuracy"],
)


# In[ ]:





# 
# # policeman 

# In[80]:


shape=(N_DIM, N_DIM)


# In[125]:


inputs = keras.Input(shape=(N_DIM, N_DIM, 3))

x = Conv2D(100, (3, 3), padding="same")(inputs)
x = MaxPool2D((2, 2))(x)

x = Conv2D(308, (3, 3), padding="same")(x)
x = MaxPool2D((3, 3),   padding="same")(x)
x = Conv2D(100, (3, 3), padding="same")(x)
x = MaxPool2D((2, 2), padding="same")(x)


x = Flatten()(x)
x = Dense(500, activation='sigmoid', kernel_regularizer='l2')(x)


outputs = Dense(1, activation='sigmoid')(x)
policeman = keras.Model(inputs=inputs, outputs=outputs, name="policeman")


# In[126]:


policeman.compile(
    loss=keras.losses.BinaryCrossentropy(),

    optimizer=keras.optimizers.Adam(1e-5),
    metrics=["accuracy"],
)


# # Среда

# In[127]:


policeman.trainable = False
inputs = keras.Input(shape=(512))
x = face_gen(inputs)
outputs = policeman(x) 
env_model = keras.Model(inputs=inputs, outputs=outputs, name="face_generator")


# In[128]:


env_model.compile( loss="binary_crossentropy" 
)


# In[129]:


env_model.summary()


# In[130]:




from importlib import reload

from videorecorder import get_new_cv2_imshow
from videorecorder import save_to_video, execute_at_exit

reload(sys.modules['videorecorder'])
reload(sys.modules['cv2'])
my_imshow = get_new_cv2_imshow(cv2.imshow, video_frame_rate=1,
                                redraw_only_every_nth_frame=1,
                                #last_frame_meditation_time_sec=3,
                                #fading_out_time_sec=5)
                               )


# In[145]:


2


# In[ ]:


cv2.destroyAllWindows()
from importlib import reload

reload(sys.modules['videorecorder'])
reload(sys.modules['cv2'])
#from videorecorder import save_to_video, execute_at_exit
from artiom_convenience_functions import *

from videorecorder import get_new_cv2_imshow
my_imshow = get_new_cv2_imshow(cv2.imshow, video_frame_rate=1,
                                redraw_only_every_nth_frame=1,
                                #last_frame_meditation_time_sec=3,
                                #fading_out_time_sec=5)
                               )




key = None
policeman_epochs = 1
env_epochs = 10
i = 0
while key not in (27,):
    i += 1
    N = 100
    seeds = np.random.normal(size=(N, 512)).astype("float32").reshape(-1, 512)
    faces_drawn_from_seeds = face_gen.predict(seeds, verbose=0)

    
    
    best_ind = np.argmax(policeman.predict(faces_drawn_from_seeds, verbose=0).reshape(-1))
    illustration_img = uint8_normalized(cv2.cvtColor(
        
        fit_img_center(faces_drawn_from_seeds[best_ind].reshape(N_DIM, N_DIM, 3), width=500, height=500,
                               interpolation=cv2.INTER_NEAREST), cv2.COLOR_BGR2RGB))
    my_imshow("result", illustration_img)
    #send_to_artmonitor(illustration_img, secret="artgan")
#     save_to_video((faces_drawn_from_seeds[best_ind].reshape(N_DIM, N_DIM), width=500,
#                                interpolation=cv2.INTER_NEAREST))
    key = cv2.waitKey(1)

    indices = np.random.randint(0, len(faces), size=N)
    real_faces = faces[indices]
    #plt.imshow(real_faces[1])
    #plt.show()
    #exit()
    X = np.concatenate([faces_drawn_from_seeds, real_faces])
    Y = np.zeros(
    N * 2)

    Y[N:] = 1

    
    

    policeman.fit(X, Y, epochs=int(policeman_epochs), shuffle=1, verbose=0)
    caught = policeman.evaluate(X, Y, verbose=0)[1]
    print(f"[{i}] - {round(caught,3)}   {int(policeman_epochs)} / {int(env_epochs)}")





    #policeman
    N = 500
    seeds = np.random.normal(size=(N, 512)).astype("float32").reshape(-1, 512)

    ones = np.ones(len(seeds)).reshape(-1, 1)
    
    if caught > 0.9:
          env_epochs = policeman_epochs * 3 
    elif caught < 0.6:
        
        policeman_epochs = env_epochs * 1.5
        
        
    if env_epochs > 10 and policeman_epochs > 10:
        env_epochs /= 5
        policeman_epochs /= 5
    env_model.fit(seeds, ones, shuffle=1, epochs=int(env_epochs), verbose=0)



    if i % 10000 == 0:
        face_generator.save("yerevan_face_gan.h5")
    
execute_at_exit()


# In[ ]:


execute_at_exit()





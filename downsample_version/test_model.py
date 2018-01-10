
# coding: utf-8

# In[4]:


from UNet import *
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd


# In[5]:


raw_image = dd.io.load('sample_origonal_images.h5')
vector_field = dd.io.load('vector_fields.h5')


# In[6]:


n_imgs = len(raw_image)
n_input_channel = 1
n_vf_channel = 4
size = raw_image[0].shape[0]
raw_image = raw_image.reshape(n_imgs, n_input_channel, size, size)


# In[7]:


factor = 9


# In[8]:


down_inputs = raw_image[:,:, ::factor,::factor] / factor
down_vf = vector_field[:, :, ::factor, ::factor] / factor


# In[9]:


down_inputs_pad = np.zeros((n_imgs, n_input_channel, 96, 96))
down_vf_pad = np.zeros((n_imgs, n_vf_channel, 96, 96))


# In[10]:


down_inputs_pad[:,:,:94, :94] = down_inputs
down_vf_pad[:,:,:94, :94] = down_vf 


# In[11]:


size = down_inputs.shape[-1]


# ### lode pre-trained model

# In[14]:


model = torch.load('./model/unet.pkl')


# In[16]:


test_data = torch.from_numpy(down_inputs_pad[:-1].astype(np.float32))
test_data = Variable(test_data)


# In[17]:


prediction = model(test_data)


# ### save prediction result

# In[18]:


dd.io.save('./model/prediction_result.h5', prediction)


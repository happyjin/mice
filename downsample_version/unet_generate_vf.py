
# coding: utf-8

# In[1]:


from UNet import *
from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt
import deepdish as dd
import sys
import time
import os


# ### load data

# In[2]:
start_time = time.time()

raw_image = dd.io.load('sample_origonal_images.h5')
vector_field = dd.io.load('vector_fields.h5')



# In[4]:


n_imgs = len(raw_image)
n_input_channel = 1
n_vf_channel = 4
size = raw_image[0].shape[0]
raw_image = raw_image.reshape(n_imgs, n_input_channel, size, size)


# ### downsample both input images and vector fields

# In[5]:


factor = 9


# In[6]:


down_inputs = raw_image[:,:, ::factor,::factor] / factor


# In[7]:


down_vf = vector_field[:, :, ::factor, ::factor] / factor


# In[8]:



# ### zero padding in order to get tensor consistency

# In[10]:


down_inputs_pad = np.zeros((n_imgs, n_input_channel, 96, 96))
down_vf_pad = np.zeros((n_imgs, n_vf_channel, 96, 96))


# In[11]:


down_inputs_pad[:,:,:94, :94] = down_inputs
down_vf_pad[:,:,:94, :94] = down_vf


# In[12]:


down_inputs_pad.shape


# In[13]:


down_vf_pad.shape


# ### downsampling dataset visualization

# In[14]:


size = down_inputs.shape[-1]


# In[15]:




# ### set hyperparameters

# In[17]:


batch_size = 1
img_size = down_inputs.shape[-1]
lr = .0002
epoch = 1000


# ### transform to torch tensor

# In[18]:
#### loop for CV
folder_path = './model/'
counter = 1
vali_indeces_list = []
kf = KFold(n_splits=10)
x = np.arange(100) # using 100 out of 105 to do cross validation to verify the result
for train_indeces, validation_indeces in kf.split(x): 
    print('the %dth cross validation computing\n' %(counter))
    vali_indeces_list.append(validation_indeces)
    print('current valiadation indeces: %s' %(validation_indeces))

    tensor_x = torch.stack([torch.Tensor(i) for i in down_inputs_pad[train_indeces]])
    tensor_y = torch.stack([torch.Tensor(i) for i in down_vf_pad[train_indeces]])


    # ### create dataset
    my_dataset = data.TensorDataset(tensor_x, tensor_y)


    # In[20]:


    my_dataset.target_tensor.shape


    # In[21]:


    my_dataset.data_tensor.shape


    # ### input pipeline

    # In[22]:


    img_batch = data.DataLoader(my_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


    # ### initiate U-Net model generator

    # In[23]:


    #generator = UnetGenerator(1, 4, 64) # using CPU
    #generator = nn.DataParallel(UnetGenerator(1, 4, 64)).cuda() # using GPU
    generator = UnetGenerator(1, 4, 64).cuda() # using GPU


    # ### set loss function and optimizer

    # In[24]:


    recon_loss_func = nn.MSELoss()
    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)


    # ### training the model

    # In[25]:


    loss_list = []
    for i in range(epoch):
        for k, (img, vec_field) in enumerate(img_batch):
            gen_optimizer.zero_grad()
            x = Variable(img).cuda(0)
            y_ = Variable(vec_field).cuda(0)
            y = generator.forward(x)
            loss = recon_loss_func(y, y_)
            loss_list.append(loss.data.cpu().numpy())
            loss.backward()
            gen_optimizer.step()
        # display iteration and error
        sys.stdout.write("\riteration %d/%d and current error is %f" % (i+1, epoch, loss.data.cpu().numpy()))
        sys.stdout.flush()

    file_name = 'unet_gpu_cv' + str(counter) + '.pkl'
    torch.save(generator, os.path.join(folder_path, file_name))
    print("save model successfully!")
    loss_list = np.array(loss_list)
    file_name = 'unet_mse_loss_cv' + str(counter) + '.h5'
    dd.io.save(os.path.join(folder_path, file_name), loss_list)
    print("save loss successfully!")
    print("\n--- %s seconds ---" % (time.time() - start_time))
    counter += 1
    
# save 'validation_indeces'
file_name = 'validation_indeces.npy'
vali_indeces_list = np.array(vali_indeces_list)
np.save(os.path.join(folder_path, file_name), vali_indeces_list)
print("save cross_validation indeces successfully!")
print("\n--- total time: %s seconds ---" % (time.time() - start_time))
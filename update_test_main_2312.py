
# coding: utf-8

# In[1]:


from UNet import *
import numpy as np
import matplotlib.pyplot as plt


# set hyperparameters

# In[45]:


batch_size = 20
img_size = 128
lr = .0002
epoch = 20


# input pipeline

# In[46]:


img_dir = './maps/'
img_data = dset.ImageFolder(root=img_dir, transform=transforms.Compose([
                                                    transforms.Resize(size=img_size),
                                                    transforms.CenterCrop(size=(img_size, img_size*2)),
                                                    transforms.ToTensor(),
                                                    ]))
img_batch = data.DataLoader(img_data, batch_size=batch_size, shuffle=True, num_workers=2)


# initiate generator

# In[47]:


generator = nn.DataParallel(UnetGenerator(3,3,64), device_ids=[2]).cuda()
#generator = UnetGenerator(3,3,64)


# load pretrained model

# In[48]:


try:
    generator = torch.load('./model/unet.pkl')
    print("\n------model restored------\n")
except:
    print("\n------model not restored------\n")
    pass


# set loss function and optimizer

# In[49]:


recon_loss_func = nn.MSELoss()
gen_optimizer = torch.optim.Adam(generator.parameters(), lr=lr)


# training the model

# In[50]:


#ile = open('./unet_mse_loss', 'w')


# In[52]:

for i in range(epoch):
    for _, (image, label) in enumerate(img_batch):
        satel_image, map_image = torch.chunk(image, chunks=2, dim=3)
        
        gen_optimizer.zero_grad()
        
        x = Variable(satel_image).cuda(0)
        y_ = Variable(map_image).cuda(0)
        y = generator.forward(x)
        
        loss = recon_loss_func(y, y_)
        loss.backward()
        print(loss)

        gen_optimizer.step()

    #break



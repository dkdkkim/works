from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as scim
import random


split = 'TP'   ####### select split

crop_path = glob('/data/crops/train/'+split+'/*/*/*/*/*/*.npy')


# print crop_path

arr = np.arange(len(crop_path))
print len(crop_path)
np.random.shuffle(arr)
k=1
# for i in arr[0:500]:
for i in range(0,len(crop_path)):
    # if crop_path[i][-7:] == '201.npy' or crop_path[i][-7:] == '120.npy':   continue
    # if crop_path[i].split('/')[5] == 'DSBtrain':
    if crop_path[i].split('/')[-1][24:27] == '1.0' and crop_path[i].split('/')[-1][-10:] == '_0_012.npy':

    # print crop_path[i].split('/')[-1][24:27]
    # print crop_path[i].split('/')[-1][-10:]
    # raw_input("enter")
    # print crop_path[i].split('/')
    # raw_input("enter")
    # print '_'.join(crop_path[i].split('/')[3:7])+crop_path[i].split('/')[-1]



        im = np.load(crop_path[i])
        save_name = crop_path[i].split('/')[-1].split('.')[0]
        # z, y, x
        select_im = im[int(im.shape[0]/2)]
        # scim.imresize(select_im,(144,144),interp='nearest',mode=None)
        # scim.imsave('/home/dkkim/data/FN/%s.jpg'%'_'.join(crop_path[i].split('/')),select_im)
        plt.imshow(select_im, cmap='bone')
        plt.savefig('/home/dkkim/data/%s.jpg' %(split + '/' + '_'.join(crop_path[i].split('/')[3:7])+crop_path[i].split('/')[-1]))
        # scim.imsave('/data/figs/8001/%s.jpg'%'_'.join(crop_path[i].split('/')),select_im)
        # scim.imsave('/home/dkkim/data/FP/%s.jpg'%crop_path[i],select_im)

        print k, ' : success'
        k += 1





# im = np.load(crop_path[0])
#
# print im.shape
#
#
# print im[:,:,0].max()
# print im[:,:,0].min()


# plt.imshow(im[:,:,0], cmap='bone', clim=(im[:,:,0].min(),im[:,:,0].max()))


# plt.imshow(im[:,:,0], cmap='bone')
#
# plt.figure(1)
# plt.imshow(im[:,:,0], cmap='Greys')
#
# plt.figure(2)
# plt.imshow(im[:,:,0], cmap='gray')
#
# plt.figure(3)
# plt.imshow(im[:,:,0], cmap='bone')
#
# plt.show()

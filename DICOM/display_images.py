import load_dicom as ld
from glob import glob
import numpy as np

data_path = "/home/dkkim/CTs/115278/01-02-1999-NLST-LSS-57267/6617-0OPAPHMX8000C3383.212039.00.01.75-25145"
output_path = working_path = "/home/dkkim/works"
g = glob(data_path + '/*.dcm')

id=0

patient = ld.load_scan(data_path)
imgs = ld.get_pixels_hu(patient)
np.save(output_path + "fullimages_%d.npy" % id, imgs)

id = 0
imgs_to_process = np.load(output_path+'fullimages_{}.npy'.format(id))

ld.sample_stack(imgs_to_process,5,4,0,5,10)

# import matplotlib.pyplot as plt
#
# plt.imshow(imgs_to_process[0])
# plt.imshow(imgs_to_process[1])
# plt.imshow(imgs_to_process[2])
plt.show()


def load_scan(path):
    import dicom
    import os
    import numpy as np

    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key=lambda x: int(x.InstanceNumber))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)

    for s in slices:
        s.SliceThickness = slice_thickness

    return slices

def get_pixels_hu(scans):

    import numpy as np

    image = np.stack([s.pixel_array for s in scans])
    # Convert to int16 (from sometimes int16),
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 1
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0

    # Convert to Hounsfield units (HU)
    intercept = scans[0].RescaleIntercept
    slope = scans[0].RescaleSlope

    if slope != 1:
        image = slope * image.astype(np.float64)
        image = image.astype(np.int16)

    image += np.int16(intercept)

    return np.array(image, dtype=np.int16)

def sample_stack(stack, rows = 6, cols = 6, start_with = 10, show_every = 3,size = 6):

    import matplotlib.pyplot as plt

    # fig,ax = plt.subplots(rows,cols,figsize=[12,12])
    # for i in range(rows*cols):
    #     ind = start_with + i*show_every
    #     ax[int(i/rows),int(i%rows)].set_title('slice %d' %ind)
    #     ax[int(i/rows),int(i%rows)].imshow(stack[ind],cmap='gray')
    #     ax[int(i/rows),int(i%rows)].axis('off')
    # plt.show()

    fig, ax = plt.subplots(rows+1, cols+1, figsize=[size,size])
    for i in range(rows * cols):
        ind = start_with + i * show_every
        ax[int(i / rows), int(i % rows)].set_title('slice %d' % ind)
        ax[int(i / rows), int(i % rows)].imshow(stack[ind], cmap='gray')
        ax[int(i / rows), int(i % rows)].axis('off')

    print('index : ', ind, )

    plt.show()

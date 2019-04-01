import os
from glob import glob

train_crop = glob('/data/crops/train/*/*/*/*/*/*/*.npy')
val_crop = glob('/data/crops/valid/*/*/*/*/*/*/*.npy')

train_path_1 ='/home/dkkim/data/train/class1'
train_path_0 ='/home/dkkim/data/train/class0'

val_path_1 = '/home/dkkim/data/valid/class1'
val_path_0 = '/home/dkkim/data/valid/class0'

t0,t1,v0,v1 = 0,0,0,0;

# for path in train_crop:
#     if path.split('/')[-7] == 'TP' or path.split('/')[-7] == 'FN':
#         if path.split('/')[-6] == 'DSBtrain': continue
#         if t1 > 10000: continue
#         os.system('cp '+path+' '+train_path_1)
#         t1+=1
#         print 't1 %d, t0 %d, v1 %d, v0 %d' % (t1, t0, v1, v0)
#
#     if path.split('/')[-7] == 'FP':
#         if path.split('/')[-1][-7:] == '012.npy':
#             if t0 > 10000: continue
#             os.system('cp ' + path + ' ' + train_path_0)
#             t0 += 1
#             print 't1 %d, t0 %d, v1 %d, v0 %d' % (t1, t0, v1, v0)

for path in val_crop:
    if path.split('/')[-7] == 'TP' or path.split('/')[-7] == 'FN':
        if path.split('/')[-6] == 'DSBtrain': continue
        if v1 >= 5000: continue
        os.system('cp ' + path + ' ' + val_path_1)
        v1 += 1
        print 't1 %d, t0 %d, v1 %d, v0 %d' % (t1, t0, v1, v0)

    if path.split('/')[-7] == 'FP':
        if path.split('/')[-1][-7:] == '012.npy':
            if v0 >= 5000: continue
            os.system('cp ' + path + ' ' + val_path_0)
            v0 += 1
            print 't1 %d, t0 %d, v1 %d, v0 %d' % (t1, t0, v1, v0)

            # raw_input("enter")


# if not os.path.exists(base_dir): os.mkdir(base_dir)
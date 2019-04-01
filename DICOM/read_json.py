import json
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np

with open('/home/dkkim/works/DICOM/merge_2.json','rb') as infile:
    log = json.load(infile)

thickness_list = []
interval_list = []
spacing_list = []
manufacturer_list = []
exposure_list = []

for i in range(0,len(log)-1):
    if log[i]['SliceThickness'] != None :
        if type(log[i]['SliceThickness']) == unicode:
            log[i]['SliceThickness'] = round(float(log[i]['SliceThickness'].encode("ascii")),3)
            # thickness_list.append(round(float(log[i]['SliceThickness'].encode("ascii")),3))
        if log[i]['SliceThickness'] < 10:
            thickness_list.append(log[i]['SliceThickness'])
    if log[i]['Interval'] != None:
        if type(log[i]['Interval']) == unicode:
            log[i]['Interval'] = log[i]['Interval'].encode("ascii")
            # if log[i]['Interval'].encode("ascii") < 10:
            #     interval_list.append(log[i]['Interval'].encode("ascii"))
        if log[i]['Interval'] < 10:
            interval_list.append(log[i]['Interval'])
    if log[i]['PixelSpacing_x'] != None:
        if type(log[i]['PixelSpacing_x']) == unicode:
            log[i]['PixelSpacing_x'] = round(float(log[i]['PixelSpacing_x'].encode("ascii")), 5)
            # spacing_list.append(round(float(log[i]['PixelSpacing_x'].encode("ascii")),5))
        spacing_list.append(log[i]['PixelSpacing_x'])
    if log[i]['Manufacturer'] != None:
        if type(log[i]['Manufacturer']) == unicode:
            log[i]['Manufacturer'] = log[i]['Manufacturer'].encode("ascii")
            # manufacturer_list.append(log[i]['Manufacturer'].encode("ascii"))
        manufacturer_list.append(log[i]['Manufacturer'])
    if log[i]['mAs'] != None:
        if type(log[i]['mAs']) == unicode:
            log[i]['mAs'] = log[i]['mAs'].encode("ascii")
            # exposure_list.append(log[i]['mAs'].encode("ascii"))
        if log[i]['mAs'] < 2000:
            exposure_list.append(log[i]['mAs'])

# for i in range(0,len(log['patient'])-1):
#     thickness_list.append(log['patient'][i]['SliceThickness'])
#     interval_list.append(log['patient'][i]['Interval'])
#     spacing_list.append(log['patient'][i]['PixelSpaceing_x'])
#     manufacturer_list.append(log['patient'][i]['Manufacturer'])
#     exposure_list.append(log['patient'][i]['mAs'])



# n : n in periods / bins : boundary value / patches : matplotlib patch
plt.figure(1)
ys,xs,patches = plt.hist(thickness_list, align = 'mid', range=[0.5,5.5], bins = 10, rwidth = 0.8)
plt.title('Thickness(total %d)'% len(thickness_list),fontsize=25)
plt.grid()

for i in range(0, len(ys)):
    plt.text(x=xs[i]+0.1,y=ys[i]+0.015, s='{:.0f}'.format(ys[i]),fontsize=10,color = 'blue')

plt.legend

plt.figure(2)
ys,xs,patches = plt.hist(interval_list, align = 'mid', range=[0,2], bins = 20, rwidth = 0.8)
plt.title('Interval(total %d)'% len(interval_list),fontsize=25)
plt.grid()

for i in range(0, len(ys)):
    plt.text(x=xs[i],y=ys[i]+0.015, s='{:.0f}'.format(ys[i]),fontsize=8,color = 'blue')

plt.legend

plt.figure(3)
ys,xs,patches = plt.hist(spacing_list, align = 'mid', range=[0.1,1.1], bins = 10)
plt.title('Spacing(total %d)' % len(spacing_list),fontsize=25)
plt.grid()

for i in range(0, len(ys)):
    plt.text(x=xs[i]+0.02,y=ys[i]+0.015, s='{:.0f}'.format(ys[i]),fontsize=10,color = 'blue')

# print sorted(spacing_list)
# print max(spacing_list), min(spacing_list)
# print ys
# print xs
# plt.xticks(np.arange(0,0.5,0.1))
plt.legend

plt.figure(4)
ys,xs,patches = plt.hist(manufacturer_list, align = 'mid', bins = 5, rwidth = 0.8)
plt.title('Manufacturer(total %d)'% len(manufacturer_list),fontsize=25)
plt.grid()

for i in range(0, len(ys)):
    plt.text(x=xs[i]+0.1,y=ys[i]+0.015, s='{:.0f}'.format(ys[i]),fontsize=10,color = 'blue')


# print n
# print bins
# print patches[0]
plt.legend

plt.figure(5)
ys,xs,patches = plt.hist(exposure_list, align = 'mid', rwidth = 0.8, range=[0,2000],bins = 20)
plt.title('Exposure(total %d)' % len(exposure_list),fontsize=25)
plt.grid()

for i in range(0, len(ys)):
    plt.text(x=xs[i],y=ys[i]+0.015, s='{:.0f}'.format(ys[i]),fontsize=8,color = 'blue')
print sorted(exposure_list)
print max(exposure_list), min(exposure_list)
print ys
print xs

# sbn.distplot(thickness_list, kde=False)

plt.show()


#
# print log['patient'][0]['StudyDate']
# print len(log['patient'])

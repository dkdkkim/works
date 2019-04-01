import json
import matplotlib.pyplot as plt
import seaborn as sbn
import numpy as np

with open('/home/dkkim/works/DICOM/merge.json','rb') as infile:
    log = json.load(infile)

thickness_list = []
interval_list = []
spacing_list = []
manufacturer_list = []
exposure_list = []

for i in range(0,len(log)-1):
    if log[i]['SliceThickness'] != None :
        if type(log[i]['SliceThickness']) == unicode:
            thickness_list.append(round(float(log[i]['SliceThickness'].encode("ascii")),3))
        elif log[i]['SliceThickness'] < 10:
            thickness_list.append(log[i]['SliceThickness'])

print max(thickness_list)
print sorted(thickness_list)
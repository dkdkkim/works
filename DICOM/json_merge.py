import json
from glob import glob

file_list = glob('/home/dkkim/works/*.json')

head =[]

with open('/home/dkkim/works/DICOM/merge_2.json',"w") as outfile:
    for f in file_list:
        with open(f,'rb') as infile:
            file_data = json.load(infile)
            head += file_data['patient']
    json.dump(head,outfile)

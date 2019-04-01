def send_email(Subject, Text, From='nam_research@naver.com', To='sunnmoon137@gmail.com'):
    import smtplib
    from email.mime.text import MIMEText

    pwd = '~!Q@W#E$R'
    msg = MIMEText(Text, _charset='euc-kr')
    msg['Subject'] = Subject
    msg['From'] = From
    msg['To'] = To

    try:
        server = smtplib.SMTP('smtp.naver.com', 587)
        server.ehlo()
        server.starttls()
        server.ehlo()
        server.login(From, pwd)
        server.sendmail(From, To, msg.as_string())
        server.quit()

    except smtplib.SMTPException as e:
        print e


import pydicom as dicom
import json
from glob import glob
import pprint

# data_path = "/home/dkkim/CTs/115278/01-02-1999-NLST-LSS-57267/6617-0OPAPHMX8000C3383.212039.00.01.75-25145"

# data_path = "/home/dkkim/CTs/"
data_path = "/data/datasets"
output_path = "/home/dkkim/works"
folder_path = glob(data_path + '/*/*/*/*/*')
folder_path = sorted(folder_path)

# print '\n'.join(folder_path)
tags = ['Patient ID','PatientName']
dicom_log = {}
dicom_log['patient']=[]
k=1

for didx, path in enumerate(folder_path,1):
    iskind, studyDate = path.split('/')[3], path.split('/')[5]
    dcm_path = glob(path + '/*.dcm')

    if didx % 10 == 0: print didx
    if len(dcm_path) < 30: continue

    print path

    # dcm_path = sorted(dcm_path)
    # print '\n'.join(dcm_path)

    ### interval setting
    slices = [dicom.dcmread(s) for s in dcm_path]
    try:
        slices.sort(key=lambda x: x.ImagePositionPatient[2])
    except:
        try:
            slices.sort(key=lambda x: x.SliceLocation)
        except:
            print 'no data', path

    try:
        for i in range(0, len(slices) - 1):
            try:
                interval = abs(slices[i].ImagePositionPatient[2] - slices[i+1].ImagePOsitionPatient[2])
            except:
                interval = abs(slices[i].SliceLocation - slices[i+1].SliceLocation)
            if interval > 0: break
    except:
        interval = None

    # print didx, interval

    ds = slices[0]

    ### Exposure setting
    try:
        mAs = ds.Exposure
    except AttributeError:
        mAs = None

    ### SliceThickness setting
    try:
        SliceThickness = round(float(ds.SliceThickness), 3)
    except AttributeError:
        SliceThickness = interval

    ### Spacing setting
    try:
        PixelSpacing_x = round(float(ds.PixelSpacing[0]), 5)
        PixelSpacing_y = round(float(ds.PixelSpacing[1]), 5)
    except AttributeError:
        PixelSpacing_x = None
        PixelSpacing_y = None

    ### mA setting
    try:
        mA = ds.XRayTubeCurrent
    except AttributeError:
        mA = None

    ### KVP setting
    try:
        KVP = int(ds.KVP)
    except AttributeError:
        KVP = None

    dicom_log['patient'].append(
        {'PatientID': ds.PatientID, 'PatientName': ds.PatientName, 'StudyInstanceUID': ds.StudyInstanceUID, \
         'SeriesInstanceUID': ds.SeriesInstanceUID, 'Manufacturer': ds.Manufacturer, \
         'ManufacturerModelName': ds.ManufacturerModelName, 'SliceThickness': SliceThickness, \
         'PixelSpaceing_x': PixelSpacing_x, 'PixelSpacing_y': PixelSpacing_y, 'mA': mA,
         'KVP': KVP, 'mAs': mAs, \
         'Interval': interval, 'Path': path, 'iskind':iskind,'StudyDate':studyDate})

    print ds.SeriesInstanceUID

    if didx // 100 == k:
        file_path = output_path + '/dicom_log_%d.json'%k
        with open(file_path, 'w') as outfile:
            json.dump(dicom_log, outfile)
        dicom_log['patient'] = []
        k = k+1




## write json file
# with open(output_path + '/dicom_log.json', 'w') as outfile:
#     json.dump(dicom_log, outfile)

## read json file
# with open(output_path + '/dicom_log.json') as fd:
#     json_data = json.load(fd)

# print json file
# key = 'SeriesInstanceUID'
# for d in json_data['patient']:
#     print d[key]



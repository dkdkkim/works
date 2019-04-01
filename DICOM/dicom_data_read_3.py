try:

    import pydicom as dicom
    import json
    from glob import glob
    import pprint
    import send_email

    data_path = "/data/datasets"
    output_path = "/home/dkkim/works"
    study_path = data_path+"/*/*/*/*"
    series_path = glob(study_path + '/*')
    series_path = sorted(series_path)

    tags = ['Patient ID', 'PatientName']
    dicom_log = {}
    dicom_log['patient'] = []
    k = 1

    for didx, path in enumerate(series_path, 1):
        iskind, studyDate = path.split('/')[3], path.split('/')[5]
        dcm_path = glob(path + '/*.dcm')

        if didx % 100 == 0: print didx
        if len(dcm_path) < 30: continue

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
                    interval = abs(slices[i].ImagePositionPatient[2] - slices[i + 1].ImagePositionPatient[2])
                except:
                    interval = abs(slices[i].SliceLocation - slices[i + 1].SliceLocation)
                if interval > 0: break
        except:
            interval = None

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


        ### append to log
        dicom_log['patient'].append(
            {'PatientID': ds.PatientID, 'PatientName': ds.PatientName, 'StudyInstanceUID': ds.StudyInstanceUID, \
             'SeriesInstanceUID': ds.SeriesInstanceUID, 'Manufacturer': ds.Manufacturer, \
             'ManufacturerModelName': ds.ManufacturerModelName, 'SliceThickness': SliceThickness, \
             'PixelSpacing_x': PixelSpacing_x, 'PixelSpacing_y': PixelSpacing_y, 'mA': mA,
             'KVP': KVP, 'mAs': mAs, \
             'Interval': interval, 'Path': path, 'iskind': iskind, 'StudyDate': studyDate})

        ## make jason file
        if didx // 1000 == k:
            file_path = output_path + '/dicom_log_%d.json' % k
            with open(file_path, 'w') as outfile:
                json.dump(dicom_log, outfile)
            dicom_log['patient'] = []
            k = k + 1


except Exception as e:
    with open(output_path + '/dicom_log_' + k + 'error.json', 'w') as outfile:
        json.dump(dicom_log, outfile)

    Text = 'Error: ' + e + '\n check ' + str(didx)+ ' ' + path
    send_email.send_email('Something wrong happens','Need to check it!')


# print slices
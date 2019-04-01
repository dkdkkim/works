import sys, os, json, argparse, numpy as np
from glob import glob
from utils import *
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
# parser.add_argument("--split", type=str, default='train', help='train or valid or test')
# parser.add_argument("--cropType", type=str, default='TP', help='TP or FP or FN')
# parser.add_argument("--iskind", type=str, default='ssn', help="what kind of data, e.g. ssn, missed, etc.")
parser.add_argument("--cropSize", type=int, default=48, help="(int) crop size, default=48")
args = parser.parse_args()

# print args.split, args.cropType, args.iskind, args.cropSize
cs = args.cropSize


def crop_save():
    crop_save_dir = '/data/crops'
    fig_save_dir = '/data/figs'
    log_path = 'logs/crop_save_crop_save'
    if not os.path.exists(log_path): os.makedirs(log_path)

    transpose = [(0, 1, 2), (1, 2, 0), (2, 0, 1)]
    spacings = [0.9, 1.0, 1.2, 1.4, 1.6, 1.8]

    keys = ['iskind', 'patientID', 'studyDate',
            'studyUID', 'seriesUID', 'intervals',
            'spacing', 'split', 'cropType',
            'coordZ', 'coordY', 'coordX']
    print ','.join(keys)
    # raw_input("enter")
    try:
        db = Database()
        rows = db.select("select " + ','.join(keys) + " from crops")
    except Exception as e:
        print e
    finally:
        db.conn.close()

    try:
        prev_dcm_dir = ''
        for ridx, row in enumerate(rows, 1):
            print ridx, '/', len(rows)
            cur_db = {keys[i]: row[i] for i in range(len(keys))}

            cur_dcm_dir = '/'.join(['/data/datasets', cur_db['iskind'], str(cur_db['patientID']),
                                    str(cur_db['studyDate']), cur_db['studyUID'], cur_db['seriesUID'], '*.dcm'])

            dcms = glob(cur_dcm_dir)
            if len(dcms) == 0:
                cur_log_path = log_path + '/' + '_'.join([str(is_time())])
                simple_logs_saver(msg={'exception': 'len(dcms)==0, ' + cur_dcm_dir},
                                  log_path=cur_log_path,
                                  email_Subject='crop_save',
                                  email_Text='crop_save \n len(dcms)==0 occurs \n not exception\n' + cur_log_path)
                continue
            if prev_dcm_dir != cur_dcm_dir:
                ct = load_dcm(glob(cur_dcm_dir))
                ct = np.pad(ct, pad_width=((cs, cs), (cs, cs), (cs, cs)), mode='constant', constant_values=-1000)
            prev_dcm_dir = cur_dcm_dir

            spacing = np.array([cur_db['intervals'], cur_db['spacing'], cur_db['spacing']])
            coord = np.array([cur_db['coordZ'], cur_db['coordY'], cur_db['coordX']])

            crop = ct[coord[0]:coord[0] + 2 * cs, coord[1]:coord[1] + 2 * cs, coord[2]:coord[2] + 2 * cs]
            crop = (crop + 1000.) / 1400.

            for spidx, new_sp in enumerate(spacings):

                if spidx == 0:
                    new_spacing = np.array(3 * [cur_db['spacing'] * new_sp])
                    cur_crop = np.copy(crop)
                else:
                    new_spacing = np.array(3 * [cur_db['spacing'] * new_sp])
                    cur_crop = interpolation(np.copy(crop),
                                             spacing=spacing,
                                             new_spacing=new_spacing)

                for tridx, new_idxs in enumerate(transpose):

                    if tridx == 0:
                        tmp_crop = np.copy(cur_crop)
                    else:
                        tmp_crop = np.transpose(np.copy(cur_crop))

                    for rg in range(4):

                        if rg != 0:
                            tmp_crop = rotation(tmp_crop, 90)

                        shp = np.array(tmp_crop.shape) / 2

                        tmp = tmp_crop[shp[0] - cs / 2: shp[0] + cs / 2,
                              shp[1] - cs / 2: shp[1] + cs / 2,
                              shp[2] - cs / 2: shp[2] + cs / 2]

                        if tmp.shape != (cs, cs, cs):
                            cur_log_path = log_path + '/' + '_'.join([str(is_time())])
                            simple_logs_saver(msg={'exception': 'crop shape != (cs, cs, cs)'},
                                              log_path=cur_log_path,
                                              email_Subject='crop_save -- shape',
                                              email_Text='crop_save \n shape != cropSize occurs \n not exception\n' + cur_log_path)
                            continue
                        else:
                            ### figure
                            # cur_fig_save_dir = '/'.join([fig_save_dir, cur_db['split'], cur_db['cropType'], cur_db['iskind'], str(cur_db['patientID']),
                            #                          str(cur_db['studyDate']), cur_db['studyUID'], cur_db['seriesUID']])
                            # if not os.path.exists(cur_fig_save_dir): os.makedirs(cur_fig_save_dir)
                            # fig, ax = plt.subplots(4, 4, figsize=(6, 6))
                            # for pidx in range(16):
                            #     ax[pidx / 4, pidx % 4].imshow(tmp[tmp.shape[0]/2 - 8 + pidx], cmap='bone', clim=(0., 1.5))
                            #     ax[pidx / 4, pidx % 4].axis('off')

                            ### crop
                            cur_save_dir = '/'.join(
                                [crop_save_dir, cur_db['split'], cur_db['cropType'], cur_db['iskind'],
                                 str(cur_db['patientID']),
                                 str(cur_db['studyDate']), cur_db['studyUID'], cur_db['seriesUID']])
                            if not os.path.exists(cur_save_dir): os.makedirs(cur_save_dir)
                            transpos_val = ''
                            for temp_t_val in new_idxs: transpos_val += str(temp_t_val)
                            crop_name = '_'.join([str(cur_db['coordZ']), str(cur_db['coordY']), str(cur_db['coordX']),
                                                  str(cur_db['intervals']), str(cur_db['spacing']), str(new_sp),
                                                  str(new_spacing[0]), str(rg * 90), transpos_val])
                            cur_save_dir += '/' + crop_name

                            ### crops save
                            np.save(str(cur_save_dir), tmp)

                            ### fig save
                            # cur_fig_save_dir += '/' + crop_name
                            #
                            # plt.savefig(cur_fig_save_dir + '.png')
                            # plt.close()
                            # raw_input("enter")


    except Exception as e:
        cur_log_path = log_path + '/' + '_'.join([str(is_time())])
        simple_logs_saver(msg={'exception': str(e)},
                          log_path=cur_log_path,
                          email_Subject='crop_save Exception occurs ',
                          email_Text='crop_save \n Exception occurs \n' + str(e) + '\n' + cur_log_path)


# crop_save()
import os


def crop_fig_save():
    cnt = 0
    root_dir = '/data/crops/train'
    root_fig_dir = '/data/figs/'
    for cropType in glob(root_dir + '/*'):
        cur_cropType = cropType.split('/')[-1]
        for iskind in glob(cropType + '/*'):
            for patientID in glob(iskind + '/*'):
                for studyDate in glob(patientID + '/*'):
                    for studyUID in glob(studyDate + '/*'):
                        for seriesUID in glob(studyUID + '/*'):
                            for npy in glob(seriesUID + '/*'):

                                cur_npy = os.path.splitext(npy.split('/')[-1])[0].split('_')

                                if cur_npy[-1] == '012' and cur_npy[-2] == '0' and cur_npy[-4] == '1.0':
                                    # tmp = np.load(npy)
                                    cnt += 1
                                    print cnt
                                    fig_suptitle = '/'.join(npy.split('/')[3:-3])
                                    fig_name = '_'.join(npy.split('/')[3:-3]) + '_' + '_'.join(cur_npy[:3])
                                    cur_fig_dir = root_fig_dir + cur_cropType
                                    if not os.path.exists(cur_fig_dir): os.makedirs(cur_fig_dir)
                                    # if not os.path.exists(fig_save_dir): os.makedirs(fig_save_dir)
                                    try:
                                        tmp = np.load(npy)
                                        fig, ax = plt.subplots(4, 4, figsize=(6, 6))
                                        for pidx in range(16):
                                            ax[pidx / 4, pidx % 4].imshow(tmp[tmp.shape[0] / 2 - 8 + pidx], cmap='bone',
                                                                          clim=(0., 1.5))
                                            ax[pidx / 4, pidx % 4].axis('off')
                                        plt.suptitle(fig_suptitle)
                                        plt.savefig(cur_fig_dir + '/' + fig_name + '.png')
                                    except Exception as e:
                                        print e
                                        log_path = 'logs/crop_save_crop_fig_save'
                                        cur_log_path = log_path + '/' + '_'.join([str(cnt), str(is_time())])
                                        simple_logs_saver(msg={'exception': str(e) + '\t' + cur_log_path},
                                                          log_path=cur_log_path,
                                                          email_Subject='crop_save Exception occurs ',
                                                          email_Text='crop_save \n Exception occurs \n' + str(
                                                              e) + '\n' + cur_log_path + '\n cnt = ' + str(cnt))
                                    finally:
                                        plt.close()
                                else:
                                    continue
                                # raw_input("enter")

# crop_fig_save()
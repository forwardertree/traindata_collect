"""
데이터 전처리
"""

import numpy as np
import pickle
import os

raw_path = r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\24'
raw_list = np.array([x for x in os.listdir(raw_path) if 'ecgrdvq' in x])
# raw_list = np.array([x for x in os.listdir(raw_path) if 'ecgrdvq_1010_1_62.5sampling_training.csv' in x])


def differential(arr, period):
    # arr = raw , outp = diff
    outp = np.empty_like(arr)
    outp[1:] = (arr[1:] - arr[:-1]) / period
    outp[0] = 0
    return outp

def add_21(arr, fname):

    if 'vf' in fname:
        outp = np.append(arr, 0)
        outp = np.append(outp, 1)
    elif np.sum(arr) == 0:
        outp = np.append(arr, 1)
        outp = np.append(outp, 0)
    else:
        outp = np.append(arr, 0)
        outp = np.append(outp, 0)

    return outp


save_inp = np.zeros((1, 160))
save_ps = np.zeros((1, 22))
save_pe = np.zeros((1, 22))
save_qrss = np.zeros((1, 22))
save_qrse = np.zeros((1, 22))
save_ts = np.zeros((1, 22))
save_te = np.zeros((1, 22))

for x in raw_list:
    data = np.loadtxt('%s/%s' % (raw_path, x), dtype='str', delimiter=',')

    ps_ = np.array(data[1::, 1], dtype='float32')
    pe_ = np.array(data[1::, 2], dtype='float32')
    qrss_ = np.array(data[1::, 3], dtype='float32')
    qrse_ = np.array(data[1::, 4], dtype='float32')
    ts_ = np.array(data[1::, 5], dtype='float32')
    te_ = np.array(data[1::, 6], dtype='float32')

    for lead_i in range(0, 1):
        y1 = np.array(data[1::, lead_i], dtype='float32')  # sampling = 62.5

        if np.sum(y1) == 0:
            pass

        else:
            for raw_i in range(0, len(y1)):

                if raw_i < 60:
                    continue

                if raw_i + 60 > len(y1):
                    break

                print(f'f:{x}, lead:{lead_i}, raw:{raw_i}/{len(y1)}', end='\r')

                ecg_ = y1[raw_i - 60:raw_i + 60]
                ecg_max_val = max(ecg_)
                ecg_min_val = min(ecg_)
                ecg_ = (ecg_ - ecg_min_val) / (ecg_max_val - ecg_min_val)
                ecg = np.array(ecg_[60::])

                tompk_ = differential(arr=ecg_, period=1 / 62.5) ** 2  # tompkins = 미분 + 제곱
                tompk_max_val = max(tompk_)
                tompk_min_val = min(tompk_)
                tompk_ = (tompk_ - tompk_min_val) / (tompk_max_val - tompk_min_val)
                tompk = np.array(tompk_[60::])

                tompk_max = np.zeros(20)
                ref = np.argmax(tompk[20:40])
                if tompk[20 + ref] == max(tompk[10 + ref:30 + ref]) and tompk[20 + ref] > 0.7:
                    tompk_max[ref] = 1

                ecg_max = np.zeros(20)
                ref = np.argmax(ecg[20:40])
                if ecg[20 + ref] == max(ecg[10 + ref:30 + ref]) and ecg[20 + ref] > 0.7 and np.sum(tompk_max[ref-3:ref+3]) > 0:
                    ecg_max[ref] = 1

                inp = np.hstack((ecg, tompk, ecg_max, tompk_max))

                ps = add_21(ps_[raw_i + 20:raw_i + 40], x)
                pe = add_21(pe_[raw_i + 20:raw_i + 40], x)
                qrss = add_21(qrss_[raw_i + 20:raw_i + 40], x)
                qrse = add_21(qrse_[raw_i + 20:raw_i + 40], x)
                ts = add_21(ts_[raw_i + 20:raw_i + 40], x)
                te = add_21(te_[raw_i + 20:raw_i + 40], x)

                save_inp = np.vstack((save_inp, inp))
                save_ps = np.vstack((save_ps, ps))
                save_pe = np.vstack((save_pe, pe))
                save_qrss = np.vstack((save_qrss, qrss))
                save_qrse = np.vstack((save_qrse, qrse))
                save_ts = np.vstack((save_ts, ts))
                save_te = np.vstack((save_te, te))

save_dict = {
    'inp': save_inp[1::, :], 'ps': save_ps[1::, :], 'pe': save_pe[1::, :],
    'qrss': save_qrss[1::, :], 'qrse': save_qrse[1::, :],
    'ts': save_ts[1::, :], 'te': save_te[1::, :]
}

with open(r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\trainingDB_9_13_231213_0.8_2.pickle', 'wb') as f:
    pickle.dump(save_dict, f)

# save_data = np.hstack((save_inp[1::, :], save_qrss[1::, :]))
# np.savetxt(r'K:\inp_231117.csv', save_data, fmt='%s', delimiter=",")

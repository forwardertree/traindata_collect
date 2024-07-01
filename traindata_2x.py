"""
데이터 전처리
"""

import numpy as np
import pandas as pd
import pickle
import os
import statistics

raw_path = r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\24'
raw_list = np.array([x for x in os.listdir(raw_path) if '1009' in x])
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


save_inp_ps = np.zeros((1, 160))
save_inp_pe = np.zeros((1, 160))
save_inp_qrss = np.zeros((1, 160))
save_inp_qrse = np.zeros((1, 160))
save_inp_ts = np.zeros((1, 160))
save_inp_te = np.zeros((1, 160))
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
                if tompk[20 + ref] == max(tompk[10 + ref:30 + ref]) and tompk[20 + ref] > 0.8 and np.sum(tompk_max) == 0:
                    tompk_max[ref] = 1

                ecg_max = np.zeros(20)
                ref = np.argmax(ecg[20:40])
                if ecg[20 + ref] == max(ecg[10 + ref:30 + ref]) and ecg[20 + ref] > 0.8 and np.sum(tompk_max[ref-3:ref+3]) > 0 and np.sum(ecg_max) == 0:
                    ecg_max[ref] = 1

                inp = np.hstack((
                    ecg, tompk, ecg_max, tompk_max
                ))

                ps = add_21(ps_[raw_i + 20:raw_i + 40], x)
                pe = add_21(pe_[raw_i + 20:raw_i + 40], x)
                qrss = add_21(qrss_[raw_i + 20:raw_i + 40], x)
                qrse = add_21(qrse_[raw_i + 20:raw_i + 40], x)
                ts = add_21(ts_[raw_i + 20:raw_i + 40], x)
                te = add_21(te_[raw_i + 20:raw_i + 40], x)

                # save_inp = np.vstack((save_inp, inp))
                # save_ps = np.vstack((save_ps, ps))
                # save_pe = np.vstack((save_pe, pe))
                # save_qrss = np.vstack((save_qrss, qrss))
                # save_qrse = np.vstack((save_qrse, qrse))
                # save_ts = np.vstack((save_ts, ts))
                # save_te = np.vstack((save_te, te))

                if ps[20] == 0:
                    save_ps = np.vstack((save_ps, ps))
                    save_ps = np.vstack((save_ps, ps))
                    # save_ps = np.vstack((save_ps, ps, ps))
                    save_inp_ps = np.vstack((save_inp_ps, inp))
                    save_inp_ps = np.vstack((save_inp_ps, inp))

                else:
                    save_ps = np.vstack((save_ps, ps))
                    save_inp_ps = np.vstack((save_inp_ps, inp))


                if pe[20] == 0:
                    save_pe = np.vstack((save_pe, pe))
                    save_pe = np.vstack((save_pe, pe))
                    save_inp_pe = np.vstack((save_inp_pe, inp))
                    save_inp_pe = np.vstack((save_inp_pe, inp))


                else:
                    save_pe = np.vstack((save_pe, pe))
                    save_inp_pe = np.vstack((save_inp_pe, inp))


                if qrss[20] == 0:
                    save_qrss = np.vstack((save_qrss, qrss))
                    save_qrss = np.vstack((save_qrss, qrss))
                    save_inp_qrss = np.vstack((save_inp_qrss, inp))
                    save_inp_qrss = np.vstack((save_inp_qrss, inp))

                else:
                    save_qrss = np.vstack((save_qrss, qrss))
                    save_inp_qrss = np.vstack((save_inp_qrss, inp))


                if qrse[20] == 0:
                    save_qrse = np.vstack((save_qrse, qrse))
                    save_qrse = np.vstack((save_qrse, qrse))
                    save_inp_qrse = np.vstack((save_inp_qrse, inp))
                    save_inp_qrse = np.vstack((save_inp_qrse, inp))

                else:
                    save_qrse = np.vstack((save_qrse, qrse))
                    save_inp_qrse = np.vstack((save_inp_qrse, inp))


                if ts[20] == 0:
                    save_ts = np.vstack((save_ts, ts))
                    save_ts = np.vstack((save_ts, ts))
                    save_inp_ts = np.vstack((save_inp_ts, inp))
                    save_inp_ts = np.vstack((save_inp_ts, inp))

                else:
                    save_ts = np.vstack((save_ts, ts))
                    save_inp_ts = np.vstack((save_inp_ts, inp))


                if te[20] == 0:
                    save_te = np.vstack((save_te, te))
                    save_te = np.vstack((save_te, te))
                    save_inp_te = np.vstack((save_inp_te, inp))
                    save_inp_te = np.vstack((save_inp_te, inp))

                else:
                    save_te = np.vstack((save_te, te))
                    save_inp_te = np.vstack((save_inp_te, inp))


save_dict = {
    'inp_ps': save_inp_ps[1::, :], 'inp_pe': save_inp_pe[1::, :], 'ps': save_ps[1::, :], 'pe': save_pe[1::, :],
    'inp_qrss': save_inp_qrss[1::, :], 'inp_qrse': save_inp_qrse[1::, :], 'qrss': save_qrss[1::, :], 'qrse': save_qrse[1::, :],
    'inp_ts': save_inp_ts[1::, :], 'inp_te': save_inp_te[1::, :], 'ts': save_ts[1::, :], 'te': save_te[1::, :]
}





with open(r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\trainingDB_9_13_240325_0.8test.pickle', 'wb') as f:
    pickle.dump(save_dict, f)

df_inp_ps = pd.DataFrame(save_dict['inp_ps'])
df_inp_pe = pd.DataFrame(save_dict['inp_pe'])
df_inp_qrss = pd.DataFrame(save_dict['inp_qrss'])
df_inp_qrse = pd.DataFrame(save_dict['inp_qrse'])
df_inp_ts = pd.DataFrame(save_dict['inp_ts'])
df_inp_te = pd.DataFrame(save_dict['inp_te'])
df_ps = pd.DataFrame(save_dict['ps'])
df_pe = pd.DataFrame(save_dict['pe'])
df_qrss = pd.DataFrame(save_dict['qrss'])
df_qrse = pd.DataFrame(save_dict['qrse'])
df_ts = pd.DataFrame(save_dict['ts'])
df_te = pd.DataFrame(save_dict['te'])
# Excel 파일로 저장
excel_path = r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\trainingDB_9_13_240325_0.8test.xlsx'
with pd.ExcelWriter(excel_path) as writer:
    df_inp_ps.to_excel(writer, sheet_name='inp_ps', index=False)
    df_inp_pe.to_excel(writer, sheet_name='inp_pe', index=False)
    df_inp_qrss.to_excel(writer, sheet_name='inp_qrss', index=False)
    df_inp_qrse.to_excel(writer, sheet_name='inp_qrse', index=False)
    df_inp_ts.to_excel(writer, sheet_name='inp_ts', index=False)
    df_inp_te.to_excel(writer, sheet_name='inp_te', index=False)
    df_ps.to_excel(writer, sheet_name='ps', index=False)
    df_pe.to_excel(writer, sheet_name='pe', index=False)
    df_qrss.to_excel(writer, sheet_name='qrss', index=False)
    df_qrse.to_excel(writer, sheet_name='qrse', index=False)
    df_ts.to_excel(writer, sheet_name='ts', index=False)
    df_te.to_excel(writer, sheet_name='te', index=False)

# save_data = np.hstack((save_inp[1::, :], save_qrss[1::, :]))
# np.savetxt(r'K:\inp_231117.csv', save_data, fmt='%s', delimiter=",")

import numpy as np
from datetime import datetime
import pickle

# 1. 파일호출
with open(r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\inp_outp.pickle', 'rb') as f:
    stack_arr = pickle.load(f)
    inp = stack_arr[:, 0:160]
    ps_outp = stack_arr[:, 160:180]
    ps_outp_vf = stack_arr[:, 180:181]
    pe_outp = stack_arr[:, 181:201]
    pe_outp_vf = stack_arr[:, 201:202]
    qrss_outp = stack_arr[:, 202:222]
    qrss_outp_vf = stack_arr[:, 222:223]
    qrse_outp = stack_arr[:, 223:243]
    qrse_outp_vf = stack_arr[:, 243:244]
    ts_outp = stack_arr[:, 244:264]
    ts_outp_vf = stack_arr[:, 264:265]
    te_outp = stack_arr[:, 265:285]
    te_outp_vf = stack_arr[:, 285:286]

with open(r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\rs_i.pickle', 'rb') as f:
    ecg_parameter = pickle.load(f)

# 2. P부터, RS 탐색 및 저장
ps_stack_arr = np.array([])
pe_stack_arr = np.array([])
qrss_stack_arr = np.array([])
qrse_stack_arr = np.array([])
ts_stack_arr = np.array([])
te_stack_arr = np.array([])

p_rs = ecg_parameter[1::, 1]
qrs_rs = ecg_parameter[1::, 2]
t_rs = ecg_parameter[1::, 3]
ps_i = ecg_parameter[1::, -6]
pe_i = ecg_parameter[1::, -5]
qrss_i = ecg_parameter[1::, -4]
qrse_i = ecg_parameter[1::, -3]
ts_i = ecg_parameter[1::, -2]
te_i = ecg_parameter[1::, -1]

for ps_i in range(0, len(p_rs)):
    if int(float(p_rs[ps_i])) > 15:
        ref = int(ps_i / 4)

        if len(ps_stack_arr) == 0:
            ps_stack_arr = np.hstack((
                inp[ref-10:ref+10, :], ps_outp[ref-10:ref+10, :]
            ))
        else:
            ps_stack_tmp = np.hstack((
                inp[ref-10:ref+10, :], ps_outp[ref-10:ref+10, :]
            ))
            ps_stack_arr = np.vstack((ps_stack_arr, ps_stack_tmp))

for pe_i in range(0, len(p_rs)):
    if int(float(p_rs[pe_i])) > 15:
        ref = int(pe_i / 4)

        if len(pe_stack_arr) == 0:
            pe_stack_arr = np.hstack((
                inp[ref-10:ref+10, :], pe_outp[ref-10:ref+10, :]
            ))
        else:
            pe_stack_tmp = np.hstack((
                inp[ref-10:ref+10, :], pe_outp[ref-10:ref+10, :]
            ))
            pe_stack_arr = np.vstack((pe_stack_arr, pe_stack_tmp))

for qrss_i in range(0, len(qrs_rs)):
    if int(float(qrs_rs[qrss_i])) > 15:
        ref = int(qrss_i / 4)

        if len(qrss_stack_arr) == 0:
            qrss_stack_arr = np.hstack((
                inp[ref - 10:ref + 10, :], qrss_outp[ref - 10:ref + 10, :]
            ))
        else:
            qrss_stack_tmp = np.hstack((
                inp[ref - 10:ref + 10, :], qrss_outp[ref - 10:ref + 10, :]
            ))
            qrss_stack_arr = np.vstack((qrss_stack_arr, qrss_stack_tmp))

for qrse_i in range(0, len(qrs_rs)):
    if int(float(qrs_rs[qrse_i])) > 15:
        ref = int(qrse_i / 4)

        if len(qrse_stack_arr) == 0:
            qrse_stack_arr = np.hstack((
                inp[ref - 10:ref + 10, :], qrse_outp[ref - 10:ref + 10, :]
            ))
        else:
            qrse_stack_tmp = np.hstack((
                inp[ref - 10:ref + 10, :], qrse_outp[ref - 10:ref + 10, :]
            ))
            qrse_stack_arr = np.vstack((qrse_stack_arr, qrse_stack_tmp))

for ts_i in range(0, len(t_rs)):
    if int(float(t_rs[ts_i])) > 15:
        ref = int(ts_i / 4)

        if len(ts_stack_arr) == 0:
            ts_stack_arr = np.hstack((
                inp[ref - 16:ref + 10, :], ts_outp[ref - 16:ref + 10, :]
            ))
        else:
            ts_stack_tmp = np.hstack((
                inp[ref - 16:ref + 10, :], ts_outp[ref - 16:ref + 10, :]
            ))
            ts_stack_arr = np.vstack((ts_stack_arr, ts_stack_tmp))

for te_i in range(0, len(t_rs)):
    if int(float(t_rs[te_i])) > 15:
        ref = int(te_i / 4)

        if len(te_stack_arr) == 0:
            te_stack_arr = np.hstack((
                inp[ref - 16:ref + 10, :], te_outp[ref - 16:ref + 10, :]
            ))
        else:
            te_stack_tmp = np.hstack((
                inp[ref - 16:ref + 10, :], te_outp[ref - 16:ref + 10, :]
            ))
            te_stack_arr = np.vstack((te_stack_arr, te_stack_tmp))
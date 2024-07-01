import numpy as np
from datetime import datetime
import pickle
import pandas as pd
import os

# 1. 파일호출
with open(r'C:\Users\jjm\Desktop\ooo1\0001\240404\inp_outp_1001_p.pickle', 'rb') as f:
    stack_arr = pickle.load(f)
    inp = stack_arr[:, 0:160]
    ps_outp = stack_arr[:, 160:180]
    ps_outp_vf = stack_arr[:, 180:181]
    ps_outp_21 = np.zeros((len(ps_outp), 1))
    for i in range(0, len(ps_outp)):
        if np.sum(ps_outp[i, :]) == 0 and ps_outp_vf[i, 0] == 0:
            ps_outp_21[i, 0] = 1
    ps_outp = np.hstack((
        ps_outp, ps_outp_21, ps_outp_vf
    ))

    pe_outp = stack_arr[:, 181:201]
    pe_outp_vf = stack_arr[:, 201:202]
    pe_outp_21 = np.zeros((len(pe_outp), 1))
    for i in range(0, len(pe_outp)):
        if np.sum(pe_outp[i, :]) == 0 and pe_outp_vf[i, 0] == 0:
            pe_outp_21[i, 0] = 1
    pe_outp = np.hstack((
        pe_outp, pe_outp_21, pe_outp_vf
    ))

    qrss_outp = stack_arr[:, 202:222]
    qrss_outp_vf = stack_arr[:, 222:223]
    qrss_outp_21 = np.zeros((len(qrss_outp), 1))
    for i in range(0, len(qrss_outp)):
        if np.sum(qrss_outp[i, :]) == 0 and qrss_outp_vf[i, 0] == 0:
            qrss_outp_21[i, 0] = 1
    qrss_outp = np.hstack((
        qrss_outp, qrss_outp_21, qrss_outp_vf
    ))

    qrse_outp = stack_arr[:, 223:243]
    qrse_outp_vf = stack_arr[:, 243:244]
    qrse_outp_21 = np.zeros((len(qrse_outp), 1))
    for i in range(0, len(qrse_outp)):
        if np.sum(qrse_outp[i, :]) == 0 and qrse_outp_vf[i, 0] == 0:
            qrse_outp_21[i, 0] = 1
    qrse_outp = np.hstack((
        qrse_outp, qrse_outp_21, qrse_outp_vf
    ))

    ts_outp = stack_arr[:, 244:264]
    ts_outp_vf = stack_arr[:, 264:265]
    ts_outp_21 = np.zeros((len(ts_outp), 1))
    for i in range(0, len(ts_outp)):
        if np.sum(ts_outp[i, :]) == 0 and ts_outp_vf[i, 0] == 0:
            ts_outp_21[i, 0] = 1
    ts_outp = np.hstack((
        ts_outp, ts_outp_21, ts_outp_vf
    ))

    te_outp = stack_arr[:, 265:285]
    te_outp_vf = stack_arr[:, 285:286]
    te_outp_21 = np.zeros((len(te_outp), 1))
    for i in range(0, len(te_outp)):
        if np.sum(te_outp[i, :]) == 0 and te_outp_vf[i, 0] == 0:
            te_outp_21[i, 0] = 1
    te_outp = np.hstack((
        te_outp, te_outp_21, te_outp_vf
    ))

with open(r'C:\Users\jjm\Desktop\ooo1\0001\240404\rs_i_1001_p.pickle', 'rb') as f:
    ecg_parameter = pickle.load(f)

# 2. P부터, RS 탐색 및 저장
ps_inp_arr = np.array([])
ps_outp_arr = np.array([])
pe_inp_arr = np.array([])
pe_outp_arr = np.array([])
qrss_inp_arr = np.array([])
qrss_outp_arr = np.array([])
qrse_inp_arr = np.array([])
qrse_outp_arr = np.array([])
qrse_stack_arr = np.array([])
ts_inp_arr = np.array([])
ts_outp_arr = np.array([])
te_inp_arr = np.array([])
te_outp_arr = np.array([])

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
    if float(p_rs[ps_i]) > 20:
        ref = int(ps_i / 4)

        if len(ps_inp_arr) == 0:
            ps_inp_arr = inp[ref-10:ref+10, :]

        else:
            ps_inp_tmp = inp[ref-10:ref+10, :]
            ps_inp_arr = np.vstack((ps_inp_arr, ps_inp_tmp))

        if len(ps_outp_arr) == 0:
            ps_outp_arr = ps_outp[ref-10:ref+10, :]
        else:
            ps_outp_tmp = ps_outp[ref-10:ref+10, :]
            ps_outp_arr = np.vstack((ps_outp_arr, ps_outp_tmp))

for pe_i in range(0, len(p_rs)):
    if float(p_rs[pe_i]) > 20:
        ref = int(pe_i / 4)

        if len(pe_inp_arr) == 0:
            pe_inp_arr = inp[ref-10:ref+10, :]

        else:
            pe_inp_tmp = inp[ref-10:ref+10, :]
            pe_inp_arr = np.vstack((pe_inp_arr, pe_inp_tmp))

        if len(pe_outp_arr) == 0:
            pe_outp_arr = pe_outp[ref-10:ref+10, :]
        else:
            pe_outp_tmp = pe_outp[ref-10:ref+10, :]
            pe_outp_arr = np.vstack((pe_outp_arr, pe_outp_tmp))

for qrss_i in range(0, len(qrs_rs)):
    if float(qrs_rs[qrss_i]) > 20:
        ref = int(qrss_i / 4)

        if len(qrss_inp_arr) == 0:
            qrss_inp_arr = inp[ref-10:ref+10, :]

        else:
            qrss_inp_tmp = inp[ref-10:ref+10, :]
            qrss_inp_arr = np.vstack((qrss_inp_arr, qrss_inp_tmp))

        if len(qrss_outp_arr) == 0:
            qrss_outp_arr = qrss_outp[ref-10:ref+10, :]
        else:
            qrss_outp_tmp = qrss_outp[ref-10:ref+10, :]
            qrss_outp_arr = np.vstack((qrss_outp_arr, qrss_outp_tmp))

for qrse_i in range(0, len(qrs_rs)):
    if float(qrs_rs[qrse_i]) > 20:
        ref = int(qrse_i / 4)

        if len(qrse_inp_arr) == 0:
            qrse_inp_arr = inp[ref-10:ref+10, :]

        else:
            qrse_inp_tmp = inp[ref-10:ref+10, :]
            qrse_inp_arr = np.vstack((qrse_inp_arr, qrse_inp_tmp))

        if len(qrse_outp_arr) == 0:
            qrse_outp_arr = qrse_outp[ref-10:ref+10, :]
        else:
            qrse_outp_tmp = qrse_outp[ref-10:ref+10, :]
            qrse_outp_arr = np.vstack((qrse_outp_arr, qrse_outp_tmp))

for ts_i in range(0, len(t_rs)):
    if float(t_rs[ts_i]) > 35:
        ref = int(ts_i / 4)

        if len(ts_inp_arr) == 0:
            ts_inp_arr = inp[ref-10:ref+10, :]

        else:
            ts_inp_tmp = inp[ref-10:ref+10, :]
            ts_inp_arr = np.vstack((ts_inp_arr, ts_inp_tmp))

        if len(ts_outp_arr) == 0:
            ts_outp_arr = ts_outp[ref-10:ref+10, :]
        else:
            ts_outp_tmp = ts_outp[ref-10:ref+10, :]
            ts_outp_arr = np.vstack((ts_outp_arr, ts_outp_tmp))

for te_i in range(0, len(t_rs)):
    if float(t_rs[te_i]) > 37:
        ref = int(te_i / 4)

        if len(te_inp_arr) == 0:
            te_inp_arr = inp[ref-10:ref+10, :]

        else:
            te_inp_tmp = inp[ref-10:ref+10, :]
            te_inp_arr = np.vstack((te_inp_arr, te_inp_tmp))

        if len(te_outp_arr) == 0:
            te_outp_arr = te_outp[ref-10:ref+10, :]
        else:
            te_outp_tmp = te_outp[ref-10:ref+10, :]
            te_outp_arr = np.vstack((te_outp_arr, te_outp_tmp))

# for p_ie in range(0, len(p_rs)):
#     if float(p_rs[p_ie]) > 15:
#         ref = int(p_ie / 4)
#
#         if len(pe_stack_arr) == 0:
#             pe_stack_arr = np.hstack((
#                 inp[ref-10:ref+10, :],  pe_outp[ref-10:ref+10, :]
#             ))
#         else:
#             pe_stack_tmp = np.hstack((
#                 inp[ref-10:ref+10, :],  pe_outp[ref-10:ref+10, :]
#             ))
#             pe_stack_arr = np.vstack((pe_stack_arr, pe_stack_tmp))
#

#
save_dict = {
    'inp_ps': ps_inp_arr[1::, :], 'inp_pe': pe_inp_arr[1::, :], 'ps': ps_outp_arr[1::, :], 'pe': pe_outp_arr[1::, :],
    'inp_qrss': qrss_inp_arr[1::, :], 'inp_qrse': qrse_inp_arr[1::, :], 'qrss': qrss_outp_arr[1::, :], 'qrse': qrse_outp_arr[1::, :],
    'inp_ts': ts_inp_arr[1::, :], 'inp_te': te_inp_arr[1::, :], 'ts': ts_outp_arr[1::, :], 'te': te_outp_arr[1::, :]
}

with open(r'C:\Users\jjm\Desktop\ooo1\0001\240404\trainingDB_9_13_240404_1001_p.pickle', 'wb') as f:
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
excel_path = r'C:\Users\jjm\Desktop\ooo1\0001\240404\trainingDB_9_13_240404_1001_p.xlsx'
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
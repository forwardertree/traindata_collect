"""
신경망 평가
"""

import numpy as np
import os
from ecg_pkg.ecgdnn import ECG_DNN_Thread

if __name__ == '__main__':

    # 수정할 부분 1: 주소
    raw_path = r'C:\Users\jjm\Desktop\ecg_ai\aha_ecg_data'
    # 수정할 부분 2: 파일명
    raw_list = np.array([x for x in os.listdir(raw_path) if '1001' in x])

    for lead_i in range(0, 1):
        for j in range(len(raw_list)):
            x = raw_list[j]
            print(x)
            data = np.loadtxt(r'%s\%s' % (raw_path, x), dtype='str', delimiter=',')

            ecg = np.array(data[1::, lead_i], dtype='float32')
            ecg_dnn_thread = ECG_DNN_Thread(ecg, 250)
            ai_return, parameter_return, parameter_vf_return = ecg_dnn_thread.run_dnn()

            ecg_point = np.array([
                'raw',
                'major_ps', 'major_pe', 'major_qrss', 'major_qrse', 'major_ts', 'major_te',
                'point_ps', 'point_pe', 'point_qrss', 'point_qrse', 'point_ts', 'point_te'
            ])
            ecg_point_tmp = np.vstack((
                ai_return[0],
                ai_return[1], ai_return[2], ai_return[3], ai_return[4], ai_return[5], ai_return[6],
                ai_return[7], ai_return[8], ai_return[9], ai_return[10], ai_return[11], ai_return[12]
            )).T
            ecg_point = np.vstack((ecg_point, ecg_point_tmp))
            ecg_point = np.array(ecg_point, dtype='str')

            # 수정된 부분: 저장할 경로, 특이점
            np.savetxt(r'C:\Users\jjm\Desktop\ooo1\0001\240404\1001_%s_%s_%d.csv' % (lead_i, 'point', j), ecg_point, fmt='%s', delimiter=",")

            ecg_parameter = np.array([
                'time',
                'p_rs', 'qrs_rs', 't_rs',
                'p_duration', 'p_area', 'qrs_duration', 'qrs_area', 't_duration', 't_area',
                'pq_interval', 'st_interval', 'qq_interval',
                'pq_segment', 'qt_segment', 'tp_segment',
                'p_amp', 'qrs_amp', 't_amp',
                'p_avg', 'qrs_avg', 't_avg'
            ])
            ecg_parameter_tmp = np.vstack((
                parameter_return[0],
                parameter_return[1], parameter_return[2], parameter_return[3],
                parameter_return[4], parameter_return[5],
                parameter_return[6], parameter_return[7],
                parameter_return[8], parameter_return[9],
                parameter_return[10], parameter_return[11], parameter_return[12],
                parameter_return[13], parameter_return[14], parameter_return[15],
                parameter_return[16], parameter_return[17], parameter_return[18],
                parameter_return[19], parameter_return[20], parameter_return[21],
            )).T
            ecg_parameter = np.vstack((ecg_parameter, ecg_parameter_tmp))
            ecg_parameter = np.array(ecg_parameter, dtype='str')

            # 수정된 부분: 저장할 경로, 파라미터
            np.savetxt(r'C:\Users\jjm\Desktop\ooo1\0001\240404\1001_%s_%s_%d.csv' % (lead_i, 'parameter', j), ecg_parameter, fmt='%s', delimiter=",")

            ecg_parameter_vf = np.array([
                'time',
                'ps_vf', 'pe_vf', 'qrss_vf', 'qrse_vf', 'ts_vf', 'te_vf'
            ])
            ecg_parameter_vf_tmp = np.vstack((
                parameter_vf_return[0],
                parameter_vf_return[1], parameter_vf_return[2],
                parameter_vf_return[3], parameter_vf_return[4],
                parameter_vf_return[5], parameter_vf_return[6]
            )).T
            ecg_parameter_vf = np.vstack((ecg_parameter_vf, ecg_parameter_vf_tmp))
            ecg_parameter_vf = np.array(ecg_parameter_vf, dtype='str')

            # 수정된 부분: 저장할 경로, VF 파라미터
            np.savetxt(r'C:\Users\jjm\Desktop\ooo1\0001\240404\1001_%s_%s_%d.csv' % (lead_i, 'parameter_vf', j), ecg_parameter_vf, fmt='%s', delimiter=",")
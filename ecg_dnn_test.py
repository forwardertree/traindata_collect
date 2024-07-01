"""
신경망 평가
"""

import numpy as np
import os
from ecg_pkg.ecg_dnn import ECG_DNN_Thread

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
                'point_ps', 'point_pe', 'point_qrss', 'point_qrse', 'point_ts', 'point_te',
                'rs',
                'major_vf_ps', 'major_vf_pe', 'major_vf_qrss', 'major_vf_qrse', 'major_vf_ts', 'major_vf_te',
                'point_vf_ps', 'point_vf_pe', 'point_vf_qrss', 'point_vf_qrse', 'point_vf_ts', 'point_vf_te'
            ])
            ecg_point_tmp = np.vstack((
                ai_return[0],
                ai_return[1], ai_return[2], ai_return[3], ai_return[4], ai_return[5], ai_return[6],
                ai_return[7], ai_return[8], ai_return[9], ai_return[10], ai_return[11], ai_return[12],
                ai_return[13],
                ai_return[14], ai_return[15], ai_return[16], ai_return[17], ai_return[18], ai_return[19],
                ai_return[20], ai_return[21], ai_return[22], ai_return[23], ai_return[24], ai_return[25]
            )).T
            ecg_point = np.vstack((ecg_point, ecg_point_tmp))
            ecg_point = np.array(ecg_point, dtype='str')

            # 수정된 부분: 저장할 경로, 특이점
            np.savetxt(r'C:\Users\jjm\Desktop\ooo1\0001\240404\1001%s_%s_%d.csv' % (lead_i, 'point', j), ecg_point, fmt='%s', delimiter=",")

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
                parameter_return[19], parameter_return[20], parameter_return[21]
            )).T
            ecg_parameter = np.vstack((ecg_parameter, ecg_parameter_tmp))
            ecg_parameter = np.array(ecg_parameter, dtype='str')

            # 수정된 부분: 저장할 경로, 파라미터
            np.savetxt(r'C:\Users\jjm\Desktop\ooo1\0001\240404\1001%s_%s_%d.csv' % (lead_i, 'parameter', j), ecg_parameter, fmt='%s', delimiter=",")

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
            np.savetxt(r'C:\Users\jjm\Desktop\ooo1\0001\240404\1001%s_%s_%d.csv' % (lead_i, 'parameter_vf', j), ecg_parameter_vf, fmt='%s', delimiter=",")

            # # 'p_rs' 값이 10 이하인 경우를 확인하는 부분입니다.
            # low_rs_indices = [i for i, value in enumerate(parameter_return[1]) if float(value) <= 10]
            # print(low_rs_indices)
            #
            # # 'p_rs' 값이 10 이하인 경우에 대한 조건문입니다.
            # if low_rs_indices:
            #     # 'time' 값을 불러와서 'raw'의 행 시간을 계산합니다.
            #     raw_time = (np.array(parameter_return[0], dtype=float)[low_rs_indices] * 250).astype(int)
            #     print(raw_time)
            #
            #     # 'raw' 및 'rs' 열을 가진 배열을 생성합니다.
            #     ecg_raw_low_n = np.array(['raw', 'rs'])
            #     ecg_raw_low_v = np.vstack((ai_return[0], ai_return[13])).T
            #
            #     # 'ecg_raw_low_n'에 'raw' 및 'rs' 열을 추가합니다.
            #     ecg_raw_low_n = np.vstack((ecg_raw_low_n, ecg_raw_low_v))
            #     ecg_raw_low_n = np.array(ecg_raw_low_n, dtype='str')
            #     print(ecg_raw_low_n)
            #
            #     # 수정된 부분: 앞에 한 개 행과 뒤에 두 개 행 삭제
            #     for idx in raw_time:
            #         if idx >= 30 and idx + 50 < ecg_raw_low_n.shape[0]:
            #             ecg_raw_low_n = np.delete(ecg_raw_low_n, range(idx - 30, idx + 51), axis=0)
            #
            #     # 저장할 경로와 파일명을 지정합니다.
            #     save_path_point_low = r'C:\Users\jjm\Desktop\ecg_ai\aha_result_240221\%s_%s_low_%d.csv' % (
            #     lead_i, 'point', j)
            #
            #     # 'ecg_raw_low_n'을 CSV 파일로 저장합니다.
            #     np.savetxt(save_path_point_low, ecg_raw_low_n, fmt='%s', delimiter=",")



"""
박동별 파라미터 계산
"""

import numpy as np


def create_parameter_vf(ecg_parameter, raw, major_arr, point_arr):
    """
    박동별 파라미터를 계산하기 위한 모듈
    :param ecg_parameter: 박동별 파라미터 array
    :param raw: 심층신경망 판정된 ECG
    :param major_arr: 누적값 array
    :param point_arr: 최종판정값 array
    :return: ecg_parameter
            -> 파라미터별 저장 순서 주의해야 함
               time, p_rs, qrs_rs, ... 순서
    """
    major_ps, major_pe, major_qrss, major_qrse, major_ts, major_te = major_arr
    point_ps, point_pe, point_qrss, point_qrse, point_ts, point_te = point_arr

    # qrss_1 = qrs start 로 최종판정된 인덱스 저장
    # [0:-164] 인 이유는 [-164::] 범위는 신경망 판정, sum 누적, point 최종판정 구간이기 때문
    # qrss_1 앞뒤로 0을 append 하는 이유는 qrss_1[parm_i - 1], qrss_1[parm_i + 1] 등의 코드 때문
    qrss_1 = np.where(point_qrss[0:-164] == 1)[0]
    qrss_1 = np.append(0, qrss_1)
    qrss_1 = np.append(qrss_1, 0)

    # 현재 파라미터 연산이 완료된 박동 수
    beat_param_now = len(ecg_parameter)

    # parameter 계산 시, 현재 qrs 뿐만 아니라 앞 뒤 qrs 또한 이용하므로
    #   범위를 (1, len(qrss_1) - 1)로 설정함
    # len(qrss_1) - 2 이유 : 정확한 연산은 위해 마지막 박동에 대한 파라미터 연산은 진행하지 않음
    # 수정건: 굳이 len(qrss_1) - 1 대신 len(qrss_1) - 2로 설정하는 이유가 명확하지 않음
    for parm_i in range(1, len(qrss_1) - 2):
        '''
        Continue 조건
        '''
        # parm_i : 박동 순서, beat_param_now : 현재 파마리터 계산이 완료된 박동 수
        if parm_i - beat_param_now <= 0:
            continue

        '''
        현재 계산하려는 박동 위치에 해당하는 ps_i ~ te_i, post_qrss_i, post_ps_i 인덱스 탐색
        '''
        qrss_i = qrss_1[parm_i]

        '''
        심박동 파라미터 계산
        '''
        ps_vf = np.sum(major_ps[qrss_1[parm_i]:qrss_1[parm_i + 1]])
        pe_vf = np.sum(major_pe[qrss_1[parm_i]:qrss_1[parm_i + 1]])
        qrss_vf = np.sum(major_qrss[qrss_1[parm_i]:qrss_1[parm_i + 1]])
        qrse_vf = np.sum(major_qrse[qrss_1[parm_i]:qrss_1[parm_i + 1]])
        ts_vf = np.sum(major_ts[qrss_1[parm_i]:qrss_1[parm_i + 1]])
        te_vf = np.sum(major_te[qrss_1[parm_i]:qrss_1[parm_i + 1]])

        '''
        심박동 파라미터 저장
        '''
        beat_param_tmp = np.array([
            qrss_i / 250,
            ps_vf, pe_vf, qrss_vf, qrse_vf, ts_vf, te_vf
        ])

        ecg_parameter = np.vstack((ecg_parameter, beat_param_tmp))

        # print(f'ecg_parameter:{ecg_parameter}')

    return ecg_parameter

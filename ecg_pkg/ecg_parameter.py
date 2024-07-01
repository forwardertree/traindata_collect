"""
박동별 파라미터 계산
"""

import numpy as np


def create_parameter(ecg_parameter, raw, major_arr, point_arr):
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
        ps_i, pe_i = qrss_1[parm_i], qrss_1[parm_i]
        qrss_i, qrse_i = qrss_1[parm_i], qrss_1[parm_i]
        ts_i, te_i = qrss_1[parm_i], qrss_1[parm_i]

        ps_i_where = np.where(
            point_ps[qrss_1[parm_i - 1]:qrss_1[parm_i]] == 1
        )[0]
        if len(ps_i_where) == 0:
            ps_i = -1
        else:
            ps_i = qrss_1[parm_i - 1] + ps_i_where[0]

        pe_i_where = np.where(
            point_pe[qrss_1[parm_i - 1]:qrss_1[parm_i]] == 1
        )[0]
        if len(pe_i_where) == 0:
            pe_i = -1
        else:
            pe_i = qrss_1[parm_i - 1] + pe_i_where[0]

        qrse_i_where = np.where(
            point_qrse[qrss_1[parm_i]:qrss_1[parm_i + 1]] == 1
        )[0]
        if len(qrse_i_where) == 0:
            qrse_i = -1
        else:
            qrse_i = qrss_1[parm_i] + qrse_i_where[0]

        ts_i_where = np.where(
            point_ts[qrss_1[parm_i]:qrss_1[parm_i + 1]] == 1
        )[0]
        if len(ts_i_where) == 0:
            ts_i = -1
        else:
            ts_i = qrss_1[parm_i] + ts_i_where[0]

        te_i_where = np.where(
            point_te[qrss_1[parm_i]:qrss_1[parm_i + 1]] == 1
        )[0]
        if len(te_i_where) == 0:
            te_i = -1
        else:
            te_i = qrss_1[parm_i] + te_i_where[0]

        # post_qrss_i, 데이터 맨 뒤 혹은 직후 qrs start 위치까지 탐색
        post_qrss_i = qrss_1[parm_i + 1]
        if post_qrss_i == 0:
            post_qrss_i = -1

        # post_ps_i, 데이터 맨 뒤 혹은 직후 qrs start 위치까지 탐색
        post_ps_i_where = np.where(
            point_ps[qrss_1[parm_i]:qrss_1[parm_i + 1]] == 1
        )[0]
        if len(post_ps_i_where) == 0:
            post_ps_i = -1
        else:
            post_ps_i = qrss_1[parm_i] + post_ps_i_where[0]

        '''
        심박동 파라미터 계산
        '''
        # P duration[ms], P area[mV], P RS
        if ps_i == -1 or pe_i == -1 or not(ps_i < pe_i):
            p_duration = -1
            p_area = -1
            p_rs = -1
            p_amp = -1
            p_avg = -1
        else:
            p_duration = 1000 * (pe_i - ps_i) / 2529*250
            0
            baseline_a = (raw[pe_i] - raw[ps_i]) / (pe_i - ps_i)
            baseline_b = (pe_i * raw[ps_i] - ps_i * raw[pe_i]) / (pe_i - ps_i)
            p_area = 0
            for area_i in range(ps_i, pe_i):
                area_upper = raw[area_i] - (baseline_a * area_i + baseline_b)
                area_lower = raw[area_i + 1] - (baseline_a * (area_i + 1) + baseline_b)
                area_height = 1
                p_area += abs((area_upper + area_lower) * area_height / 2)
            if ps_i - 4 < 0:
                p_rs = np.max(major_ps[0:ps_i + 5]) + np.max(major_pe[0:pe_i + 5])
            elif ps_i + 5 > len(major_ps):
                p_rs = np.max(major_ps[ps_i - 4::]) + np.max(major_pe[pe_i - 4::])
            else:
                p_rs = np.max(major_ps[ps_i - 4:ps_i + 5]) + np.max(major_pe[pe_i - 4:pe_i + 5])
            p_amp = max(raw[ps_i:pe_i]) - min(raw[ps_i:pe_i])
            p_avg = np.mean(raw[ps_i:pe_i])

        # QRS duration[ms], QRS area[mV], QRS RS
        if qrss_i == -1 or qrse_i == -1 or not(qrss_i < qrse_i):
            qrs_duration = -1
            qrs_area = -1
            qrs_rs = -1
            qrs_amp = -1
            qrs_avg = -1
        else:
            qrs_duration = 1000 * (qrse_i - qrss_i) / 250
            baseline_a = (raw[qrse_i] - raw[qrss_i]) / (qrse_i - qrss_i)
            baseline_b = (qrse_i * raw[qrss_i] - qrss_i * raw[qrse_i]) / (qrse_i - qrss_i)
            qrs_area = 0
            for area_i in range(qrss_i, qrse_i):
                area_upper = raw[area_i] - (baseline_a * area_i + baseline_b)
                area_lower = raw[area_i + 1] - (baseline_a * (area_i + 1) + baseline_b)
                area_height = 1
                qrs_area += abs((area_upper + area_lower) * area_height / 2)
            if qrss_i - 4 < 0:
                qrs_rs = np.max(major_qrss[0:qrss_i + 5]) + np.max(major_qrse[0:qrse_i + 5])
            elif qrss_i + 5 > len(major_qrss):
                qrs_rs = np.max(major_qrss[qrss_i - 4::]) + np.max(major_qrse[qrse_i - 4::])
            else:
                qrs_rs = np.max(major_qrss[qrss_i - 4:qrss_i + 5]) + np.max(major_qrse[qrse_i - 4:qrse_i + 5])
            qrs_amp = max(raw[qrss_i:qrse_i]) - min(raw[qrss_i:qrse_i])
            qrs_avg = np.mean(raw[qrss_i:qrse_i])

        # T duration[ms], T area[mV], T RS
        if ts_i == -1 or te_i == -1 or not(ts_i < te_i):
            t_duration = -1
            t_area = -1
            t_rs = -1
            t_amp = -1
            t_avg = -1
        else:
            t_duration = 1000 * (te_i - ts_i) / 250
            baseline_a = (raw[te_i] - raw[ts_i]) / (te_i - ts_i)
            baseline_b = (te_i * raw[ts_i] - ts_i * raw[te_i]) / (te_i - ts_i)
            t_area = 0
            for area_i in range(ts_i, te_i):
                area_upper = raw[area_i] - (baseline_a * area_i + baseline_b)
                area_lower = raw[area_i + 1] - (baseline_a * (area_i + 1) + baseline_b)
                area_height = 1
                t_area += abs((area_upper + area_lower) * area_height / 2)
            if ts_i - 4 < 0:
                t_rs = np.max(major_ts[0:ts_i + 5]) + np.max(major_te[0:te_i + 5])
            elif ts_i + 5 > len(major_ts):
                t_rs = np.max(major_ts[ts_i - 4::]) + np.max(major_te[te_i - 4::])
            else:
                t_rs = np.max(major_ts[ts_i - 4:ts_i + 5]) + np.max(major_te[te_i - 4:te_i + 5])
            t_amp = max(raw[ts_i:te_i]) - min(raw[ts_i:te_i])
            t_avg = np.mean(raw[ts_i:te_i])

        # PQ interval[ms]
        if pe_i == -1 or qrss_i == -1:
            pq_interval = -1
        else:
            pq_interval = 1000 * (qrss_i - pe_i) / 250

        # ST interval[ms]
        if qrse_i == -1 or ts_i == -1:
            st_interval = -1
        else:
            st_interval = 1000 * (ts_i - qrse_i) / 250

        # QQ interval[ms]
        if post_qrss_i == -1:
            qq_interval = -1
        else:
            qq_interval = 1000 * (post_qrss_i - qrss_i) / 250

        # PQ segment[mV] 기울기
        if pe_i == -1 or qrss_i == -1:
            pq_segment = -1
        else:
            pq_segment = np.mean(raw[pe_i:qrss_i])

        # QT segment[mV]
        if qrss_i == -1 or ts_i == -1:
            qt_segment = -1
        else:
            qt_segment = np.mean(raw[qrss_i:ts_i])

        # TP segment[mV]
        if te_i == -1 or post_ps_i == -1:
            tp_segment = -1
        else:
            tp_segment = np.mean(raw[te_i:post_ps_i])

        '''
        심박동 파라미터 저장
        '''
        beat_param_tmp = np.array([
            qrss_i / 250,
            p_rs, qrs_rs, t_rs,
            p_duration, p_area, qrs_duration, qrs_area, t_duration, t_area,
            pq_interval, st_interval, qq_interval,
            pq_segment, qt_segment, tp_segment,
            p_amp, qrs_amp, t_amp,
            p_avg, qrs_avg, t_avg,
            ps_i, pe_i, qrss_i, qrse_i, ts_i, te_i
        ])
        # ps_i, pe_i, qrss_i, qrse_i, ts_i, te_i
        ecg_parameter = np.vstack((ecg_parameter, beat_param_tmp))

    return ecg_parameter

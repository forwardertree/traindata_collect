"""
특이점 최종 판정
"""

import numpy as np


def define_point(raw_arr, major_arr, point_arr, major_vf_arr, point_vf_arr):
    """
    판정구간을 통과한 데이터에 대해 최종판정을 하기 위한 모듈
    :param raw_arr: 심층신경망 판정된 ECG
    :param major_arr: 누적값 array
    :param point_arr: 최종판정값 array
    :param major_vf_arr: vf 누적값 array
    :param point_vf_arr: vf 최종판정값 array
    :return: point_ps, point_pe, point_qrss, point_qrse, point_ts, point_te
    """
    major_ps, major_pe, major_qrss, major_qrse, major_ts, major_te = major_arr
    point_ps, point_pe, point_qrss, point_qrse, point_ts, point_te = point_arr
    major_vf_ps, major_vf_pe, major_vf_qrss, major_vf_qrse, major_vf_ts, major_vf_te = major_vf_arr
    point_vf_ps, point_vf_pe, point_vf_qrss, point_vf_qrse, point_vf_ts, point_vf_te = point_vf_arr

    '''
    [i - 8:i + 12] 설명
    - 방향: 현재 위치를 기준으로 전후 2칸씩의 major 값을 비교하려고 함
    - 상황: 4분주로 인해 생긴 네 개의 배열 중 첫 배열만 AI를 적용함
            major 함수 생성 시, 4분주된 4개의 배열을 1개의 배열로 합치면서 원소 4칸 간격으로 major 값이 저장된 상황
    - 결론: 전후 2칸씩 보려면 (-8)000(-4)000(현재)000(4)000(8)000(12) 까지 검토해야 하므로
            범위를 [i - 8:i + 12]으로 설정함
    [i:i + 5]) 설명
    - 방향: major 원소 중 임계값(5 or 2)보다 큰 인덱스를 기준으로
            일정 범위 내에 ecg 데이터가 최소인 지점을 특이점으로 정의하고자 함
    - 상황: 기존 4분주된 4개의 배열에서 판정된 인덱스 중 하나의 인덱스만 선택해야 하지만, 현재 첫 분주만 AI를 적용한 상황
    - 결론: 2~4번째 분주를 고려하면 임계값보다 큰 인덱스 i를 기준으로 i+4까지 범위만 검토하면 되므로
            i부터 i+4까지 인덱스 범위만 검토함
    '''
    def examine_condition(i, raw_, major_, point_):
        if 5 < major_[i] == np.max(major_[i - 8:i + 12]) and np.sum(point_[i - 10:i]) == 0:
            point_[i + np.argmin(raw_[i:i + 5])] = 1
        return point_

    def examine_condition_p(i, raw_, major_, point_):
        if 5 < major_[i] == np.max(major_[i - 8:i + 12]) and np.sum(point_[i - 10:i]) == 0:
            point_[i + np.argmin(raw_[i:i + 5])] = 1
        return point_

    def examine_condition_vf(i, raw_, major_, point_):
        if 5 < np.sum(major_[i - 20:i + 20]) and np.sum(point_[i - 20:i]) == 0:
            point_[i] = 1
        return point_

    for point_i in range(-164, -160):  # 250Hz ECG 기준) 판정범위 [-160::]를 범위를 벗어나면 더 이상 값이 누적되지 않음
        point_ps = examine_condition_p(point_i, raw_arr, major_ps, point_ps)
        point_pe = examine_condition_p(point_i, raw_arr, major_pe, point_pe)
        point_qrss = examine_condition(point_i, raw_arr, major_qrss, point_qrss)
        point_qrse = examine_condition(point_i, raw_arr, major_qrse, point_qrse)
        point_ts = examine_condition(point_i, raw_arr, major_ts, point_ts)
        point_te = examine_condition(point_i, raw_arr, major_te, point_te)

        point_vf_ps = examine_condition_vf(point_i, raw_arr, major_vf_ps, point_vf_ps)
        point_vf_pe = examine_condition_vf(point_i, raw_arr, major_vf_pe, point_vf_pe)
        point_vf_qrss = examine_condition_vf(point_i, raw_arr, major_vf_qrss, point_vf_qrss)
        point_vf_qrse = examine_condition_vf(point_i, raw_arr, major_vf_qrse, point_vf_qrse)
        point_vf_ts = examine_condition_vf(point_i, raw_arr, major_vf_ts, point_vf_ts)
        point_vf_te = examine_condition_vf(point_i, raw_arr, major_vf_te, point_vf_te)

    return point_ps, point_pe, point_qrss, point_qrse, point_ts, point_te, point_vf_ps, point_vf_pe, point_vf_qrss, point_vf_qrse, point_vf_ts, point_vf_te


def run_sum(raw_arr, major_arr, point_arr, major_vf_arr, point_vf_arr):
    """
    누적값을 기준으로 특이점을 최종판정하기 위한 모듈
    :param raw_arr: 심층신경망 판정된 ECG
    :param major_arr: 누적값 array
    :param point_arr: 최종판정값 array
    :param major_vf_arr: vf 누적값 array
    :param point_vf_arr: vf 최종판정값 array
    :return: point_ps, point_pe, point_qrss, point_qrse, point_ts, point_te
    """

    major_ps, major_pe, major_qrss, major_qrse, major_ts, major_te = major_arr
    point_ps, point_pe, point_qrss, point_qrse, point_ts, point_te = point_arr
    major_vf_ps, major_vf_pe, major_vf_qrss, major_vf_qrse, major_vf_ts, major_vf_te = major_vf_arr
    point_vf_ps, point_vf_pe, point_vf_qrss, point_vf_qrse, point_vf_ts, point_vf_te = point_vf_arr

    if len(point_ps) == 0:
        # 최초 실행 시, 0으로 구성된 array로 생성
        len_n = len(major_ps)
        point_ps, point_pe = np.zeros(len_n), np.zeros(len_n)
        point_qrss, point_qrse = np.zeros(len_n), np.zeros(len_n)
        point_ts, point_te = np.zeros(len_n), np.zeros(len_n)

        point_vf_ps, point_vf_pe = np.zeros(len_n), np.zeros(len_n)
        point_vf_qrss, point_vf_qrse = np.zeros(len_n), np.zeros(len_n)
        point_vf_ts, point_vf_te = np.zeros(len_n), np.zeros(len_n)

        point = define_point(
            raw_arr,
            major_arr,
            (point_ps, point_pe, point_qrss, point_qrse, point_ts, point_te),
            major_vf_arr,
            (point_vf_ps, point_vf_pe, point_vf_qrss, point_vf_qrse, point_vf_ts, point_vf_te)
        )

        point_ps, point_pe, point_qrss, point_qrse, point_ts, point_te = point[0:6]
        point_vf_ps, point_vf_pe, point_vf_qrss, point_vf_qrse, point_vf_ts, point_vf_te = point[6::]

    else:
        # 최초 실행 이후 길이가 4만큼씩 증가함
        point_ps, point_pe = np.append(point_ps, np.zeros(4)), np.append(point_pe, np.zeros(4))
        point_qrss, point_qrse = np.append(point_qrss, np.zeros(4)), np.append(point_qrse, np.zeros(4))
        point_ts, point_te = np.append(point_ts, np.zeros(4)), np.append(point_te, np.zeros(4))

        point_vf_ps, point_vf_pe = np.append(point_vf_ps, np.zeros(4)), np.append(point_vf_pe, np.zeros(4))
        point_vf_qrss, point_vf_qrse = np.append(point_vf_qrss, np.zeros(4)), np.append(point_vf_qrse, np.zeros(4))
        point_vf_ts, point_vf_te = np.append(point_vf_ts, np.zeros(4)), np.append(point_vf_te, np.zeros(4))

        point = define_point(
            raw_arr,
            major_arr,
            (point_ps, point_pe, point_qrss, point_qrse, point_ts, point_te),
            major_vf_arr,
            (point_vf_ps, point_vf_pe, point_vf_qrss, point_vf_qrse, point_vf_ts, point_vf_te)
        )

        point_ps, point_pe, point_qrss, point_qrse, point_ts, point_te = point[0:6]
        point_vf_ps, point_vf_pe, point_vf_qrss, point_vf_qrse, point_vf_ts, point_vf_te = point[6::]

    return point_ps, point_pe, point_qrss, point_qrse, point_ts, point_te, point_vf_ps, point_vf_pe, point_vf_qrss, point_vf_qrse, point_vf_ts, point_vf_te

"""
데이터 전처리
"""

import numpy as np


def differential(arr, period):
    """
    1-D 데이터(arr)를 주기(period)에 맞게 미분하여 반환하는 모듈
    :param arr: 1차원 데이터
    :param period: 주기(unit=second)
    :return: outp
            -> arr 미분값
    """
    # arr = raw , outp = diff
    outp = np.empty_like(arr)
    outp[1:] = (arr[1:] - arr[:-1]) / period
    outp[0] = 0
    return outp


def pre_process(ecg_, ref_i):
    """
    신경망 입력데이터 전처리를 위한 모듈
    :param ecg_: 1차원 ECG 데이터
    :param ref_i: n/a
    :return: train_inp
            -> type=array, len=160
               ecg(60개) + tompk(60개) + ecg_max(20개) + tompk_min(20개)
    """

    '''
    ECG 표준화 및 사용할 범위 설정정
   '''
    ecg_max_val = max(ecg_)
    ecg_min_val = min(ecg_)
    ecg_mean = sum(ecg_) / len(ecg_)
    if ecg_max_val == ecg_min_val:
        ecg_max_val = 1
        ecg_min_val = 0
    ecg_ = (ecg_ - ecg_mean) / (ecg_max_val - ecg_mean)
    ecg_[ecg_ > 2] = 2
    ecg_[ecg_ < -1] = -2
    ecg = np.array(ecg_[60::])  # 표준화를 위해 [i - 60:i]를 포함시켰으므로, 실제 사용할 범위만 재설정

    tompk_ = differential(arr=ecg_, period=1 / 62.5) ** 2  # tompkins = 미분 + 제곱
    tompk_max_val = max(tompk_)
    tompk_min_val = min(tompk_)
    tompk_mean = sum(tompk_) / len(tompk_)
    if tompk_max_val == tompk_min_val:
        tompk_max_val = 1
        tompk_min_val = 0
    tompk_ = (tompk_ - tompk_min_val) / (tompk_max_val - tompk_min_val)
    tompk_[tompk_ > 2] = 2
    tompk_[tompk_ < -1] = -2
    tompk = np.array(tompk_[60::])  # 표준화를 위해 [i - 60:i]를 포함시켰으므로, 실제 사용할 범위만 재설정

    tompk_max = np.zeros(20)
    ref = np.argmax(tompk[20:40])  # 특이점 판정구간 [20:40] 내 max 값 탐색
    # max 조건: 앞, 뒤 10칸씩의 범위 내 최대 and 해당 값 0.7 이상
    if tompk[20 + ref] == max(tompk[10 + ref:30 + ref]) and np.sum(tompk_max) == 0:
        tompk_max[ref] = 1

    ecg_max = np.zeros(20)
    ref = np.argmax(ecg[20:40])  # 특이점 판정구간 [20:40] 내 max 값 탐색
    # max 조건: 앞, 뒤 10칸씩의 범위 내 최대 and 해당 값 0.7 이상
    # R-peak를 포함하여 표준화 했으므로, 대부분의 경우 R-peak 위치는 0.7보다 큼
    if ecg[20 + ref] == max(ecg[10 + ref:30 + ref]) and np.sum(tompk_max[ref-3:ref+3]) > 0 and np.sum(ecg_max) == 0:
        ecg_max[ref] = 1
    # ref = np.argmin(ecg[20:40])
    if ecg[20 + ref] == min(ecg[10 + ref:30 + ref]) and np.sum(ecg_max) == 0:
        ecg_max[ref] = -1
    # ecg(60개) + tompk(60개) + ecg_max(20개) + tompk_min(20개)
    train_inp = np.hstack((
        ecg, tompk, ecg_max, tompk_max
    ))

    return train_inp
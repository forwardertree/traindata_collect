"""
신경망 판정값 누적
"""

import numpy as np


def add_major(major, outp):
    """
    판정구간 범위 내에서, 기존 누적값에 새롭게 누적할 값을 더하기 위한 모듈
    :param major: 기존 누적값
    :param outp: 새롭게 누적할 값
    :return: major
            -> 길이가 4씩 증가하는 array
    """
    major = np.append(major, np.zeros(4))  # 250Hz ECG 기준 4칸씩(62.5Hz ECG 기준 1칸씩) 이동하므로, 4씩 길이 증가
    # 250Hz 기준) 240개 길이의 ECG에서, 심층신경망 판정구간은 뒤에서 160번째부터 뒤에서 80번째까지
    # 62.5Hz 기준) 60개 길이의 ECG에서, 심층신경망 판정구간은 뒤에서 40번째부터 뒤에서 20번째까지
    major[-160:-80] = major[-160:-80] + outp
    return major


def add_major_vf(major, outp):
    """
    판정구간 범위 내에서, 기존 누적값에 새롭게 누적할 값을 더하기 위한 모듈
    :param major: 기존 누적값
    :param outp: 새롭게 누적할 값
    :return: major
            -> 길이가 4씩 증가하는 array
    """
    major = np.append(major, major[-1])
    major = np.append(major, major[-1])
    major = np.append(major, major[-1])
    major = np.append(major, major[-1] + outp[-1])
    return major


def run_major(i, dnn_outp, major, dnn_outp_vf, major_vf):
    """
    심층신경망 판정결과를 누적시키기 위한 모듈
    :param i: 현재까지 window가 이동한 수(62.5Hz 데이터에 대해 1칸씩 증가함)
    :param dnn_outp: 심층신경망 판정결과(=apply_dnn 모듈 출력값)
    :param major: 기존 누적값
    :param dnn_outp_vf: 심층신경망 vf 판정결과(=apply_dnn 모듈 출력값)
    :param major_vf: 기존 vf 누적값
    :return: raw_arr, major_ps, major_pe, major_qrss, major_qrse, major_ts, major_te
            -> 순서대로 (심층신경망 판정된 ECG, ps 누적값, ...) 에 대한 array
    """

    # raw = ecg_dnn.py 실행 시 입력한 250Hz ECG 데이터
    # outp_xx = 새롭게 누적할 값
    # raw_arr = 현재까지 심층신경망 판정이 진행된 ECG 범위
    # major_xx = 존 누적되어 있는 값
    # outp_vf_xx = 새롭게 누적할 vf 값
    # major_vf_xx = 기존 누적되어 있는 vf 값
    raw, outp_ps, outp_pe, outp_qrss, outp_qrse, outp_ts, outp_te = dnn_outp
    raw_arr, major_ps, major_pe, major_qrss, major_qrse, major_ts, major_te = major
    outp_vf_ps, outp_vf_pe, outp_vf_qrss, outp_vf_qrse, outp_vf_ts, outp_vf_te = dnn_outp_vf
    major_vf_ps, major_vf_pe, major_vf_qrss, major_vf_qrse, major_vf_ts, major_vf_te = major_vf

    if len(raw_arr) == 0:
        # 최초 실행 시, 표준화를 위해 가져온 앞 240개의 ECG 데이터 범위를 포함하여 데이터를 생성함
        # 실제 심층신경망에 사용한 범위는 뒤 240개
        raw_arr = raw[0:240 + i * 4]  # len = 480, i는 60부터 시작
        major_ps = np.hstack((np.zeros(320), outp_ps, np.zeros(80)))  # len = 480
        major_pe = np.hstack((np.zeros(320), outp_pe, np.zeros(80)))
        major_qrss = np.hstack((np.zeros(320), outp_qrss, np.zeros(80)))
        major_qrse = np.hstack((np.zeros(320), outp_qrse, np.zeros(80)))
        major_ts = np.hstack((np.zeros(320), outp_ts, np.zeros(80)))
        major_te = np.hstack((np.zeros(320), outp_te, np.zeros(80)))

        # major_vf_ps = np.hstack((np.zeros(320), outp_vf_ps, np.zeros(80)))
        # major_vf_pe = np.hstack((np.zeros(320), outp_vf_pe, np.zeros(80)))
        # major_vf_qrss = np.hstack((np.zeros(320), outp_vf_qrss, np.zeros(80)))
        # major_vf_qrse = np.hstack((np.zeros(320), outp_vf_qrse, np.zeros(80)))
        # major_vf_ts = np.hstack((np.zeros(320), outp_vf_ts, np.zeros(80)))
        # major_vf_te = np.hstack((np.zeros(320), outp_vf_te, np.zeros(80)))

        major_vf_ps = np.hstack((np.zeros(400), outp_vf_ps))
        major_vf_pe = np.hstack((np.zeros(400), outp_vf_pe))
        major_vf_qrss = np.hstack((np.zeros(400), outp_vf_qrss))
        major_vf_qrse = np.hstack((np.zeros(400), outp_vf_qrse))
        major_vf_ts = np.hstack((np.zeros(400), outp_vf_ts))
        major_vf_te = np.hstack((np.zeros(400), outp_vf_te))

    else:
        # 최초 실행 이후 길이가 4만큼씩 증가함
        raw_arr = raw[0:240 + i * 4]  # 윈도우는 62.5Hz ECG 데이터 위에서 1씩 움직이므로 250Hz 기준으로는 4씩 증가해야 함
        major_ps = add_major(major_ps, outp_ps)
        major_pe = add_major(major_pe, outp_pe)
        major_qrss = add_major(major_qrss, outp_qrss)
        major_qrse = add_major(major_qrse, outp_qrse)
        major_ts = add_major(major_ts, outp_ts)
        major_te = add_major(major_te, outp_te)

        major_vf_ps = add_major_vf(major_vf_ps, outp_vf_ps)
        major_vf_pe = add_major_vf(major_vf_pe, outp_vf_pe)
        major_vf_qrss = add_major_vf(major_vf_qrss, outp_vf_qrss)
        major_vf_qrse = add_major_vf(major_vf_qrse, outp_vf_qrse)
        major_vf_ts = add_major_vf(major_vf_ts, outp_vf_ts)
        major_vf_te = add_major_vf(major_vf_te, outp_vf_te)

    return raw_arr, major_ps, major_pe, major_qrss, major_qrse, major_ts, major_te, major_vf_ps, major_vf_pe, major_vf_qrss, major_vf_qrse, major_vf_ts, major_vf_te

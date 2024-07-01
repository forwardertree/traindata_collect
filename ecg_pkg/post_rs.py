"""
박동별 RS 계산
"""

import numpy as np


def run_rs(recog_score, point_qrss, major_arr, rs_ref):
    """
    박동별 rs를 계산하기 위한 모듈
    :param recog_score: 인식률 array
    :param point_qrss: qrss 최종판정값 array
    :param major_arr: 누적값 array
    :param rs_ref: 현재 rs가 계산된 심박동 수
    :return: recog_score, rs_ref
    """
    major_ps, major_pe, major_qrss, major_qrse, major_ts, major_te = major_arr

    if len(recog_score) == 0:
        # 최초 실행 시, 0으로 구성된 array로 생성
        recog_score = np.zeros(len(point_qrss))
    else:
        # 최초 실행 이후 길이가 4만큼씩 증가함
        recog_score = np.append(recog_score, np.zeros(4))

    # 수정건) run_sum 모듈에서 point_xx 판정범위는 (-164, -160)이므로 [0:-80] 대신 [0:-160] 으로 수정해야 함
    qrss_1 = np.where(point_qrss[0:-80] == 1)[0]

    # 새롭게 point_qrss == 1인 인덱스가 탐색되고 and  point_qrss == 1인 인덱스가 3개 이상인 경우 연산 진행
    # point_qrss == 1인 인덱스가 3개 이상인 경우를 탐색하는 이유는 RS 계산에 이전, 현재, 이후 심박동정보를 사용하기 떄문
    if (len(qrss_1) > rs_ref) and (len(qrss_1) >= 3):
        rs_ref = len(qrss_1)  # 현재 rs가 계산된 심박동 수 갱신

        rs_before_i, rs_now_i, rs_after_i = qrss_1[-3], qrss_1[-2], qrss_1[-1]  # 이전, 현재, 이후 심박동 정보

        # 수정건) if 조건 있을 필요가 있나 검토해야 함
        if (rs_now_i - rs_before_i) / 250 < 3 and (rs_after_i - rs_now_i) / 250 < 3:
            # ps ~ te까지 범위 내 최대 누적값을 찾아서 모두 더함
            recog_score[rs_now_i] = major_qrss[rs_now_i] + \
                                         np.max(major_ps[rs_before_i:rs_now_i]) + \
                                         np.max(major_pe[rs_before_i:rs_now_i]) + \
                                         np.max(major_qrse[rs_now_i:rs_after_i]) + \
                                         np.max(major_ts[rs_now_i:rs_after_i]) + \
                                         np.max(major_te[rs_now_i:rs_after_i])

    return recog_score, rs_ref

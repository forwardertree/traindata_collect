"""
신경망 적용
"""

import numpy as np


def DNN(x, h1_w, h1_b, h2_w, h2_b, o_w, o_b):
    """
    심층신경망 알고리즘 구현을 위한 모듈
    :param x: 심층신경망 입력데이터
    :param h1_w: 1번 은닉층 Weight
    :param h1_b: 1번 은닉층 bias
    :param h2_w: 2번 은닉층 Weight
    :param h2_b: 2번 은닉층 bias
    :param o_w: 출력층 Weight
    :param o_b: 출력층 bias
    :return: outp
            -> 1차원 array
    """

    z1 = np.matmul(x, h1_w) + h1_b
    z1 = np.array(z1.reshape(z1.shape[0] * z1.shape[1]))  # 연산과정에서 2차원으로 변형될 경우, 데이터 1차원으로 변경
    z1[z1 < 0] = 0  # ReLu 함수 구현
    z2 = np.matmul(z1, h2_w) + h2_b
    z2 = np.array(z2.reshape(z2.shape[0] * z2.shape[1]))
    z2[z2 < 0] = 0
    z = np.matmul(z2, o_w) + o_b
    z = np.array(z.reshape(z.shape[0] * z.shape[1]))
    outp = np.zeros(len(z))
    outp[np.argmax(z)] = 1  # Softmax 함수 구현

    return outp


def apply_dnn(weight, bias, train_inp0):
    """
    심층신경망 적용을 적용하기 위한 모듈
    :param weight: weight array 18개
    :param bias: bias array 18개
    :param train_inp0: 심층신경망 입력데이터
            -> ecg(60개) + tompk(60개) + ecg_max(20개) + tompk_min(20개)
    :return: (outp_ps, outp_pe, outp_qrss, outp_qrse, outp_ts, outp_te)
            -> 각각 80개 길이의 array
               80개 = 20개 * 4분주, 62.5Hz Sampling을 250Hz로 up-sampling 하기 위함
    """

    w1, w2, w3, w4, w5, w6, w7, w8, w9, w10, w11, w12, w13, w14, w15, w16, w17, w18 = weight
    b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, b16, b17, b18 = bias

    outp_ps_ = DNN(train_inp0, w1, b1, w2, b2, w3, b3)
    outp_pe_ = DNN(train_inp0, w4, b4, w5, b5, w6, b6)
    outp_qrss_ = DNN(train_inp0, w7, b7, w8, b8, w9, b9)
    outp_qrse_ = DNN(train_inp0, w10, b10, w11, b11, w12, b12)
    outp_ts_ = DNN(train_inp0, w13, b13, w14, b14, w15, b15)
    outp_te_ = DNN(train_inp0, w16, b16, w17, b17, w18, b18)

    outp_ps, outp_pe = np.zeros(80), np.zeros(80)
    outp_qrss, outp_qrse = np.zeros(80), np.zeros(80)
    outp_ts, outp_te = np.zeros(80), np.zeros(80)

    # 62.5Hz ECG 판독결과를 down-sampling 하기 전인 250Hz ECG에 적용하기 위해, 4칸마다 0을 3개씩 추가함
    # [0:-1] : "특이점 없음"을 의미하는 신경망 출력층의 마지막 뉴련(21번째)을 판정결과에서 제외함
    outp_ps[::4] = outp_ps_[0:-2]
    outp_pe[::4] = outp_pe_[0:-2]
    outp_qrss[::4] = outp_qrss_[0:-2]
    outp_qrse[::4] = outp_qrse_[0:-2]
    outp_ts[::4] = outp_ts_[0:-2]
    outp_te[::4] = outp_te_[0:-2]

    outp_vf_ps, outp_vf_pe = np.zeros(80), np.zeros(80)
    outp_vf_qrss, outp_vf_qrse = np.zeros(80), np.zeros(80)
    outp_vf_ts, outp_vf_te = np.zeros(80), np.zeros(80)

    # 21번째 뉴런([-1])이 의미하는 vf 판정결과를 위 outp_xx 과 동일한 방식으로 누적시키기 위함
    outp_vf_ps[-1] = outp_ps_[-1]
    outp_vf_pe[-1] = outp_pe_[-1]
    outp_vf_qrss[-1] = outp_qrss_[-1]
    outp_vf_qrse[-1] = outp_qrse_[-1]
    outp_vf_ts[-1] = outp_ts_[-1]
    outp_vf_te[-1] = outp_te_[-1]

    return outp_ps, outp_pe, outp_qrss, outp_qrse, outp_ts, outp_te, \
           outp_vf_ps, outp_vf_pe, outp_vf_qrss, outp_vf_qrse, outp_vf_ts, outp_vf_te

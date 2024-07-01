"""
신경망 학습모델 호출
"""

import pickle


def call_model(model_path):
    """
    신경망 학습모델 호출을 위한 모듈
    :param model_path: 저장경로, ex) './ecgrdvq_ai_model_231117.pickle'
    :return: (weight, bias)
        weight -> type=tuple, len=18
                  array 18개로 구성, (160, 100), (100, 50), (50, 21) 가 6개 있음
        bias   -> type=tuple, len=18
                  array 18개로 구성, (1, 100), (1, 50), (1, 21) 가 6개 있음
    """
    with open('%s' % model_path, 'rb') as f:
        # dnn model, type=dict
        # key = ['ps', 'pe', 'qrss', 'qrse', 'ts', 'te']
        # 각 key의 value는 동일하게 weight 3개와 bias 3개로 구성됨
        dnn_model = pickle.load(f)

    ps_model = dnn_model['ps']
    pe_model = dnn_model['pe']
    qrss_model = dnn_model['qrss']
    qrse_model = dnn_model['qrse']
    ts_model = dnn_model['ts']
    te_model = dnn_model['te']

    weight = (
        ps_model['w1'], ps_model['w2'], ps_model['w3'],
        pe_model['w4'], pe_model['w5'], pe_model['w6'],
        qrss_model['w7'], qrss_model['w8'], qrss_model['w9'],
        qrse_model['w10'], qrse_model['w11'], qrse_model['w12'],
        ts_model['w13'], ts_model['w14'], ts_model['w15'],
        te_model['w16'], te_model['w17'], te_model['w18']
    )

    bias = (
        ps_model['b1'], ps_model['b2'], ps_model['b3'],
        pe_model['b4'], pe_model['b5'], pe_model['b6'],
        qrss_model['b7'], qrss_model['b8'], qrss_model['b9'],
        qrse_model['b10'], qrse_model['b11'], qrse_model['b12'],
        ts_model['b13'], ts_model['b14'], ts_model['b15'],
        te_model['b16'], te_model['b17'], te_model['b18']
    )

    return weight, bias

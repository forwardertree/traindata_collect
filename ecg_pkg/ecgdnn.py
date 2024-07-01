from ecg_pkg.pre_processing_v1_1 import pre_process
from ecg_pkg.dnn_model import call_model
from ecg_pkg.dnn_application_v1_1 import apply_dnn
from ecg_pkg.dnn_major import run_major
from ecg_pkg.post_sum import run_sum
from ecg_pkg.post_rs import run_rs
from ecg_pkg.ecg_parameter import create_parameter
from ecg_pkg.ecg_parameter_vf import create_parameter_vf
import numpy as np
from datetime import datetime
import pickle


class ECG_DNN_Thread:

    def __init__(self, raw, sampling):
        super().__init__()
        self.raw = np.array(raw, dtype='float32')  # list로 들어온 raw 데이터를 array로 변환 + 문자형을 실수형으로 변환
        self.sampling = float(sampling)  # 62.5Hz 등을 고려하여 sampling을 float 형으로 변환

        self.raw_arr = np.array([])  # AI 판정이 진행된 ECG 범위 저장

        self.major_ps, self.point_ps = np.array([]), np.array([])  # major = 누적값 array, point = 최종판정값 array
        self.major_pe, self.point_pe = np.array([]), np.array([])
        self.major_qrss, self.point_qrss = np.array([]), np.array([])
        self.major_qrse, self.point_qrse = np.array([]), np.array([])
        self.major_ts, self.point_ts = np.array([]), np.array([])
        self.major_te, self.point_te = np.array([]), np.array([])

        self.major_vf_ps, self.point_vf_ps = np.array([]), np.array([])
        self.major_vf_pe, self.point_vf_pe = np.array([]), np.array([])
        self.major_vf_qrss, self.point_vf_qrss = np.array([]), np.array([])
        self.major_vf_qrse, self.point_vf_qrse = np.array([]), np.array([])
        self.major_vf_ts, self.point_vf_ts = np.array([]), np.array([])
        self.major_vf_te, self.point_vf_te = np.array([]), np.array([])

        self.recog_score = np.array([])  # 인식률 array
        self.rs_ref = 0  # 현재 rs가 계산된 심박동 수

        # ecg parameter
        self.ecg_parameter = np.array([
            'time',
            'p_rs', 'qrs_rs', 't_rs',
            'p_duration', 'p_area', 'qrs_duration', 'qrs_area', 't_duration', 't_area',
            'pq_interval', 'st_interval', 'qq_interval',
            'pq_segment', 'qt_segment', 'tp_segment',
            'p_amp', 'qrs_amp', 't_amp',
            'p_avg', 'qrs_avg', 't_avg',
            'ps_i', 'pe_i', 'qrss_i', 'qrse_i', 'ts_i', 'te_i'
        ])
        self.ecg_parameter = np.reshape(self.ecg_parameter, (1, len(self.ecg_parameter)))  # vstack을 위한 reshape

        # ecg parameter vf
        self.ecg_parameter_vf = np.array([
            'time',
            'ps_vf', 'pe_vf', 'qrss_vf', 'qrse_vf', 'ts_vf', 'te_vf'
        ])
        self.ecg_parameter_vf = np.reshape(self.ecg_parameter_vf, (1, len(self.ecg_parameter_vf)))  # vstack을 위한 reshape

        # self.dnn_model_path = './ecgrdvq_ai_model_231213.pickle'  # 학습모델 저장경로
        self.dnn_model_path = './ecgrdvq_ai_model_240404.pickle'  # 학습모델 저장경로

        self.stack_arr = np.array([])

    # a = 0x61
    def run_dnn(self):
        """
        심층신경망 실행
        :return: (ai_return, parameter_return)
                -> ai_return: ecg, major_ps ~ major_te, point_ps ~ point_te, rs로 구성
                   parameter_return: time, p_rs, qrs_rs, ... ,qrs_avg, t_avg 순으로 저장
        """
        start_datetime = datetime.now()
        # formatted_start = start_datetime.strftime("%y.%m.%d %H:%M:%S")

        '''
        학습모델 불러오기
        '''
        # weight -> type=tuple, len=18
        #           array 18개로 구성, (160, 100), (100, 50), (50, 21) 가 6개 있음
        # bias   -> type=tuple, len=18
        #           array 18개로 구성, (1, 100), (1, 50), (1, 21) 가 6개 있음
        weight, bias = call_model(self.dnn_model_path)

        """
        입력데이터 분주
        """
        def NDivision(arr, n):
            outp = np.array([])
            for div_i in range(0, len(arr)):
                if div_i % n == 0:
                    outp = np.append(outp, arr[div_i])
            return outp

        # ECG Sampling을 62.5Hz로 맞추기 위함, 자사 장비로 사용할 경우 250Hz 고정
        ecg = NDivision(self.raw, int(self.sampling / 62.5))

        '''
        학습데이터 가공 및 학습모델 적용 시작
        '''
        for i in range(0, len(ecg)):

            print(f'i:{i}, total:{len(ecg)}')

            '''
            계속조건
            '''
            if i < 60:  # 사용하는 ecg 데이터 범위 = [i - 60:i + 60]
                continue

            '''
            종료조건
            '''
            if i + 60 > len(ecg):  # 사용하는 ecg 데이터 범위 = [i - 60:i + 60]
                break

            '''
            현재 판정값
            '''
            # 범위에 [i - 60:i]를 포함시킨 이유: ecg 표준화 시, R-paek를 포함시키기 위함
            y1 = ecg[i - 60:i + 60]

            '''
            데이터 전처리
            '''
            # train_inp = ecg(60개) + tompk(60개) + ecg_max(20개) + tompk_min(20개)
            train_inp = pre_process(y1, i)

            '''
            DNN 적용
            '''
            # 각각 길이 80개의 array
            outp = apply_dnn(weight, bias, train_inp)
            outp_ps, outp_pe, outp_qrss, outp_qrse, outp_ts, outp_te = outp[0:6]
            outp_vf_ps, outp_vf_pe, outp_vf_qrss, outp_vf_qrse, outp_vf_ts, outp_vf_te = outp[6::]

            '''
            입력데이터 & 출력데이터 쌓기
            '''
            stack_tmp = np.hstack((
                train_inp, outp_ps[::4], outp_pe[::4], outp_qrss[::4], outp_qrse[::4], outp_ts[::4], outp_te[::4],
                np.array([outp_vf_ps[-1], outp_vf_pe[-1], outp_vf_qrss[-1], outp_vf_qrse[-1], outp_vf_ts[-1], outp_vf_te[-1]])
            ))

            if len(self.stack_arr) == 0:
                self.stack_arr = np.array(stack_tmp)
            else:
                self.stack_arr = np.vstack((self.stack_arr, stack_tmp))

            '''
            다수결 누적
            '''
            # 순서대로 (심층신경망 판정된 ECG, ps 누적값, ...) 에 대한 array, 각각 4씩 증가함
            major = run_major(
                i,
                (self.raw, outp_ps, outp_pe, outp_qrss, outp_qrse, outp_ts, outp_te),
                (self.raw_arr, self.major_ps, self.major_pe, self.major_qrss, self.major_qrse, self.major_ts, self.major_te),
                (outp_vf_ps, outp_vf_pe, outp_vf_qrss, outp_vf_qrse, outp_vf_ts, outp_vf_te),
                (self.major_vf_ps, self.major_vf_pe, self.major_vf_qrss, self.major_vf_qrse, self.major_vf_ts, self.major_vf_te)
            )

            self.raw_arr = major[0]
            self.major_ps, self.major_pe, self.major_qrss, self.major_qrse, self.major_ts, self.major_te = major[1:7]
            self.major_vf_ps, self.major_vf_pe = major[7:9]
            self.major_vf_qrss, self.major_vf_qrse = major[9:11]
            self.major_vf_ts, self.major_vf_te = major[11::]

            '''
            Major -> Sum, Point
            '''
            # major_xx array와 길이 동일
            point = run_sum(
                self.raw_arr,
                (self.major_ps, self.major_pe, self.major_qrss, self.major_qrse, self.major_ts, self.major_te),
                (self.point_ps, self.point_pe, self.point_qrss, self.point_qrse, self.point_ts, self.point_te),
                (self.major_vf_ps, self.major_vf_pe, self.major_vf_qrss, self.major_vf_qrse, self.major_vf_ts, self.major_vf_te),
                (self.point_vf_ps, self.point_vf_pe, self.point_vf_qrss, self.point_vf_qrse, self.point_vf_ts, self.point_vf_te)
            )

            self.point_ps, self.point_pe, self.point_qrss, self.point_qrse, self.point_ts, self.point_te = point[0:6]
            self.point_vf_ps, self.point_vf_pe = point[6:8]
            self.point_vf_qrss, self.point_vf_qrse = point[8:10]
            self.point_vf_ts, self.point_vf_te = point[10::]

            '''
            RS값 누적
            '''
            # self.recog_score: major_xx array와 길이 동일
            self.recog_score, self.rs_ref = run_rs(
                self.recog_score,
                self.point_qrss,
                (self.major_ps, self.major_pe, self.major_qrss, self.major_qrse, self.major_ts, self.major_te),
                self.rs_ref
            )

            '''
            ecg_parameter 계산
            '''
            # 수정건) if 조건문 i % 10 == 0 보다 연산량을 줄일 효율적인 방법 탐색 필요
            if i % 10 == 0:
                # self.ecg_parameter: 현재 발견된 심박동 수(= self.point_qrss가 1인 수)만큼의 행이 저장됨
                self.ecg_parameter = create_parameter(
                    self.ecg_parameter,
                    self.raw_arr,
                    (self.major_ps, self.major_pe, self.major_qrss, self.major_qrse, self.major_ts, self.major_te),
                    (self.point_ps, self.point_pe, self.point_qrss, self.point_qrse, self.point_ts, self.point_te)
                )
                # self.ecg_parameter_vf: 현재 발견된 VF 수(= self.point_vf_qrss가 1인 수)만큼의 행이 저장됨
                self.ecg_parameter_vf = create_parameter_vf(
                    self.ecg_parameter_vf,
                    self.raw_arr,
                    (self.major_vf_ps, self.major_vf_pe, self.major_vf_qrss, self.major_vf_qrse, self.major_vf_ts, self.major_vf_te),
                    (self.point_vf_ps, self.point_vf_pe, self.point_vf_qrss, self.point_vf_qrse, self.point_vf_ts, self.point_vf_te)
                )

        '''
        저장
        '''
        with open(r'C:\Users\jjm\Desktop\ooo1\0001\240404\inp_outp_1001_p.pickle', 'wb') as f:  # 입,출력 데이터 stack
            pickle.dump(self.stack_arr, f)

        with open(r'C:\Users\jjm\Desktop\ooo1\0001\240404\rs_i_1001_p.pickle', 'wb') as f:  # parameter
            pickle.dump(self.ecg_parameter, f)

        # k = aaa(self.stack_arr, self.ecg_parameter)

        '''
        return 관련
        '''
        save_ref = len(self.raw_arr)

        # 수정건: ai_return 인자별 길이가 안 맞는지 확인 필요
        ai_return = (
            np.round(self.raw_arr[0:save_ref], 2),
            self.major_ps[0:save_ref].astype(float), self.major_pe[0:save_ref].astype(float),
            self.major_qrss[0:save_ref].astype(float), self.major_qrse[0:save_ref].astype(float),
            self.major_ts[0:save_ref].astype(float), self.major_te[0:save_ref].astype(float),
            self.point_ps[0:save_ref].astype(float), self.point_pe[0:save_ref].astype(float),
            self.point_qrss[0:save_ref].astype(float), self.point_qrse[0:save_ref].astype(float),
            self.point_ts[0:save_ref].astype(float), self.point_te[0:save_ref].astype(float),
            self.recog_score[0:save_ref].astype(float),
            self.major_vf_ps[0:save_ref].astype(float), self.major_vf_pe[0:save_ref].astype(float),
            self.major_vf_qrss[0:save_ref].astype(float), self.major_vf_qrse[0:save_ref].astype(float),
            self.major_vf_ts[0:save_ref].astype(float), self.major_vf_te[0:save_ref].astype(float),
            self.point_vf_ps[0:save_ref].astype(float), self.point_vf_pe[0:save_ref].astype(float),
            self.point_vf_qrss[0:save_ref].astype(float), self.point_vf_qrse[0:save_ref].astype(float),
            self.point_vf_ts[0:save_ref].astype(float), self.point_vf_te[0:save_ref].astype(float)
        )

        # time / p_rs, qrs_rs, t_rs / p_duration, p_area / qrs_duration, qrs_area / t_duration, t_area /
        # pq_interval, st_interval, qq_interval / pq_segment, qt_segment, tp_segment /
        # p_amp, qrs_amp, t_amp / p_avg, qrs_avg, t_avg /
        parameter_return = (
            list(self.ecg_parameter[1::, 0]),
            list(self.ecg_parameter[1::, 1]), list(self.ecg_parameter[1::, 2]), list(self.ecg_parameter[1::, 3]),
            list(self.ecg_parameter[1::, 4]), list(self.ecg_parameter[1::, 5]),
            list(self.ecg_parameter[1::, 6]), list(self.ecg_parameter[1::, 7]),
            list(self.ecg_parameter[1::, 8]), list(self.ecg_parameter[1::, 9]),
            list(self.ecg_parameter[1::, 10]), list(self.ecg_parameter[1::, 11]), list(self.ecg_parameter[1::, 12]),
            list(self.ecg_parameter[1::, 13]), list(self.ecg_parameter[1::, 14]), list(self.ecg_parameter[1::, 15]),
            list(self.ecg_parameter[1::, 16]), list(self.ecg_parameter[1::, 17]), list(self.ecg_parameter[1::, 18]),
            list(self.ecg_parameter[1::, 19]), list(self.ecg_parameter[1::, 20]), list(self.ecg_parameter[1::, 21])
        )

        # time / ps_vf, pe_vf, qrss_vf, qrse_vf, ts_vf, te_vf
        parameter_vf_return = (
            list(self.ecg_parameter_vf[1::, 0]),
            list(self.ecg_parameter_vf[1::, 1]), list(self.ecg_parameter_vf[1::, 2]),
            list(self.ecg_parameter_vf[1::, 3]), list(self.ecg_parameter_vf[1::, 4]),
            list(self.ecg_parameter_vf[1::, 5]), list(self.ecg_parameter_vf[1::, 6])
        )

        return ai_return, parameter_return, parameter_vf_return

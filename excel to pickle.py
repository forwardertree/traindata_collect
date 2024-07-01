

import pandas as pd

# 엑셀 파일 경로 (여기서는 예시로 'path_to_file.xlsx'을 사용합니다)
file_path = r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\trainingDB_9_13_240329_1,2_plus.xlsx'

# 엑셀 파일 로드
df = pd.read_excel(file_path, engine='openpyxl')

# 각 컬럼을 딕셔너리로 저장
save_dict = {col: df[col].tolist() for col in df.columns}

# 각 딕셔너리를 사용하여 별도의 데이터프레임 생성
df_inp_ps = pd.DataFrame({'inp_ps': save_dict['inp_ps']})
df_inp_pe = pd.DataFrame({'inp_pe': save_dict['inp_pe']})
df_inp_qrss = pd.DataFrame({'inp_qrss': save_dict['inp_qrss']})
df_inp_qrse = pd.DataFrame({'inp_qrse': save_dict['inp_qrse']})
df_inp_ts = pd.DataFrame({'inp_ts': save_dict['inp_ts']})
df_inp_te = pd.DataFrame({'inp_te': save_dict['inp_te']})
df_ps = pd.DataFrame({'ps': save_dict['ps']})
df_pe = pd.DataFrame({'pe': save_dict['pe']})
df_qrss = pd.DataFrame({'qrss': save_dict['qrss']})
df_qrse = pd.DataFrame({'qrse': save_dict['qrse']})
df_ts = pd.DataFrame({'ts': save_dict['ts']})
df_te = pd.DataFrame({'te': save_dict['te']})


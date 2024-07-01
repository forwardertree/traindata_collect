import pandas as pd
import numpy as np

def modify_array(df):
    arr = df.values
    rows, cols = arr.shape
    modified = False
    new_arr = np.zeros_like(arr)  # 모든 값을 0으로 초기화한 새 배열 생성

    # 배열의 모든 행과 열을 순회하면서 패턴 확인
    for r in range(rows - 1):  # 마지막 행은 아래 행이 없으므로 제외
        for c in range(1, cols):  # 첫번째 열은 왼쪽 열이 없으므로 제외
            if arr[r, c] == 1:
                count = 0
                for k in range(1, min(rows - r, cols - c)):
                    if r + k < rows and c - k >= 0 and arr[r + k, c - k] == 1:
                        count += 1
                    else:
                        break

                if count >= 5:  # 연속된 1이 5개 이상일 경우
                    # 해당 위치의 한칸 오른쪽 열부터 20번째 열까지 반복적으로 검사 및 변경
                    for i in range(min(20, cols - c - 1)):
                        if r - 1 - i >= 0:
                            new_arr[r - 1 - i][c + 1 + i] = 1
                            modified = True
                    # 해당 위치의 한칸 아래와 한 칸 왼쪽으로 시작하여 값을 1로 변경
                    for i in range(min(cols, cols - c)):
                        if r + 1 + i < rows and c - 1 - i >= 0:
                            new_arr[r + 1 + i][c - 1 - i] = 1
                            modified = True

    # 20번째와 21번째 열을 0으로 설정
    if cols >= 20:
        new_arr[:, 20] = 0  # 20번째 열을 0으로 설정
    if cols >= 21:
        new_arr[:, 21] = 0  # 21번째 열을 0으로 설정

    return pd.DataFrame(new_arr, columns=df.columns)

# 엑셀 파일 경로
filename = r'C:\Users\jjm\Desktop\ooo1\0001\240404\trainingDB_9_13_240404_1001_p.xlsx'

# 업데이트를 요구하는 시트와 그렇지 않은 시트 구분
sheets_to_read = ['inp_ps', 'inp_pe', 'inp_qrss', 'inp_qrse', 'inp_ts', 'inp_te', 'ps', 'pe', 'qrss', 'qrse', 'ts', 'te']
update_sheets = ['ps', 'pe', 'qrss', 'qrse', 'ts', 'te']  # 이 시트들만 데이터를 업데이트
df_sheets = pd.read_excel(filename, sheet_name=sheets_to_read, engine='openpyxl')

# 결과 딕셔너리 초기화
updated_sheets_dict = {}

# 수정해야 할 시트들만 수정
for sheet_name, df_sheet in df_sheets.items():
    if sheet_name in update_sheets:
        updated_sheets_dict[sheet_name] = modify_array(df_sheet)  # modify_array 함수는 수정된 데이터를 반환
    else:
        updated_sheets_dict[sheet_name] = df_sheet  # 수정하지 않고 원본 데이터 그대로 저장

# 결과 파일 경로 수정
output_filename = filename.replace(r'C:\Users\jjm\Desktop\ooo1\0001\240404\trainingDB_9_13_240404_1001_p.xlsx',
                                   r'C:\Users\jjm\Desktop\ooo1\0001\240404\updated_trainingDB_9_13_240404_1001_p.xlsx')

# 결과 딕셔너리를 DataFrame으로 다시 변환하여 Excel 파일로 저장
with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
    for sheet_name, df in updated_sheets_dict.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)

# # 파일과 시트 처리
# filename = r'C:\Users\jjm\Desktop\ooo1\0001\240404\trainingDB_9_13_240404_1001_p.xlsx'
# sheets_to_read = ['inp_ps', 'inp_pe', 'inp_qrss', 'inp_qrse', 'inp_ts', 'inp_te', 'ps', 'pe', 'qrss', 'qrse', 'ts', 'te']
# df_sheets = pd.read_excel(filename, sheet_name=sheets_to_read, engine='openpyxl')
#
# # 각 시트를 함수로 업데이트하고 결과를 딕셔너리로 저장
# updated_sheets_dict = {sheet_name: modify_array(df_sheet) for sheet_name, df_sheet in df_sheets.items()}
#
# # 결과 저장
# output_filename = filename.replace(r'C:\Users\jjm\Desktop\ooo1\0001\240404\trainingDB_9_13_240404_1001_p.xlsx', r'C:\Users\jjm\Desktop\ooo1\0001\240404\updated_trainingDB_9_13_240404_1001_p.xlsx')
# with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
#     for sheet_name, df in updated_sheets_dict.items():
#         df.to_excel(writer, sheet_name=sheet_name, index=False)

# import pandas as pd
# import numpy as np
#
# # 읽어올 엑셀 파일 지정
# filename = r'C:\Users\jjm\Desktop\ooo1\0001\240404\trainingDB_9_13_240404_1001_p.xlsx'
#
# # 인덱스를 지정해 시트 설정
# sheets_to_read = ['ps', 'pe', 'qrss', 'qrse', 'ts', 'te']
#
# # 엑셀 파일 읽어 오기
# df_sheets = pd.read_excel(filename, sheet_name=sheets_to_read, engine='openpyxl')
#
# def modify_array(df):
#     arr = df.values
#     rows, cols = arr.shape
#     modified = False
#
#     # 배열의 모든 행과 열을 순회하면서 패턴 확인
#     for r in range(rows - 1):  # 마지막 행은 아래 행이 없으므로 제외
#         for c in range(1, cols):  # 첫번째 열은 왼쪽 열이 없으므로 제외
#             if arr[r, c] == 1:
#                 # 대각선 아래 방향으로 5번 이상 연속된 '1' 찾기
#                 count = 0
#                 for k in range(1, min(rows - r, cols - c)):
#                     if r + k < rows and c - k >= 0 and arr[r + k, c - k] == 1:
#                         count += 1
#                     else:
#                         break
#                 # if count >= 5:
#                 #     # 조건을 만족하고, 한 칸 오른쪽 열이 20번째 열 이내일 때
#                 #     if c + 1 < cols and c + 1 < 20 and r > 0 and arr[r - 1, c + 1] == 0:
#                 #         arr[r - 1, c + 1] = 1
#
#                 # 연속된 1이 5개 이상일 경우, 한 칸 위로 오른쪽으로 1을 채우는 조건 처리
#                 if count >= 5:
#                     for i in range(min(20, cols - c - 1)):  # 최대 20칸, 열의 끝까지만 처리
#                         if r - 1 - i >= 0:
#                             arr[r - 1 - i][c + 1 + i] = 1
#                             modified = True
#
#                 # 연속된 1이 5개 이상일 경우, 한 칸 아래와 한 칸 왼쪽으로 시작하여 값을 1로 변경
#                 if count >= 5:
#                     for i in range(min(cols, cols - c)):  # 최대 열의 끝까지만 처리
#                         if r + 1 + i < rows and c - 1 - i >= 0:
#                             arr[r + 1 + i][c - 1 - i] = 1
#                             modified = True
#
#
#     return pd.DataFrame(arr, columns=df.columns)
#
# # 각 시트를 함수로 업데이트하고 결과를 딕셔너리로 저장
# updated_sheets_dict = {sheet_name: modify_array(df_sheet) for sheet_name, df_sheet in df_sheets.items()}
#
# # 엑셀 파일 경로 수정
# output_filename = filename.replace("240404\\trainingDB_9_13_240404_1001_p.xlsx", "240404\\updated_trainingDB_9_13_240404_1001_p.xlsx")
#
# # 결과 딕셔너리를 DataFrame으로 다시 변환하여 Excel 파일로 저장하기
# with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
#     for sheet_name, df in updated_sheets_dict.items():
#         df.to_excel(writer, sheet_name=sheet_name, index=False)

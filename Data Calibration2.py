import pandas as pd

# 파일 경로 정의
file_path = r'C:\Users\jjm\Desktop\ooo1\0001\240404\updated_trainingDB_9_13_240404_1001_p.xlsx'

try:
    # Excel 파일 불러오기
    xl = pd.ExcelFile(file_path)

    # 지정된 시트를 데이터프레임으로 로드
    df1 = xl.parse('inp_ps')  # 시트 이름이 파일과 정확히 일치해야 함
    df2 = xl.parse('ps')

    # 첫 20개 열이 모두 0인 행 찾기
    rows_to_remove = df2.iloc[:, 0:20].eq(0).all(axis=1)

    # 이 인덱스를 사용하여 두 데이터프레임에서 행 제거
    df2_cleaned = df2.loc[~rows_to_remove]
    df1_cleaned = df1.loc[~rows_to_remove]


    # 결과 저장할 파일 경로 정의
    output_path = r'C:\Users\jjm\Desktop\ooo1\0001\240404\updated.xlsx'

    # 정리된 데이터프레임을 새 Excel 파일로 저장
    with pd.ExcelWriter(output_path) as writer:
        df1_cleaned.to_excel(writer, sheet_name='inp_ps', index=False)
        df2_cleaned.to_excel(writer, sheet_name='ps', index=False)


except Exception as e:
    print("Excel 파일 처리 중 오류 발생:", e)

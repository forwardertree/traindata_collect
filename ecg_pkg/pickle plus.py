import pickle

# 첫 번째 pickle 파일 로드
with open(r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\trainingDB_9_13_240328.pickle', 'rb') as f:
    data1 = pickle.load(f)

# 두 번째 pickle 파일 로드
with open(r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\trainingDB_9_13_240326_6.pickle', 'rb') as f:
    data2 = pickle.load(f)

# 두 데이터 합치기
combined_data = data1 + data2

# 합친 데이터를 새로운 pickle 파일로 저장
with open(r'C:\Users\jjm\Desktop\ecg_ai\trainingDB\ecgdvq_62.5\trainingDB_9_13_240328plus1001.pickle', 'wb') as f:
    pickle.dump(combined_data, f)

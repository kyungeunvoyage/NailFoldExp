import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("start")
# 그래프 스타일 설정
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)

# 파일 리스트 (본인의 파일명에 맞게 수정)
file_names = ['P3_Exp1_AbsoluteThresholdDetection.csv','P4_Exp1_AbsoluteThresholdDetection.csv'
              ,'P5_Exp1_AbsoluteThresholdDetection.csv', 'P6_Exp1_AbsoluteThresholdDetection.csv']
data_list = []

for file in file_names:
    try:
        temp_df = pd.read_csv(file)
        data_list.append(temp_df)
    except FileNotFoundError:
        print(f"no: {file}")

# 데이터 통합
if data_list:
    df = pd.concat(data_list, ignore_index=True)

    # 1. Condition 명칭 세분화 (Active -> Soft, Hard 구분)
    condition_map = {
        'Active': 'On-touch (Soft)',
        'On-touch (Hard)': 'On-touch (Hard)',
        'In-air': 'In-air'
    }
    # .map()을 쓰거나 .replace()를 사용하여 변경
    df['Condition'] = df['Condition'].replace(condition_map)

    # 2. 분석 대상 구역 필터링 (Area A, B, C)
    df = df[df['Area'].isin(['A', 'B', 'C'])]

    # 3. Force 문자열에서 숫자만 추출 (예: '0.07g' -> 0.07)
    df['Force_Val'] = df['Force'].str.extract('(\d+\.?\d*)').astype(float)

    print("완")
else:
    print("없")
    

# 1. 전반적인 강도별 정답률 계산 (Psychometric Curve용)
overall_acc = df.groupby(['Condition', 'Force_Val'])['IsCorrect'].mean().reset_index()

# 2. 구역별(Area) 평균 정답률 계산 (Bar Chart용)
area_acc = df.groupby(['Condition', 'Area'])['IsCorrect'].mean().reset_index()

# 그래프 그리기
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# (Left) Psychometric Curve
sns.lineplot(ax=axes[0], data=overall_acc, x='Force_Val', y='IsCorrect',
             hue='Condition', marker='o', linewidth=3, markersize=8)
axes[0].axhline(0.8, color='red', linestyle='--', label='80% Threshold')
axes[0].set_title('Psychometric Curve: Detection Accuracy vs Force', fontsize=15)
axes[0].set_xlabel('Stimulus Force (g)', fontsize=12)
axes[0].set_ylabel('Accuracy (Detection Probability)', fontsize=12)
axes[0].set_ylim(-0.05, 1.05)
axes[0].legend()

# (Right) Regional Accuracy
sns.barplot(ax=axes[1], data=area_acc, x='Area', y='IsCorrect',
            hue='Condition', order=['A', 'B', 'C'], palette='coolwarm')
axes[1].set_title('Detection Accuracy by Finger Region (A, B, C)', fontsize=15)
axes[1].set_xlabel('Finger Region', fontsize=12)
axes[1].set_ylabel('Mean Accuracy', fontsize=12)
axes[1].set_ylim(0, 1.1)

plt.tight_layout()
plt.show()

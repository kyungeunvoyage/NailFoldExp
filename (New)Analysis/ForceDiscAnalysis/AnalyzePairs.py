import pandas as pd

df = pd.read_csv('/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/allinone.csv')

# Correct: chose the larger stimulus
df['Correct'] = (
    ((df['Comparison'] > df['Reference']) & (df['ChoseComparison'] == 1)) |
    ((df['Comparison'] < df['Reference']) & (df['ChoseComparison'] == 0))
)

results = []

for subject, group in df.groupby('Subject'):
    incorrect = group[~group['Correct']]
    
    results.append({'Subject': subject, 'Metric': 'Correct Pairs',        'Value': group['Correct'].sum()})
    results.append({'Subject': subject, 'Metric': 'Incorrect Pairs',      'Value': (~group['Correct']).sum()})
    results.append({'Subject': subject, 'Metric': '1 in all pairs',       'Value': (group['UserChoice'] == 1).sum()})
    results.append({'Subject': subject, 'Metric': '2 in all pairs',       'Value': (group['UserChoice'] == 2).sum()})
    results.append({'Subject': subject, 'Metric': '1 in incorrect pairs', 'Value': (incorrect['UserChoice'] == 1).sum()})
    results.append({'Subject': subject, 'Metric': '2 in incorrect pairs', 'Value': (incorrect['UserChoice'] == 2).sum()})

result_df = pd.DataFrame(results)

# 스크린샷처럼 Subject가 첫 행에만 나오도록 출력
result_df.to_csv('/Users/kyungeunjung/NailFoldExp/Data/(FD)CurData/pair_analysis.csv', index=False)
print(result_df.to_string(index=False))
print("\n저장 완료: pair_analysis.csv")
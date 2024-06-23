import pandas as pd

# 讀取Excel文件中的所有工作簿
all_sheets_data = pd.read_excel('ragtest_systex_0811.xlsx', sheet_name=None)

# 初始化一個空的DataFrame用於合併數據
combined_data = pd.DataFrame()

# 遍歷每個工作簿，選擇需要的列，並合併到combined_data中
for sheet_data in all_sheets_data.values():
    # 檢查並選擇需要的列
    if 'Q' in sheet_data and 'A' in sheet_data:
        selected_data = sheet_data[['Q', 'A']]
    else:
        selected_data = pd.DataFrame(columns=['Q', 'A'])
    combined_data = pd.concat([combined_data, selected_data], ignore_index=True)

# ID
combined_data['id'] = range(1, len(combined_data) + 1)

# 順序
combined_data = combined_data[['id', 'Q', 'A']]

# 保存處理後的數據到新的Excel文件
combined_data.to_excel('train.xlsx', index=False)

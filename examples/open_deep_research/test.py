import ast
from utils.common_utils import safe_json_parse

str1 = "214[{'key1': 'value1'},{ 'key2': 'value2' }]fijoga "  # ← 关键：先初始化为空字符串

# with open("test_json.txt", "r", encoding="utf-8") as f:
#     for line in f:
#         str1 += line.strip()

# print(str1)
print("=" * 60)
print(safe_json_parse(str1))
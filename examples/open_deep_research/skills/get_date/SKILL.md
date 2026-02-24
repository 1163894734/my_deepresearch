---
name: get_date
description: 获取当前系统日期和时间
---
# Get Date Skill
## 功能说明
这是一个用来获取当前系统时间的工具，支持多种时间格式输出，包括默认格式、仅日期、仅时间、ISO 8601 格式和 Unix 时间戳。

## 使用示例

### 基础用法
```bash
# 默认格式输出 (YYYY-MM-DD HH:MM:SS)
python scripts/get_date.py

# 仅显示日期
python scripts/get_date.py --format date

# 仅显示时间
python scripts/get_date.py --format time

# ISO 8601 格式
python scripts/get_date.py --format iso

# Unix 时间戳
python scripts/get_date.py --format timestamp
```

### 高级用法
```bash
# JSON 格式输出（适合程序调用）
python scripts/get_date.py --json

# 结合其他参数
python scripts/get_date.py --format date --json
```

## 输出示例

**默认输出:**
```
2026-02-22 14:30:45
```

**JSON 输出:**
```json
{
  "status": "success",
  "data": {
    "date_time": "2026-02-22 14:30:45",
    "format": "default"
  }
}
```

## 支持的格式
- `default`: 完整日期和时间 (YYYY-MM-DD HH:MM:SS)
- `date`: 仅日期 (YYYY-MM-DD)
- `time`: 仅时间 (HH:MM:SS)
- `iso`: ISO 8601 格式
- `timestamp`: Unix 时间戳
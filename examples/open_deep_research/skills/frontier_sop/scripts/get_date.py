#!/usr/bin/env python3
"""
获取当前系统时间的脚本
"""

from datetime import datetime
from smolagents import Tool


class GetDateTool(Tool):
    """获取当前系统日期和时间的工具"""
    
    name = "get_date"
    description = "获取当前系统日期和时间，支持多种格式输出"
    output_type = "string"
    
    inputs = {
        "format": {
            "type": "string",
            "description": "时间格式类型: default(完整日期时间), date(仅日期), time(仅时间), iso(ISO 8601), timestamp(Unix时间戳)",
            "enum": ["default", "date", "time", "iso", "timestamp"],
            "nullable": True,
        }
    }
    
    def forward(self, format: str = "default") -> str:
        """
        获取当前系统时间
        
        Args:
            format: 时间格式类型
                - 'default': 返回完整日期和时间 (YYYY-MM-DD HH:MM:SS)
                - 'date': 仅返回日期 (YYYY-MM-DD)
                - 'time': 仅返回时间 (HH:MM:SS)
                - 'iso': ISO 8601 格式
                - 'timestamp': Unix 时间戳
        
        Returns:
            格式化的日期/时间字符串
        """
        now = datetime.now()
        
        formats = {
            'default': now.strftime('%Y-%m-%d %H:%M:%S'),
            'date': now.strftime('%Y-%m-%d'),
            'time': now.strftime('%H:%M:%S'),
            'iso': now.isoformat(),
            'timestamp': str(int(now.timestamp()))
        }
        
        result = formats.get(format, formats['default'])
        return result


if __name__ == '__main__':
    # 用于直接运行测试
    tool = GetDateTool()
    print("默认格式:", tool.forward("default"))
    print("仅日期:", tool.forward("date"))
    print("仅时间:", tool.forward("time"))
    print("ISO格式:", tool.forward("iso"))
    print("时间戳:", tool.forward("timestamp"))

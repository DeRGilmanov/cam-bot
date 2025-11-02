# utils/__init__.py
import re
from typing import Optional

def validate_time_format(time_str: str) -> bool:
    """Проверка формата времени HH:MM"""
    pattern = r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$'
    return bool(re.match(pattern, time_str))

def split_long_message(text: str, max_length: int = 4096) -> list:
    """Разделение длинного сообщения на части"""
    if len(text) <= max_length:
        return [text]
    
    parts = []
    while text:
        if len(text) <= max_length:
            parts.append(text)
            break
        else:
            # Ищем место для разрыва (последний перенос строки или пробел)
            split_pos = text.rfind('\n', 0, max_length)
            if split_pos == -1:
                split_pos = text.rfind(' ', 0, max_length)
            if split_pos == -1:
                split_pos = max_length
            
            parts.append(text[:split_pos])
            text = text[split_pos:].lstrip()
    
    return parts
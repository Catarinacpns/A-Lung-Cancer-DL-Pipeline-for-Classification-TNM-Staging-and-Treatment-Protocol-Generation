import re

def extract_number(s):
    # Extracts numbers from the string, ignoring leading characters
    numbers = re.findall(r'\d+', s)
    return int(numbers[0]) if numbers else 0
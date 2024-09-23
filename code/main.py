"""
더미데이터를 생성 및 암호화를 실시하여
분석된 데이터 결과 값을 csv로 저장합니다.
"""

import os
import time
import pandas as pd
import numpy as np
import psutil
from memory_profiler import memory_usage
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad, unpad
from Crypto.Random import get_random_bytes
import matplotlib.font_manager as fm

# 더미 데이터 생성 클래스
class DummyDataGenerator:
    
    def __init__(self, set_mb: int):
        self.set_mb = set_mb

    def generate_text_file(self, file_name):
        """지정한 크기의 텍스트 파일 생성"""
        size_in_bytes = self.set_mb * 1024 * 1024
        with open(file_name, 'w') as f:
            f.write('0' * size_in_bytes)
        print(f"텍스트 파일 {file_name} ({self.set_mb}MB) 생성 완료")

    def generate_image_file(self, file_name, width, height):
        """지정한 크기의 이미지 파일 생성"""
        target_size_in_bytes = self.set_mb * 1024 * 1024
        img_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        img = Image.fromarray(img_data, 'RGB')
        img.save(file_name, quality=100)
        current_size = os.path.getsize(file_name)
        if current_size < target_size_in_bytes:
            with open(file_name, 'ab') as f:
                f.write(b'\0' * (target_size_in_bytes - current_size))
        print(f"이미지 파일 {file_name} ({self.set_mb}MB) 생성 완료 (실제 크기: {os.path.getsize(file_name) / (1024 * 1024):.2f}MB)")

    def generate_video_file(self, file_name, width, height, duration_in_seconds):
        """지정한 크기의 동영상 파일 생성"""
        target_size_in_bytes = self.set_mb * 1024 * 1024
        frame_count = int(duration_in_seconds * 30)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        video = cv2.VideoWriter(file_name, fourcc, 30, (width, height))
        for _ in range(frame_count):
            img_data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            video.write(img_data)
        video.release()
        current_size = os.path.getsize(file_name)
        if current_size < target_size_in_bytes:
            with open(file_name, 'ab') as f:
                f.write(b'\0' * (target_size_in_bytes - current_size))
        print(f"동영상 파일 {file_name} ({self.set_mb}MB) 생성 완료 (실제 크기: {os.path.getsize(file_name) / (1024 * 1024):.2f}MB)")

# XOR 암호화 클래스
class XOR:
    
    def __init__(self, key):
        self.key = key
    
    def encrypt_decrypt(self, data):
        """텍스트 데이터를 XOR 연산으로 암호화/복호화"""
        return ''.join(chr(ord(char) ^ self.key) for char in data)
    
    def encrypt_decrypt_file(self, input_path, output_path):
        """파일 데이터를 XOR 연산으로 암호화/복호화"""
        with open(input_path, 'rb') as f:
            data = bytearray(f.read())
        
        for i in range(len(data)):
            data[i] ^= self.key
        
        with open(output_path, 'wb') as f:
            f.write(data)

# AES 암호화 클래스
class AES_Encryption:

    def __init__(self, key=None):
        self.key = key if key else get_random_bytes(16)  # 16바이트(128비트) 기본 키 설정
    
    def encrypt(self, data):
        """텍스트 데이터를 AES로 암호화"""
        cipher = AES.new(self.key, AES.MODE_CBC)
        return cipher.iv + cipher.encrypt(pad(data.encode(), AES.block_size))
    
    def decrypt(self, ciphertext):
        """AES 암호문을 복호화"""
        iv = ciphertext[:16]
        ct = ciphertext[16:]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return unpad(cipher.decrypt(ct), AES.block_size).decode()
    
    def encrypt_file(self, input_path, output_path):
        """파일 데이터를 AES로 암호화"""
        with open(input_path, 'rb') as f:
            data = f.read()
        
        cipher = AES.new(self.key, AES.MODE_CBC)
        ct_bytes = cipher.encrypt(pad(data, AES.block_size))
        
        with open(output_path, 'wb') as f:
            f.write(cipher.iv + ct_bytes)
    
    def decrypt_file(self, input_path, output_path):
        """AES 암호화된 파일을 복호화"""
        with open(input_path, 'rb') as f:
            iv = f.read(16)
            ct = f.read()
        
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        pt = unpad(cipher.decrypt(ct), AES.block_size)
        
        with open(output_path, 'wb') as f:
            f.write(pt)

# 성능 데이터 저장용 리스트
performance_data = []

# XOR 및 AES 클래스 초기화
xor = XOR(key=123)
aes = AES_Encryption()

# 성능 측정 함수
def measure_performance(algorithm, data_type, file_name):
    # XOR 또는 AES 선택
    if algorithm == "XOR":
        encrypt_function = xor.encrypt_decrypt_file
    elif algorithm == "AES":
        encrypt_function = aes.encrypt_file

    # 프로세스 정보 가져오기
    process = psutil.Process()
    initial_cpu = process.cpu_percent(interval=None)
    initial_memory = process.memory_info().rss

    # 메모리 사용량 측정 시작
    mem_usage = memory_usage(-1, interval=0.1, timeout=1)
    
    # 암호화 시간 측정 시작
    start_time = time.time()
    encrypt_function(file_name, f'encrypted_{algorithm.lower()}_{file_name}')
    end_time = time.time()
    encryption_time = end_time - start_time
    
    # 암호화 후 CPU 및 메모리 사용량 측정
    cpu_usage = process.cpu_percent(interval=None) - initial_cpu
    memory_usage_change = process.memory_info().rss - initial_memory
    
    # 메모리 사용량 측정 종료
    peak_memory_usage = max(mem_usage)
    
    # 파일 크기
    file_size = os.path.getsize(file_name) / (1024 * 1024)  # MB 단위

    # 성능 데이터 저장
    performance_data.append({
        'Algorithm': algorithm,
        'DataType': data_type,
        'FileName': file_name,
        'FileSize_MB': file_size,
        'EncryptionTime_s': encryption_time,
        'CPU_Usage_Change_%': cpu_usage,
        'Memory_Usage_Change_MB': memory_usage_change / (1024 ** 2),
        'Peak_Memory_Usage_MB': peak_memory_usage
    })

# 데이터 생성 및 암호화 성능 측정
sizes = [200, 512, 1024, 5120, 10240]  # MB 단위 크기 설정
generator = DummyDataGenerator(set_mb=0)

# 더미 데이터 생성 및 성능 측정
for size in sizes:
    generator.set_mb = size
    text_file = f'dummy_text_{size}mb.txt'
    image_file = f'dummy_image_{size}mb.jpg'
    
    # 더미 데이터 생성
    generator.generate_text_file(text_file)
    generator.generate_image_file(image_file, 1920, 1080)
    
    # 성능 측정
    measure_performance("XOR", "Text", text_file)
    measure_performance("XOR", "Image", image_file)
    
    measure_performance("AES", "Text", text_file)
    measure_performance("AES", "Image", image_file)

# 결과를 데이터프레임으로 변환
df = pd.DataFrame(performance_data)

# 결과 데이터 저장
df.to_csv("encryption_performance_results.csv", index=False)

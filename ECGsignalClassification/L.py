import wfdb
import pywt
import numpy as np
import matplotlib.pyplot as plt


# 选择记录名称
record_name = 'mit-bih-ecg-data/102'  # 更改为您要查看的记录名称

# 读取记录信息
record = wfdb.rdrecord(record_name)
annotation = wfdb.rdann(record_name, 'atr')

# 获取所有的心电类型
ecg_types = list(set(annotation.symbol))

# 进行小波变换降噪
def wavelet_denoise(signal):
    # 使用小波变换（db4小波）进行降噪
    coeffs = pywt.wavedec(signal, 'db4', level=9)

    # 通过设置阈值进行去噪
    threshold = np.std(coeffs[-1]) * 3  # 设置一个适当的阈值
    coeffs[1:] = (pywt.threshold(detail, threshold, mode='soft') for detail in coeffs[1:])

    # 重构信号
    denoised_signal = pywt.waverec(coeffs, 'db4')

    return denoised_signal


# 绘制每种类型的心电信号
plt.figure(figsize=(12, 6))

# N in 100; A in ; V in ; L in 111; R in 118
for ecg_type in ecg_types:
    if ecg_type == 'V':
        # 找到第一个指定类型的样本
        sample_index = annotation.symbol.index(ecg_type)
        sample_sample = annotation.sample[sample_index]

    # 选择信号通道（通常为0或1）
        signal_channel = 0

    # 读取信号数据
        signal = record.p_signal[:1000, signal_channel]
        # denoised_ecg_signal = wavelet_denoise(signal)
    # 绘制ECG信号图像
        # plt.subplot(len(ecg_types), 1, ecg_types.index(ecg_type) + 1)
        # plt.plot(denoised_ecg_signal)
        plt.plot(signal)
        plt.title('MIT-BIH Database - {} Beat ECG Signal (Record {})'.format(ecg_type, record_name))
        plt.xlabel('Sample')
        plt.ylabel('Amplitude')
        # plt.grid(True)
    else :
        continue

plt.tight_layout()
plt.show()

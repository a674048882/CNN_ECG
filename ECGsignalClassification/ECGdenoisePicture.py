import wfdb
import numpy as np
import pywt
import matplotlib.pyplot as plt

# 数据库路径
database_dir = '/path/to/mitdb'  # 将路径替换为MIT-BIH数据库的实际路径

# 选择记录文件和信号类型（MLII导联）
record_name = 'mit-bih-ecg-data/100'  # 这里使用一个示例记录，可以更改为其他记录
signal_name = 'MLII'

# 从MIT-BIH数据库中读取信号
record = wfdb.rdrecord(record_name)

# 获取MLII导联的信号数据
ecg_signal = record.p_signal[:800, record.sig_name.index(signal_name)]


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


plt.rc('font', family='Times New Roman')

# 对ECG信号进行小波降噪
denoised_ecg_signal = wavelet_denoise(ecg_signal)

# 绘制原始信号和降噪后的信号对比图
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(ecg_signal)
plt.title('Original ECG Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude (mV)')

plt.subplot(2, 1, 2)
plt.plot(denoised_ecg_signal)
plt.title('Denoised ECG Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude (mV)')

plt.tight_layout()
plt.show()



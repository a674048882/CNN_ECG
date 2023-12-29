import wfdb
import matplotlib.pyplot as plt

# 选择记录文件和信号类型（MLII导联）
record_name = 'mit-bih-ecg-data/114'  # 这里使用一个示例记录，可以更改为其他记录
signal_name = 'MLII'

# 从MIT-BIH数据库中读取信号
record = wfdb.rdrecord(record_name)
annotation = wfdb.rdann(record_name, 'atr')

# 初始化一个字典来存储每种心电信号的QRS波形
qrs_waveforms = {'N': None, 'A': None, 'V': None, 'L': None, 'R': None}

# 提取QRS波的样本标记
qrs_samples = annotation.sample
qrs_symbols = annotation.symbol

# 找到N、A、V、L、R五种心电信号的QRS波形并存储
for i, symbol in enumerate(qrs_symbols):
    if symbol in ['N', 'A', 'V', 'L', 'R'] and qrs_waveforms[symbol] is None:
        start = qrs_samples[i] - record.fs  # 从QRS波前一个采样点开始
        end = qrs_samples[i] + record.fs  # 到QRS波后一个采样点结束
        qrs_waveforms[symbol] = record.p_signal[start:end, record.sig_name.index(signal_name)]

plt.rc('font', family='Times New Roman')

# 绘制每种心电信号的QRS波形
for symbol, waveform in qrs_waveforms.items():
    if waveform is not None:
        plt.figure()
        plt.plot(waveform)
        plt.title(f'QRS Waveform ({symbol})')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude (mV)')
        plt.show()

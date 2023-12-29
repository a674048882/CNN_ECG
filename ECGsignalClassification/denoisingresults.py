import wfdb
import matplotlib.pyplot as plt

# 指定MIT-BIH数据库中的记录名，例如：100，101，102等
record_name = 'mit-bih-ecg-data/100'  # 你可以根据需要更改记录名

# 从MIT-BIH数据库中读取ECG信号数据
record = wfdb.rdrecord(record_name)

# 确定MLII导联的索引（通常是第0个导联）
ml_ii_index = record.sig_name.index('MLII')

# 提取MLII导联的ECG信号数据
ml_ii_signal = record.p_signal[0:1000, ml_ii_index]

# 获取信号采样频率
sampling_frequency = record.fs

# 创建时间序列
time = [i / sampling_frequency for i in range(len(ml_ii_signal))]

plt.rc('font', family='Times New Roman')

# 绘制MLII导联的ECG信号
plt.figure(figsize=(12, 4))
plt.plot(time, ml_ii_signal, lw=0.5)
plt.title('MIT-BIH ECG Signal (MLII) - Record ' + record_name)
plt.xlabel('Time (s)', font={'family': 'Times New Roman', 'size': 14})
plt.ylabel('Amplitude', font={'family': 'Times New Roman', 'size': 14})
plt.grid(True)
plt.show()

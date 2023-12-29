import wfdb
import matplotlib.pyplot as plt

# 设置数据库路径
db_dir = 'D:\Study\0708-PASEPM-401003\ECGsignalClassification\mit-bih-ecg-data'  # 替换为您的数据库路径

# 设置要查看的记录名称
record_name = '100'  # 替换为您想要查看的记录名称

# 加载心电图记录
record = wfdb.rdrecord(record_name)

# 绘制心电图波形
plt.figure(figsize=(10, 6))
for signal in record.p_signal.T:
    plt.plot(signal)

plt.title(f"ECG Signals for Record {record_name}")
plt.xlabel("Sample Number")
plt.ylabel("Amplitude")
plt.legend([f"Lead {i+1}" for i in range(record.n_sig)])
plt.show()

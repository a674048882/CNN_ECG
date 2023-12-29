import wfdb
import matplotlib.pyplot as plt


# 选择记录名称
record_name = 'mit-bih-ecg-data/111'  # 更改为您要查看的记录名称

# 读取记录信息
record = wfdb.rdrecord(record_name)
annotation = wfdb.rdann(record_name, 'atr')

# 获取所有的心电类型
ecg_types = list(set(annotation.symbol))

# 用户选择要查看的心电类型
print("Available ECG Types:")
for i, ecg_type in enumerate(ecg_types):
    print(f"{i + 1}: {ecg_type}")

user_choice = int(input("Enter the number of the ECG type you want to view: ")) - 1

if user_choice < 0 or user_choice >= len(ecg_types):
    print("Invalid choice. Exiting.")
else:
    selected_ecg_type = ecg_types[user_choice]

    # 找到指定类型的样本
    sample_indices = [i for i, symbol in enumerate(annotation.symbol) if symbol == selected_ecg_type]

    if not sample_indices:
        print(f"No samples found for {selected_ecg_type} ECG type in record {record_name}.")
    else:
        plt.figure(figsize=(12, 8))

        for i, sample_index in enumerate(sample_indices):
            # 选择信号通道（通常为0或1）
            signal_channel = 0

            # 读取信号数据
            signal = record.p_signal[:, signal_channel]

            # 绘制ECG信号图像
            plt.subplot(len(sample_indices), 1, i + 1)
            plt.plot(signal)
            plt.title(f'MIT-BIH Database - {selected_ecg_type} Beat ECG Signal (Record {record_name})')
            plt.xlabel('Sample')
            plt.ylabel('Amplitude')
            plt.grid(True)

        plt.tight_layout()
        plt.show()


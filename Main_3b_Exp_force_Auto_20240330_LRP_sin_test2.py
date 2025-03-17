import multiprocessing
import threading
import time
import numpy as np
import datetime
from NewMotor.mf5015 import CANdev
import os
import time
from sample.example_DAQUSB401x import *
from force_decouple_python.tf_downstream_txt_20240330 import *
Flaps=20
controlFre = 3000

#徐文华:
#平动传感器 200mm=2.5v
#转动传感器，360度=5v



# Base directory where all the operations will be conducted
base_dir = r''

# Folder names for each experimental case
folders = [f"MMGP_OL{i}" for i in range(8)]


def check_flag(dir_path):
    flag_path = os.path.join(dir_path, 'flag.txt')
    try:
        with open(flag_path, 'r') as file:
            flag = int(file.read().strip())
            return flag == 0
    except FileNotFoundError:
        return False


def reset_flag(dir_path):
    flag_path = os.path.join(dir_path, 'flag.txt')
    with open(flag_path, 'w') as file:
        file.write('1')


def improved_sigmoid_alpha(i):
    N = Flaps / 2
    if 0 <= i <= N:
        return 0.5 * (1 + np.tanh((4 * (i-1)) / 1))
    elif N < i <= 2 * N:
        return 0.5 * (1 + np.tanh((4 * (2 * N - (i+1))) / 1))
    else:
        return 0


# 全局变量，用于传递结果
global global_results

def motor_control(can_dict, T, datap, datah):
    i = 0
    Start = time.time()
    periodtime = 0
    while periodtime <= Flaps:
        Now = time.time()
        periodtime = (Now - Start) / T
        periodtick = periodtime % 1
        can_dict['pitch_1'] = datap[int(periodtick * len(datap))] * improved_sigmoid_alpha(periodtime)*-1
        can_dict['heave_1'] = datah[int(periodtick * len(datap))] * improved_sigmoid_alpha(periodtime)
        i += 1
    print("Motor control finished")

def daq_collection(T):
    global global_results
    time.sleep(2*T)  # 等待电机稳定
    print(' ---DAQ software started--- ')
    result = main_sample(T * (Flaps - 4),T)



    print(' ---DAQ software finished--- ')
    time.sleep(3*T)
    CT, ETA = process_data_and_calculate_metrics(result,T)
    global_results=[CT,  ETA]  # 将结果放入全局变量
    print("Results put into global_results")

def execute_experiment(folder,can_dict):
    base_dir = os.getcwd()  # 假设 base_dir 是当前工作目录
    dir_path = os.path.join(base_dir, folder)
    control_file = os.path.join(dir_path, 'control.txt')
    control2_file = os.path.join(dir_path, 'control2.txt')
    datap = np.loadtxt(control_file)
    datah = np.loadtxt(control2_file)
    T = len(datap) / controlFre



    # 创建线程
    motor_thread = threading.Thread(target=motor_control, args=(can_dict, T, datap, datah))
    daq_thread = threading.Thread(target=daq_collection, args=(T,))

    # 启动线程
    motor_thread.start()
    daq_thread.start()

    # 等待线程完成
    print("wait for MOTOR")
    motor_thread.join()
    #can.close()
    print("wait for daq")
    daq_thread.join()
    print("wait for CTETA")





    global global_results
    print(f"Got results from global_results: CT={global_results[0]}, ETA={global_results[1]}")
    data_y_path = os.path.join(dir_path, 'dataY.txt')
    np.savetxt(data_y_path, [global_results], delimiter=',', fmt='%0.4f')
    reset_flag(dir_path)  # 实验完成后重置标志


if __name__ == "__main__":
    # Maximum number of experiments to check
    max_experiments = 200

    # Main loop to check each directory for flags and run experiments
    experiment_count = 0
    can_dict = multiprocessing.Manager().dict()
    can_dict['heave_1'] = 0.0
    can_dict['heave_2'] = 0.0
    can_dict['pitch_1'] = 0.0
    can_dict['pitch_2'] = 0.0
    can = CANdev(can_dict)
    can.start()

    while experiment_count < max_experiments:
        for folder in folders:
            print("read"+folder)
            if check_flag(os.path.join(base_dir, folder)):
                execute_experiment(folder,can_dict)
                experiment_count += 1
                print("num_exp",experiment_count)
                if experiment_count >= max_experiments:
                    break
            time.sleep(2)  # Wait for 10 seconds before checking again to prevent high CPU usage

    print("Completed all experiments or reached limit of 1000.")




import time
import os
GPU=3
Run = False
hold = False
while True:
    action = 'nvidia-smi'
    state = os.popen(action).read()
    state = state.split('\n')
    for i, line in enumerate(state):
        line_sp = line.split()
        if len(line_sp) > 10 and line_sp[1] in [str(GPU)]:
            if i == len(state) - 1: break
            M_index_0 = state[i + 1].index('MiB')
            M_0 = state[i + 1][:M_index_0].split()[-1]
            M_index_1 = state[i + 1].index('MiB', M_index_0 + 1)
            M_1 = state[i + 1][:M_index_1].split()[-1]
            memory_left = float(M_1) - float(M_0)
            if memory_left > 15000:
                if hold == False:
                    hold = True
                    time.sleep(20)
                else:
                    Run = True
                break
            else:
                hold = False
                Run = False

    print('waiting of node  .......................................')
    time.sleep(20)
    if Run:
        break
os.system('CUDA_VISIBLE_DEVICES=3 sh Inference_weight_0.5_64_20_SCE.sh')
OMP_NUM_THREADS=1 nohup python run_al.py wmt14_en-es mat52 None False > nohup_none.out &;
OMP_NUM_THREADS=1 nohup python run_al.py wmt14_en-es mat52 log False > nohup_log.out &;
OMP_NUM_THREADS=1 nohup python run_al.py wmt14_en-es mat52 tanh1 False > nohup_tanh1.out &;
OMP_NUM_THREADS=1 nohup python run_al.py wmt14_en-es mat52 tanh2 False > nohup_tanh2.out &;
OMP_NUM_THREADS=1 nohup python run_al.py wmt14_en-es mat52 tanh3 False > nohup_tanh3.out &;

@echo off
echo Starting experiments... This may take a long time.

:: --- This command CREATES the log file ---
echo [LOG] Running CIFAR10  M=1000  Balanced > experiments_log.txt 2>&1
python main_OFCL.py --memory_size 1000 >> experiments_log.txt 2>&1

:: --- These commands APPEND to the log file ---
echo [LOG] Running CIFAR10  M=500  Balanced >> experiments_log.txt 2>&1
python main_OFCL.py --memory_size 500 >> experiments_log.txt 2>&1

echo [LOG] Running CIFAR10  M=200  Balanced >> experiments_log.txt 2>&1
python main_OFCL.py --memory_size 200 >> experiments_log.txt 2>&1

echo [LOG] Running CIFAR10  M=1000  Reservoir >> experiments_log.txt 2>&1
python main_OFCL.py --memory_size 1000 --update_strategy reservoir >> experiments_log.txt 2>&1

echo [LOG] Running CIFAR10  M=500  Reservoir >> experiments_log.txt 2>&1
python main_OFCL.py --memory_size 500 --update_strategy reservoir >> experiments_log.txt 2>&1

echo [LOG] Running CIFAR10  M=200  Reservoir >> experiments_log.txt 2>&1
python main_OFCL.py --memory_size 200 --update_strategy reservoir >> experiments_log.txt 2>&1

echo [LOG] Running CIFAR100  M=1000  Balanced >> experiments_log.txt 2>&1
python main_OFCL.py --memory_size 1000 --dataset_name cifar100 >> experiments_log.txt 2>&1

echo [LOG] Running CIFAR100  M=2000  Balanced >> experiments_log.txt 2>&1
python main_OFCL.py --memory_size 2000 --dataset_name cifar100 >> experiments_log.txt 2>&1

echo [LOG] Running CIFAR100  M=1000  Reservoir >> experiments_log.txt 2>&1
python main_OFCL.py --memory_size 1000 --update_strategy reservoir --dataset_name cifar100 >> experiments_log.txt 2>&1

echo [LOG] Running CIFAR100  M=2000  Reservoir >> experiments_log.txt 2>&1
python main_OFCL.py --memory_size 2000 --update_strategy reservoir --dataset_name cifar100 >> experiments_log.txt 2>&1

echo All experiments completed. Check experiments_log.txt for results.
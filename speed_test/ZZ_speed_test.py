'''
timeit.timeit(stmt, setup, timer, number)

Parameters

    stmt: This will take the code for which you want to measure the execution time. The default value is “pass”.
    setup: This will have setup details that need to be executed before stmt. The default value is “pass.”
    timer: This will have the timer value, timeit() already has a default value set, and we can ignore it.
    number: The stmt will execute as per the number is given here. The default value is 1000000.
'''

import timeit
from pprint import pprint as pp
import matplotlib.pyplot as plt

setup_code = []
test_code = []

with open('speed_test/ZZ_speed_test_code.py', 'r') as f:
    code_content = f.readlines()
    
for idx, line in enumerate(code_content):
    if '# SETUP-START' in line:
        setup_start_idx = idx
    elif '# SETUP-END' in line:
        setup_end_idx = idx
    elif '# CODE-START' in line:
        code_start_idx = idx
    elif '# CODE-END' in line:
        code_end_idx = idx

setup_code = ''.join(code_content[setup_start_idx : setup_end_idx])
test_code = ''.join(code_content[code_start_idx : code_end_idx])

# print(setup_code)
# print(test_code)

# print(timeit.timeit(stmt=test_code, setup=setup_code, number=3))
# pp(timeit.repeat(stmt=test_code, setup=setup_code, number=1, repeat=10))
timer_results = timeit.repeat(stmt=test_code, setup=setup_code, number=1, repeat=10000)

plt.plot(timer_results, '.')
plt.grid()

plt.show()

import subprocess
import pathlib
import json

for i in range(128):
    # create the run.json file with the start and end
    run_info = {'start': i*8, 'end': (i+1)*8}
    json.dump(run_info, open('data/lander/run.json', 'w'))
    print(run_info)
    # run the big_run.py script
    proc = subprocess.Popen (['python', 'LunarLander_generate.py'], shell=False)
    proc.communicate()   





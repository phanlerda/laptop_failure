import os, subprocess
subprocess.check_call(['python','src/01_preprocess.py'])
subprocess.check_call(['python','src/02_eda.py'])
print('Done.')

#!/usr/bin/env python

import subprocess
import time

subprocess.check_call(['make'])
subprocess.check_call(['sudo', 'nvidia-smi', '-c', '3'])
subprocess.check_call(['sudo', 'nvidia-smi', '-e', '0'])
tic = time.time()
p1 = subprocess.Popen(['./fastSVM', 'tests/color.png', 'packaged.gz', '/tmp/ramdisk/output.gz'])
p2 = subprocess.Popen(['./fastSVM', 'tests/color.png', 'packaged.gz', '/tmp/ramdisk/output.gz'])
p1.wait()
p2.wait()
toc = time.time()
print('Executed in %f seconds'%(toc - tic))

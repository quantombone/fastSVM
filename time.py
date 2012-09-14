#!/usr/bin/env python

import subprocess
import time

subprocess.check_call(['make'])
tic = time.time()
subprocess.check_call(['./fastSVM', 'tests/color.png', 'packaged.gz', 'output.gz'])
toc = time.time()
print('Executed in %f seconds'%(toc - tic))

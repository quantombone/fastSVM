#!/usr/bin/env python

import subprocess
import os

env = os.environ
env['CUDA_PROFILE'] = '1'
env['CUDA_PROFILE_LOG'] = 'cuda_profile.log'
env['CUDA_PROFILE_CSV'] = '1'
env['CUDA_PROFILE_CONFIG'] = 'cuda_profile.config'

with open('cuda_profile.config', 'w') as f:
#    f.write('warp_serialize\n')
    f.write('divergent_branch\n')
#    f.write('gld_incoherent\n')
#    f.write('gst_incoherent\n')

p = subprocess.Popen(
    ['./fastSVM', 'tests/color.png', 'packaged.gz', 'output.gz'], 
    env=env
)
p.wait()

subprocess.check_call(['less', 'cuda_profile.log'])

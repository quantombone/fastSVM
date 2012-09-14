#!/usr/bin/env python

import subprocess
import gzip
import itertools
import struct
import array
import math
import os

test_dir = 'tests'
tests = ['image1'];

def readResults(filename):
    with gzip.open(filename) as stream:
        results = []

        while True:
            headerBytes = stream.read(8)
            if len(headerBytes) == 0:
                return results

            scale, height, width = struct.unpack('<fHH', headerBytes)

            valuesBytes = stream.read(4*height*width)
            values = array.array('f')
            values.fromstring(valuesBytes)
        
            results.append((scale, height, width, values))

def tester(prefix):
    image = '%s.jpg'%prefix
    output = '%s.gz'%prefix
    correct = '%s-correct.gz'%prefix

    try: os.remove(output)
    except: pass
    subprocess.check_call(['./fastSVM', image, 'packaged.gz', output])

    outputResults = readResults(output)
    correctResults = readResults(correct)

    for pos, (outputResult, correctResult) in enumerate(itertools.izip(outputResults, correctResults)):
        oscale, oheight, owidth, ovalues = outputResult
        cscale, cheight, cwidth, cvalues = correctResult

        if (abs(oheight - cheight) == 1):
            continue
        if (abs(owidth - cwidth) == 1):
            continue

        if math.fabs(oscale - cscale) > 1e-3:
            print ('Mismatched scales for %s (%f, %f)'%(prefix, oscale, cscale))
            exit()

        if oheight != cheight:
            print ('Mismatched heights for %s'%prefix)
            exit()

        if owidth != cwidth:
            print ('Mismatched widths for %s'%prefix)
            exit()

        diffs = [(math.fabs(a - b) if b > -1 else 0) for a, b in itertools.izip(ovalues, cvalues)]
        maxdiff = max(diffs)
        if maxdiff > 1e-3:
            print ('Found max diff of %f for %s'%(maxdiff, prefix))
            print ('entry %d'%pos)
#            exit()

        print('@@@ Result is of size (%d x %d) at a scale of %f'%(oheight, owidth, oscale))
        print('@@@ Greatest difference was %f'%maxdiff)

if __name__ == '__main__':
    subprocess.check_call(['make'])

    for test in tests:
        tester(os.path.join(test_dir,test))   

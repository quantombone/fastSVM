#!/usr/bin/env python

import boto
from boto.s3.key import Key
import multiprocessing

def upload(filename):
    k = Key(bucket)
    k.key = filename
    with open(filename, 'rb') as f:
        k.set_contents_from_file(f)
    k.set_acl('public-read')

if __name__ == '__main__':
    manifest = [
        'fastSVM_wrapper.py',
        'bootstrap.sh',
        'packaged.gz',
        'fastSVM.cu',
        'Makefile',
        'input',
        'input_test',
        'cutil_math.h',
    ] 
    connection = boto.connect_s3()
    bucket = connection.lookup('fastsvm')
    p = multiprocessing.Pool(32)
    p.map(upload, manifest)

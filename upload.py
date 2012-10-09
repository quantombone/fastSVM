#!/usr/bin/env python

import boto
from boto.s3.key import Key
import multiprocessing
from ConfigParser import ConfigParser
import os

def upload(filename):
    k = Key(bucket)
    k.key = filename
    print(filename)
    k.set_contents_from_filename(filename)
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

    parser = ConfigParser()
    parser.read(os.path.expanduser('~/.aws'))
    aws_access_key_id = parser.get('Credentials', 'aws_access_key_id')
    aws_access_key_secret = parser.get('Credentials', 'aws_access_key_secret')

    connection = boto.connect_s3(
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_access_key_secret,
    )
    bucket = connection.lookup('fastsvm')
    p = multiprocessing.Pool(32)
    p.map(upload, manifest)

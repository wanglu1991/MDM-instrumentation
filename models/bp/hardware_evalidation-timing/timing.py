#!/usr/bin/env python 
import os
import sys
def main():
    timing=open('./timing_overhead.txt','r')
    performance_model_timing=0
    cache_timing=0
    line_number=0
    for line in timing:
        if(len(line.split(','))==2):
            performance_model_timing+=float(line.split(',')[1])*1000000
        else:
            cache_timing+=float(line)
    print('performance_model_timing')
    print(performance_model_timing)
    print('cache_timing')
    print(cache_timing)
if __name__== '__main__':
    main()

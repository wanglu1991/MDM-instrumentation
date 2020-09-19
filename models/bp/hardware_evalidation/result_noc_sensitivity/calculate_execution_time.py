#!/usr/bin/env python 
import os
import sys
def main(kernel_numbers):
    NoC=[525.0,1050.0,1700.0,2100.0,4200.0,8400.0]
    for i in range(0,6):
        NoC_bandwidth=NoC[i]
        result=open('./GPUMech_'+str(NoC_bandwidth),'r')
        accum_cycle=0
        accum_inst=0
        for line in result:
            ipc=float(line.split(',')[0])
            instruction_count=float(line.split(',')[1])
            cycle=instruction_count/(ipc*20)
            accum_cycle=accum_cycle+cycle
            accum_inst=accum_inst+instruction_count
        total_execution_time=float(accum_cycle)/1.6/1000000
        total_ipc=accum_inst/accum_cycle
        print('GPUMech_:'+str(NoC_bandwidth))
        print(total_execution_time)
        result=open('./GPUMech+_'+str(NoC_bandwidth),'r')
        accum_cycle=0
        accum_inst=0
        for line in result:
            ipc=float(line.split(',')[0])
            instruction_count=float(line.split(',')[1])
            cycle=instruction_count/(ipc*20)
            accum_cycle=accum_cycle+cycle
            accum_inst=accum_inst+instruction_count
        total_execution_time=float(accum_cycle)/1.6/1000000
        total_ipc=accum_inst/accum_cycle
        print('GPUMech+:'+str(NoC_bandwidth))
        print(total_execution_time)
        result=open('./MDM_Queue_'+str(NoC_bandwidth),'r')
        accum_cycle=0
        accum_inst=0
        for line in result:
            ipc=float(line.split(',')[0])
            instruction_count=float(line.split(',')[1])
            cycle=instruction_count/(ipc*20)
            accum_cycle=accum_cycle+cycle
            accum_inst=accum_inst+instruction_count
        total_execution_time=float(accum_cycle)/1.6/1000000
        total_ipc=accum_inst/accum_cycle
        print('MDM_Queue_:'+str(NoC_bandwidth))
        print(total_execution_time)
        result=open('./MDM-MSHR_'+str(NoC_bandwidth),'r')
        accum_cycle=0
        accum_inst=0
        for line in result:
            ipc=float(line.split(',')[0])
            instruction_count=float(line.split(',')[1])
            cycle=instruction_count/(ipc*20)
            accum_cycle=accum_cycle+cycle
            accum_inst=accum_inst+instruction_count
        total_execution_time=float(accum_cycle)/1.6/1000000
        total_ipc=accum_inst/accum_cycle
        print('MDM_MSHR_:'+str(NoC_bandwidth))
        print(total_execution_time)
        result=open('./MDM_'+str(NoC_bandwidth),'r')
        accum_cycle=0
        accum_inst=0
        for line in result:
            ipc=float(line.split(',')[0])
            instruction_count=float(line.split(',')[1])
            cycle=instruction_count/(ipc*20)
            accum_cycle=accum_cycle+cycle
            accum_inst=accum_inst+instruction_count
        total_execution_time=float(accum_cycle)/1.6/1000000
        total_ipc=accum_inst/accum_cycle
        print('MDM:'+str(NoC_bandwidth))
        print(total_execution_time)
if __name__== '__main__':
    main(sys.argv[1])

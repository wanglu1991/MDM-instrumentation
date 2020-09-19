#! /usr/bin/env python
from collections import defaultdict
import os
import sys
def main(kernel_id):
       SM_access=[]
       SM_numbers=20
       warp_numbers=64
       max_warp=0
       for i in range(0,SM_numbers):
           warp_access=[]
           for j in range(0,10000):
               addr_list=[]
               warp_access.append(addr_list)
           SM_access.append(warp_access)
       memory_trace=open('../mem_trace.txt','r')
       line_sm_0=0
       for line in memory_trace:
            access_info=[]
            length=len(line.split(','))
            for i in range(0,length-1):
                access_info.append(line.split(',')[i])
            SM_id=int(line.split(',')[1])
            if(SM_id==0):
                line_sm_0+=1
            SM_access[SM_id].append(access_info)
       print(line_sm_0)
       print(len(SM_access[3]))
       print(SM_access[3][2])
       SM_access_trace=open('SM_trace_'+str(i)+'.txt','w')
       for i in range(0,SM_numbers):
            for j in range(0,len(SM_access[i])):
                print(SM_access[i][j])
                warp_id=list[0]
                pc=list[2]
                if(int(list[3])==1):
                    W_R='W'
                else:
                    W_R='R'
                for n in range(0,len(list)):
                    SM_access_trace.write(warp_id+','+pc+','+W_R+','+list[n]+'\n')
      # print 'merge_size_1:%d' % size[0]
      # print 'merge_size_2:%d' % size[1]
      # print 'merge_size_3:%d' % size[2]
      # print 'merge_size_4:%d' % size[3]
      # print 'merge_size_5:%d' % size[4]
      # print 'merge_size_6:%d' % size[5]
      # print 'merge_size_7:%d' % size[6]
      # print 'merge_size_8:%d' % size[7]
       # time=int(line.split(',')[-1])
      # creation_time=int(line.split(',')[-1])
        #queue_time=time-creation_time
        #amount_queue_time+=queue_time
if __name__=='__main__':
   main(sys.argv[1])
      

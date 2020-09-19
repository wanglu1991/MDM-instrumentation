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
           for j in range(0,30000):
               addr_list=[]
               warp_access.append(addr_list)
           SM_access.append(warp_access)
       memory_trace=open('../mem_trace_'+kernel_id+'.txt','r')
       for line in memory_trace:
            access_info=[]
            length=len(line.split(','))
            for i in range(0,length-1):
                access_info.append(line.split(',')[i])
            warp_id=int(line.split(',')[0])
            if (warp_id>max_warp):
                max_warp=warp_id
            #if((warp_id==0)and(max_warp>100)):
            #    break
            pc=int(line.split(',')[2])
            SM_id=(warp_id//warp_numbers)%SM_numbers
            #SM_id=int(line.split(',')[1])
            round=int(warp_id//(warp_numbers*SM_numbers))
            warp=round*warp_numbers+(warp_id%warp_numbers)
            if (warp<10000*20):
                SM_access[SM_id][warp].append(access_info)
            #if(warp_id<10000):
               # warp_access[warp_id].append(access_info)
    
       print(len(warp_access))
       for i in range(0,SM_numbers):
           SM_access_trace=open('SM_trace_'+str(i)+'.txt','w')
           for j in range(0,int(len(SM_access[i])/warp_numbers)):
                    k=0
                    empty=0
                    while(empty<warp_numbers):
                        if(len(SM_access[i][j*warp_numbers+k])>0):
                            list=SM_access[i][j*warp_numbers+k][0]
                            warp_id=list[0]
                            pc=list[2]
                            if(int(list[3])==1):
                                W_R='R'
                            else:
                                W_R='W'
                            for n in range(4,len(list)):
                                SM_access_trace.write(warp_id+','+pc+','+W_R+','+list[n]+'\n')
                            SM_access[i][j*warp_numbers+k].pop(0)
                        else:
                            empty+=1
                        k=(k+1)%warp_numbers
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
      

#! /usr/bin/env python
from collections import defaultdict
import os
import sys
import math
import kmeans
import numpy
import numpy as np
class interval(object):
    def __init__(self):
        self.numMshrs=32
        self.avg_stall_cycles=175
        self.exp_reqs=0
        self.read_list=[] #READ MISS in representative warp
        self.DRAM_READ_LIST=[]
        self.DRAM_WRITE_LIST=[]
        self.region_instructions={}
        self.warp_pc_list={}
        self.warp_index=0
        self.queuing_delay_mshr=[]
        self.nonoverlap_insts=0
        self.accum_inst=0
        self.accum_stall=0
        self.warp_numbers=64
        self.dram_bandwidth=720
        self.activeCores=60
        self.freq_core=1.4
        self.max_issue=4.0
        self.X=[]
        self.Clusters=2
        self.total_instruction_counts=0
        self.ipc_list=[]
        self.instruction_counts_list=[]
    def calculate_avg_stall_cycles(self):
        i=0
        total_access=0
        miss=0
        L2_total=open('./output_L2','r')
        for line in L2_total:
            i+=1
            if(i==1):
                total_access=int(line.split(':')[-1])
                print(total_access)
            if((i==3)or(i==5)):
                miss+=int(line.split(':')[-1])
        miss_rate=float(miss)/float(total_access)
        self.avg_stall_cycles=120*(1-miss_rate)+220*miss_rate
        print(self.avg_stall_cycles)
    def RR_regioninsts(self):
        i=0
        while(i<len(self.region_instructions[self.warp_index])-1):
            current_inst=self.region_instructions[self.warp_index][i]
            self.nonoverlap_insts+=(self.warp_numbers-1)*(current_inst-1)
            if (self.region_instructions[self.warp_index][i+1]<(self.warp_numbers-1)/self.max_issue):
                self.nonoverlap_insts+=(self.warp_numbers-1)/self.max_issue-self.region_instructions[self.warp_index][i+1]
            self.accum_inst+=self.region_instructions[self.warp_index][i]
            self.accum_stall+=self.region_instructions[self.warp_index][i+1]
            i+=2
        issue_ration=float(self.accum_inst)/(float(self.accum_inst)/self.max_issue+float(self.accum_stall))*self.warp_numbers
        if (issue_ration>self.max_issue):
            issue_ration=self.max_issue
        self.nonoverlap_insts=self.nonoverlap_insts*issue_ration
        return self.nonoverlap_insts
#def multi_RR(nonoverlap_insts,warp_numbers):
    def IPC_full(self):
        print("naive_ipc")
        ipc=float(self.accum_inst)/(float(self.accum_inst)/self.max_issue+float(self.accum_stall))*self.warp_numbers
        if(ipc>self.max_issue):
            ipc=self.max_issue
        print(ipc)
        print('Multithreading_IPC')
        ipc=float(self.accum_inst)/(float(self.accum_inst)/self.max_issue+float(self.accum_stall)+float(self.nonoverlap_insts)/self.max_issue)*self.warp_numbers
        if(ipc>self.max_issue):
            ipc=self.max_issue
        print(ipc)
        print("With MSHR modeling")
        MSHR_delay_total=0
        for i in self.queuing_delay_mshr:
            MSHR_delay_total+=i
        print(MSHR_delay_total)
        ipc=float(self.accum_inst)/(float(self.accum_inst)/self.max_issue+float(self.accum_stall)+float(self.nonoverlap_insts)/self.max_issue+float(MSHR_delay_total))*self.warp_numbers
        if (ipc>self.max_issue):
            ipc=self.max_issue
        print(ipc)
        print("With DRAM modeling")
        DRAM_delay_total=0
        for j in self.queuing_delay_dram:
            DRAM_delay_total+=j
        print(DRAM_delay_total)
        cycles=float(self.accum_inst)/self.max_issue+float(self.accum_stall)+float(self.nonoverlap_insts)/self.max_issue+float(MSHR_delay_total)+float(DRAM_delay_total)
        ipc=float(self.accum_inst)*self.warp_numbers/float(cycles)
        print(ipc)
        print(self.total_instruction_counts)
        result=open('result.txt','a')
        result.write(str(ipc)+','+str(self.total_instruction_counts)+'\n')
    def kmeans_operation(self):
        KmeansInstance=kmeans.Kmeans()
        self.rep_warp,self.weights=KmeansInstance.RunKmeans(self.X,self.Clusters)
        self.warp_index=0
        weight=0
        for i in range(0,len(self.rep_warp)):
            if(self.weights[i]>weight):
                weight=self.weights[i]
                self.warp_index=self.rep_warp[i]
       # print(self.warp_index)

    def generate_feature_vector(self):
    	ipc_list=[]
    	instruction_counts=[]
    	interval_profile=open('./warp_perf.txt','r')
    	for line in interval_profile:
        	ipc_list.append(float(line.split(',')[0]))
        	instruction_counts.append(int(line.split(',')[1]))
    	ipc_avg=np.mean(ipc_list)
    	print('MIN_IPC')
    	print(min(ipc_list))
    	print('MAX_IPC')
    	print(max(ipc_list))
    	print(ipc_avg)
    	instruction_counts_avg=np.mean(instruction_counts)
    	print(instruction_counts_avg)
    	for i in range(0,len(ipc_list)):
        	a=[float(ipc_list[i])/float(ipc_avg),float(instruction_counts[i])/float(instruction_counts_avg)]
        	self.X.append(a)
     	

    

    def Get_DRAM_info(self):
           L2_info=open('./output_L2_access.txt','r')
           instruction_index=0
           print('DRAM info')
           for i in range(0,len(self.region_instructions[self.warp_index])/2):
                self.DRAM_READ_LIST.append(0)
                self.DRAM_WRITE_LIST.append(0)
           for line in L2_info:
                warp_id=int(line.split(',')[0])
                if(warp_id==self.warp_index):
                    pc=int(line.split(',')[1])
                    if (int(line.split(',')[-1])==0): #L2_miss
                       # print(pc)
                        while(instruction_index<len(self.warp_pc_list[warp_id])):
                            if(pc!=self.warp_pc_list[warp_id][instruction_index]):
                                instruction_index+=1
                            else:
                                break
                        temp=0
                        accum=0
                        while(accum<instruction_index):
                            accum+=self.region_instructions[warp_id][temp]
                            temp+=2
                        index=(temp-2)/2
                        if(line.split(',')[2]=='R'):
                            self.DRAM_READ_LIST[index]+=1
                        else:
                            self.DRAM_WRITE_LIST[index]+=1
    
    def warp_vector(self):
        for i in self.region_instructions:
            instruction_numbers=0
            total_cycles=0
            for j in range(0,len(self.region_instructions[i])/2):
                instruction_numbers+=self.region_instructions[i][2*j]
                total_cycles+=float(self.region_instructions[i][2*j])/self.max_issue+float(self.region_instructions[i][2*j+1])
            ipc=float(instruction_numbers)/float(total_cycles)
            self.ipc_list.append(ipc)
            self.instruction_counts_list.append(instruction_numbers)
        avg_ipc=np.mean(self.ipc_list)
        avg_instruction_counts=np.mean(self.instruction_counts_list)
        self.total_instruction_counts=np.sum(self.instruction_counts_list)
        for i in range(0,len(self.ipc_list)):
            a=[float(self.ipc_list[i])/float(avg_ipc),float(self.instruction_counts_list[i])/float(avg_instruction_counts)]
            self.X.append(a)
        print(len(self.X))
    
    
    
    
    def contention_DRAM_modeling(self):
        
        for i in range(0,len(self.region_instructions[self.warp_index])/2):
            if (self.DRAM_READ_LIST[i] + self.DRAM_WRITE_LIST[i]) >= 1.0:
                # self.dram_bandwidth = 192.0
                waiting_time = 0.0                

                # self.exp_reqs = (regionInfo["DRAMReads"] + regionInfo["DRAMWrites"]) * self.activeWarps[benchName]                
                
                all_reqs = (min(self.DRAM_READ_LIST[i]*self.warp_numbers,self.numMshrs) +\
                                self.DRAM_WRITE_LIST[i] * self.warp_numbers )* self.activeCores

                service_time = 128.0 / self.dram_bandwidth*self.freq_core

               # service_time = 128.0/self.dram_bandwidth

                service_time_sum = all_reqs * service_time

                service_time_sum2 = all_reqs * math.pow(128.0 / self.dram_bandwidth, 2)

                if self.avg_stall_cycles == 0:
                    self.avg_stall_cycles = 400                

                utilization = float(service_time_sum) / self.avg_stall_cycles
                #utilization = float(service_time_sum) /float(self.region_instructions[self.warp_index][2*i]+self.region_instructions[self.warp_index][2*i+1])
                # print utilization, service_time_sum, self.avg_stall_cycles
                if utilization > 1.0:
                    utilization = 0.99
                    # service_time_Es2 = service_time_sum2 / all_reqs;
                    # arrival_rate = all_reqs / self.avg_stall_cycles
                    # delay_dram = arrival_rate * service_time_Es2 / (2 * (1.0 - utilization));
                    delay_dram = service_time * all_reqs / 2.0                    
                else:
                    if all_reqs > 0:
                        service_time_Es2 = service_time_sum2 / all_reqs;
                    else:
                        service_time_Es2 = 0
                   # arrival_rate = all_reqs/float(self.region_instructions[self.warp_index][2*i]+self.region_instructions[self.warp_index][2*i+1])
                    arrival_rate = all_reqs / self.avg_stall_cycles
                    delay_dram = arrival_rate * service_time_Es2 / (2 * (1.0 - utilization));
                
                # print delay_dram, all_reqs
                                                        
                self.queuing_delay_dram.append(delay_dram)                


    def contention_modeling_MSHR(self):                
        self.exp_reqs = 0
      #  N = self.activeWarps[benchName]
      #  avg_req = 0.0        
      #  prob = 0.0
      #  utilization = 0.0
      #  region_read=[]
        self.queuing_delay_mshr = []
        self.queuing_delay_dram = []
        for i in range(0,len(self.read_list)):
            if(self.read_list[i]>=1):
                delay_mshr = 0                                               
                # self.exp_reqs = regionInfo["Reads"] * self.activeWarps[benchName] / float(regionInfo["Read_Insts"])
                self.exp_reqs =self.read_list[i] * self.warp_numbers
                if self.exp_reqs > self.numMshrs:                
                    for j in range(1, int(self.exp_reqs + 1)):
                        if self.exp_reqs > 0:                            
                            delay_mshr += float(self.avg_stall_cycles * (math.ceil(float(j) / self.numMshrs)) * 1.0) / float(self.exp_reqs)
                        else:
                            delay_mshr = 0                    
                        
                    # print "mshrs", self.numMshrs
                    # delay_mshr = (delay_mshr - self.avg_stall_cycles) * regionInfo["Read_Insts"]
                    delay_mshr -= self.avg_stall_cycles                    
                    # print delay_mshr
                # print self.exp_reqs, self.numMshrs, delay_mshr
                self.queuing_delay_mshr.append(delay_mshr)
        

    def warp_interval_profile(self,kernel_id):
           warp_interval={}
           warp_max_issue_cycle={}
           pc_list=[]
           latency_list={}
           interval_inst={}
           dependence_trace=open('../interval_info_'+str(kernel_id)+'.txt','r')
           pc_latency=open('./pc_latency_info','r')
           for line in pc_latency:
               pc=int(line.split(',')[0])
               pc_list.append(int(line.split(',')[0]))
               latency_list[pc]=float(line.split(',')[1])
      # output_trace=open('single_warp_info.txt','w')
           for line in dependence_trace:
                warp_id=int(line.split(',')[0])
                if ((warp_id==0)and(self.warp_pc_list.get(1000,0)!=0)):
                    break
                if(warp_interval.get(warp_id,0)==0):
                     warp_interval[warp_id]={}
                     self.region_instructions[warp_id]=[]
                     interval_inst[warp_id]=0
                     warp_max_issue_cycle[warp_id]=0
                     self.warp_pc_list[warp_id]=[]
                pc=int(line.split(',')[1])
                self.warp_pc_list[warp_id].append(pc)
                if(warp_interval[warp_id].get(pc,0)==0):
                     warp_interval[warp_id][pc]=[]
                latency=int(line.split(',')[-1])
                if(pc in pc_list):
                    latency=latency_list[pc]
                issue_cycle=warp_max_issue_cycle[warp_id]+1
                dependence_pc_list=[]
                for i in range(3,len(line.split(','))-1):
                    dependence_pc_list.append(int(line.split(',')[i]))
                for j in range(0,len(dependence_pc_list)):
                    dep_pc=dependence_pc_list[j]
                    if(warp_interval[warp_id].get(dep_pc,0)!=0):
                        if(len(warp_interval[warp_id][dep_pc])>0):
                            done_cycle=warp_interval[warp_id][dep_pc][-1]
                            if((done_cycle+1)>(issue_cycle)):
                                 issue_cycle=done_cycle+1
                if((issue_cycle-warp_max_issue_cycle[warp_id])==1):
                    interval_inst[warp_id]+=1
                else:
                    stall_cycles=issue_cycle-warp_max_issue_cycle[warp_id]
                    self.region_instructions[warp_id].append(interval_inst[warp_id])
                    self.region_instructions[warp_id].append(stall_cycles)
                    interval_inst[warp_id]=1
                warp_interval[warp_id][pc].append(issue_cycle)
                warp_interval[warp_id][pc].append(issue_cycle+latency)
         # output_trace.write(str(warp_id)+','+str(pc)+','+str(issue_cycle)+',')
          # output_trace.write(str(issue_cycle+latency))
          # output_trace.write('\n')
                warp_max_issue_cycle[warp_id]=issue_cycle
           for warp in self.region_instructions:
                self.region_instructions[warp].append(interval_inst[warp])
                self.region_instructions[warp].append(0)
       # get the MSHR access numbers for each interval
           i=(self.warp_index/self.warp_numbers)%self.activeCores
           L1_hit_info=open('./read_out_'+str(i)+'_L1_hit.txt','r')
           instruction_index=0
           for i in range(0,len(self.region_instructions[self.warp_index])/2):
                self.read_list.append(0)
           for line in L1_hit_info:
                warp_id=int(line.split(',')[0])
                if(warp_id==self.warp_index):
                    pc=int(line.split(',')[1])
                    print(pc)
                    if ((int(line.split(',')[-1])==0)and(line.split(',')[2]=='R')): #L1 miss
                        while(instruction_index<len(self.warp_pc_list[warp_id])):
                            if((pc!=self.warp_pc_list[warp_id][instruction_index])):
                                instruction_index+=1
                            else:
                                break
                        temp=0
                        accum=0
                        while(accum<instruction_index):
                             accum+=self.region_instructions[warp_id][temp]
                             temp+=2
                        index=(temp-2)/2
                        self.read_list[index]+=1
             




def main(kernel_id):
    intervalInstance=interval()
    intervalInstance.calculate_avg_stall_cycles()
    intervalInstance.warp_interval_profile(kernel_id)
    intervalInstance.warp_vector()
   # intervalInstance.generate_feature_vector()
    intervalInstance.kmeans_operation()

    intervalInstance.RR_regioninsts()
    intervalInstance.contention_modeling_MSHR()
    intervalInstance.Get_DRAM_info()
    intervalInstance.contention_DRAM_modeling()
    intervalInstance.IPC_full()
   
   
   # RR_regioninsts(region_instructions,512,24)
     # warp_feature=open('warp_vector.txt','w')
      # for warp in warp_interval:
      #     instruction_count=0
      #    for pc in warp_interval[warp]:
      #          instruction_count+=len(warp_interval[warp][pc])/2
      #     IPC=float(instruction_count)/float(warp_max_issue_cycle[warp])
      #     warp_feature.write(str(IPC)+','+str(instruction_count)+'\n')
      # interval_profile=open('warp_interval_profile.txt','w')
      # for warp in region_instructions:
      #     for j in range(0,len(region_instructions[warp])):
      #         if warp==0:
      #             print(region_instructions[warp])
              # interval_profile.write(str(warp)+','+str(region_instructions[warp][j])+','+str(region_instructions[warp][j+1])+'\n')
              # j+=2 

if __name__=='__main__':
   main(sys.argv[1])
      

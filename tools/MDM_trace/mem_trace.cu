/* Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <string>
#include <map>
#include <set>
/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static __managed__ ChannelDev channel_dev_inst;
static ChannelHost channel_host;
static ChannelHost channel_host_inst;
/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;
int cache_line_size = 128;
/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
int kernel_id=0;
/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* information collected in the instrumentation function */
typedef struct {
    int cta_id_x;
    int cta_id_y;
    int cta_id_z;
    int warp_id;
    int opcode_id;
    int offset;
    int RoW;
    int sm_id;
    uint64_t addrs[32];
} mem_access_t;

typedef struct
{ int warp_id;
  int opcode_id;
  int offset;
  int des_reg;
  int source_reg_1;
  int source_reg_2;
}instruction_t;







/* instrumentation function that we want to inject, please note the use of
 * 1. "extern "C" __device__ __noinline__" to prevent code elimination by the
 * compiler.
 * 2. NVBIT_EXPORT_FUNC(count_instrs) to notify nvbit the name of the function
 * we want to inject. This name must match exactly the function name */
extern "C" __device__ __noinline__ void dep_instrs(int predicate,
                                                     int count_warp_level,int offset,int opcode_id, int des_reg=10000,int reg_2=10000,int reg_3=10000) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot(1);
    /* compute the predicate mask */
    const int predicate_mask = __ballot(predicate);
    /* each thread will get a lane id (get_lane_id is in utils/utils.h) */
    const int laneid = get_laneid();
    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;
    /* count all the active thread */
    const int num_threads = __popc(predicate_mask);
    const int warp_id =get_global_warp_id();
    instruction_t ta;
    ta.warp_id = warp_id;
    ta.offset = offset;
    ta.opcode_id = opcode_id;
    ta.des_reg = des_reg;
    if(reg_2<1000)
    	ta.source_reg_1=reg_2;
    if(reg_3<1000)
    	ta.source_reg_2=reg_3;

    /* only the first active thread will perform the atomic */
    // inst->print();
    if (first_laneid == laneid)
            {
        if (count_warp_level)
            {
            /* num threads can be zero when accounting for predicates off */
            if (num_threads > 0)
            {atomicAdd((unsigned long long *)&counter, 1);
             channel_dev_inst.push(&ta, sizeof(instruction_t));
            if(warp_id==1)
            //FILE * f =fopen("./instruction_trace.txt","a");
            //if(f!=NULL)
            printf("%d,%d,%d,%d,%d,%d\n",warp_id,offset,opcode_id,des_reg,reg_2,reg_3);
            //fclose(f);
            }
            else
            {
            atomicAdd((unsigned long long *)&counter, num_threads);
            }
            }
            }
    }
NVBIT_EXPORT_FUNC(dep_instrs);

/* Instrumentation function that we want to inject, please note the use of
 * 1. extern "C" __device__ __noinline__
 *    To prevent "dead"-code elimination by the compiler.
 * 2. NVBIT_EXPORT_FUNC(dev_func)
 *    To notify nvbit the name of the function we want to inject.
 *    This name must match exactly the function name.
 */
extern "C" __device__ __noinline__ void instrument_mem(int pred, int opcode_id,
                                                       uint32_t reg_high,
                                                       uint32_t reg_low,
                                                       int32_t imm,int offset,int RoW) {
    if (!pred) {
        return;
    }

    int64_t base_addr = (((uint64_t)reg_high) << 32) | ((uint64_t)reg_low);
    uint64_t addr = base_addr + imm;

    int active_mask = __ballot(1);
    const int laneid = get_laneid();
    const int first_laneid = __ffs(active_mask) - 1;

    mem_access_t ma;
    /* collect memory address information */
    for (int i = 0; i < 32; i++) {
        ma.addrs[i] = __shfl(addr, i);
    }

    int4 cta = get_ctaid();
    ma.cta_id_x = cta.x;
    ma.cta_id_y = cta.y;
    ma.cta_id_z = cta.z;
    ma.warp_id = get_global_warp_id();
    ma.sm_id = get_smid();
    ma.opcode_id = opcode_id;
    ma.offset  = offset;
    ma.RoW =RoW;
    /* first active lane pushes information on the channel */
    if (first_laneid == laneid) {
        channel_dev.push(&ma, sizeof(mem_access_t));
    }
}
NVBIT_EXPORT_FUNC(instrument_mem);

void nvbit_at_init() {
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

/* instrument each memory instruction adding a call to the above instrumentation
 * function */
void nvbit_at_function_first_load(CUcontext ctx, CUfunction f) {
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (verbose) {
        printf("Inspecting function %s at address 0x%lx\n",
               nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
    }

    uint32_t cnt = 0;
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
        if (cnt < instr_begin_interval || cnt >= instr_end_interval ||
            instr->getMemOpType() == Instr::NONE) {
            cnt++;
            continue;
        }
        if (verbose) {
            instr->printDecoded();
        }

        if (opcode_to_id_map.find(instr->getOpcode()) ==
            opcode_to_id_map.end()) {
            int opcode_id = opcode_to_id_map.size();
            opcode_to_id_map[instr->getOpcode()] = opcode_id;
            printf("OPCODE %s MAPS TO ID %d\n",i->getOpcode(),opcode_id);
            id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
        }

        int opcode_id = opcode_to_id_map[instr->getOpcode()];
        int offset = instr->getOffset();

        /* instrument for instruction trace */

        {
        nvbit_insert_call(i, "dep_instrs", IPOINT_BEFORE);
                  if (exclude_pred_off) {
                      /* pass predicate value */
                      nvbit_add_call_arg_pred_val(i);
                  } else {
                      /* pass always true */
                      nvbit_add_call_arg_const_val32(i, 1);
                  }

        /* add count warps option */
        nvbit_add_call_arg_const_val32(i, count_warp_level);
        /* add instruction pc */
        nvbit_add_call_arg_const_val32(i,offset);
        /* add opcode */
        nvbit_add_call_arg_const_val32(i,opcode_id);
        //if(!i->isStore())
                 // {
        for (int j=0;j<i->getNumOperands();j++)
         {const Instr::operand_t * op=i->getOperand(j);/*get each operand*/
          if((op->type==Instr::REG))
          nvbit_add_call_arg_const_val32(i,op->value[0]);/* get register_id*/
          else
          {
             if(j==0)
             nvbit_add_call_arg_const_val32(i,10000);
           }
         }
        }

        /* instrument for memory trace */
        int RoW = 0;
        if(instr->isLoad())
        RoW=1;
        /* iterate on the operands */
        for (int i = 0; i < instr->getNumOperands(); i++) {
            /* get the operand "i" */
            const Instr::operand_t *op = instr->getOperand(i);

            if ((op->type == Instr::MREF)&&(instr->getMemOpType()==Instr::GLOBAL)) {
                /* insert call to the instrumentation function with its
                    * arguments */
                nvbit_insert_call(instr, "instrument_mem", IPOINT_BEFORE);
                nvbit_add_call_arg_pred_val(instr);
                nvbit_add_call_arg_const_val32(instr, opcode_id);
                if (instr->isExtended()) {
                    nvbit_add_call_arg_reg_val(instr, (int)op->value[0] + 1);
                } else {
                    nvbit_add_call_arg_reg_val(instr, (int)Instr::RZ);
                }
                nvbit_add_call_arg_reg_val(instr, (int)op->value[0]);
                nvbit_add_call_arg_const_val32(instr, (int)op->value[1]);
                nvbit_add_call_arg_const_val32(instr,offset);
                nvbit_add_call_arg_const_val32(instr,RoW);
            }
        }
        cnt++;
    }
}

__global__ void flush_channel() {
    /* push memory access with negative cta id to communicate the kernel is
     * completed */
    mem_access_t ma;
    ma.cta_id_x = -1;
    channel_dev.push(&ma, sizeof(mem_access_t));

    /* flush channel */
    channel_dev.flush();

    instruction_t ta;
    ta.warp_id=-1;
    channel_dev_inst.push(&ta, sizeof(instruction_t));

       /* flush channel */
    channel_dev_inst.flush();
}



/*
__global__ void flush_channel() {
    //push memory access with negative cta id to communicate the kernel is completed

}
*/
void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    if (skip_flag) return;

    if (cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchKernel) {
        cuLaunchKernel_params *p = (cuLaunchKernel_params *)params;

        if (!is_exit) {
            int nregs;
            CUDA_SAFECALL(
                cuFuncGetAttribute(&nregs, CU_FUNC_ATTRIBUTE_NUM_REGS, p->f));

            int shmem_static_nbytes;
            CUDA_SAFECALL(cuFuncGetAttribute(&shmem_static_nbytes,
                                          CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES,
                                          p->f));

            
            printf(
                "Kernel %s - grid size %d,%d,%d - block size %d,%d,%d - nregs "
                "%d - shmem %d - cuda stream id %ld\n",
                nvbit_get_func_name(ctx, p->f), p->gridDimX, p->gridDimY,
                p->gridDimZ, p->blockDimX, p->blockDimY, p->blockDimZ, nregs,
                shmem_static_nbytes + p->sharedMemBytes, (uint64_t)p->hStream);
            
            recv_thread_receiving = true;

        } else {
            kernel_id++;
            /* make sure current kernel is completed */
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);

            /* make sure we prevent re-entry on the nvbit_callback when issuing
             * the flush_channel kernel */
            skip_flag = true;

            /* issue flush of channel so we are sure all the memory accesses
             * have been pushed */
            flush_channel<<<1, 1>>>();
            cudaDeviceSynchronize();
            assert(cudaGetLastError() == cudaSuccess);

            /* unset the skip flag */
            skip_flag = false;

            /* wait here until the receiving thread has not finished with the
             * current kernel */
            while (recv_thread_receiving) {
                pthread_yield();
            }
        }
    }
}

void *recv_thread_fun(void *) {
    char *recv_buffer = (char *)malloc(CHANNEL_SIZE);
    std::map<int, std::vector<mem_access_t *>> per_warp_mem_trace;

    while (recv_thread_started) {
        uint32_t num_recv_bytes = 0;
        if (recv_thread_receiving &&
            (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) >
                0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                mem_access_t *ma =
                    (mem_access_t *)&recv_buffer[num_processed_bytes];

                /* when we get this cta_id_x it means the kernel has completed
                 */
                if (ma->cta_id_x == -1) {
                    recv_thread_receiving = false;
                    break;
                }
                int warp_id =ma->warp_id;
                per_warp_mem_trace[warp_id].push_back(ma);
                char fn[100];
                snprintf(fn,sizeof(fn),"./mem_trace_%d.txt",kernel_id);
                FILE * f = fopen(fn,"a");
                if(f!=NULL)
                {fprintf(f,"%d,%d,%d,%d,",ma->warp_id,ma->sm_id,ma->offset,ma->RoW);
                 std::set<uint64_t> coalesced_addr;
                 for(int i=0;i<32;i++)
                 {uint64_t cache_line_addr=ma->addrs[i]/cache_line_size;
                  coalesced_addr.insert(cache_line_addr);
                 }
                 for(std::set<uint64_t>:: iterator addr = coalesced_addr.begin(); addr!=coalesced_addr.end();++addr)
                 fprintf(f,"%lld,",*addr);
                 fprintf(f,"\n");
                }
                fclose(f);
               /*
                printf("CTA %d,%d,%d - warp %d - %s - ", ma->cta_id_x,
                       ma->cta_id_y, ma->cta_id_z, ma->warp_id,
                       id_to_opcode_map[ma->opcode_id].c_str());
                for (int i = 0; i < 32; i++) {
                    printf("0x%016lx ", ma->addrs[i]);
            
                }
                printf("\n");
                */
                num_processed_bytes += sizeof(mem_access_t);
            }
        }
    }

    while (recv_thread_started) {
          uint32_t num_recv_bytes = 0;
          if (recv_thread_receiving &&
              (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) >
                  0) {
              uint32_t num_processed_bytes = 0;
              while (num_processed_bytes < num_recv_bytes) {
                  mem_access_t *ma =
                      (mem_access_t *)&recv_buffer[num_processed_bytes];

                  /* when we get this cta_id_x it means the kernel has completed
                   */
                  if (ma->cta_id_x == -1) {
                      recv_thread_receiving = false;
                      break;
                  }
                  int warp_id =ma->warp_id;
                  per_warp_mem_trace[warp_id].push_back(ma);
                  char fn[100];
                  snprintf(fn,sizeof(fn),"./mem_trace_%d.txt",kernel_id);
                  FILE * f = fopen(fn,"a");
                  if(f!=NULL)
                  {fprintf(f,"%d,%d,%d,%d,",ma->warp_id,ma->sm_id,ma->offset,ma->RoW);
                   std::set<uint64_t> coalesced_addr;
                   for(int i=0;i<32;i++)
                   {uint64_t cache_line_addr=ma->addrs[i]/cache_line_size;
                    coalesced_addr.insert(cache_line_addr);
                   }
                   for(std::set<uint64_t>:: iterator addr = coalesced_addr.begin(); addr!=coalesced_addr.end();++addr)
                   fprintf(f,"%lld,",*addr);
                   fprintf(f,"\n");
                  }
                  fclose(f);
                 /*
                  printf("CTA %d,%d,%d - warp %d - %s - ", ma->cta_id_x,
                         ma->cta_id_y, ma->cta_id_z, ma->warp_id,
                         id_to_opcode_map[ma->opcode_id].c_str());
                  for (int i = 0; i < 32; i++) {
                      printf("0x%016lx ", ma->addrs[i]);

                  }
                  printf("\n");
                  */
                  num_processed_bytes += sizeof(mem_access_t);
              }
          }
      }



    /*
    for(std::map<int,std::vector<mem_access_t *>>::iterator it=per_warp_mem_trace.begin(); it!=per_warp_mem_trace.end();++it)
    {std::vector<mem_access_t *> trace = it->second;
     FILE * f =fopen("./mem_trace.txt","a");
     if(f!=NULL)
     {
     for (int i =0; i< trace.size();i++)
     {fprintf(f,"%d, %d,%d,%d,",trace[i]->warp_id,trace[i]->sm_id,trace[i]->offset,trace[i]->RoW);//warp_id, pc, RoW
      std::set<uint64_t> coalesced_addr;     
      for (int j=0; j< 32; j++)
      {   uint64_t cache_line_addr = trace[i]->addrs[j]/cache_line_size;
          coalesced_addr.insert(cache_line_addr);
      }
      for (std::set<uint64_t>:: iterator addr=coalesced_addr.begin();addr!=coalesced_addr.end();++addr)
         fprintf(f,"%lld,",*addr);
         fprintf(f,"\n");
     }
     }
     fclose(f);
    }
    */
    free(recv_buffer);
    return NULL;
}

void nvbit_at_ctx_init(CUcontext ctx) {
    recv_thread_started = true;
    channel_host.init(0, CHANNEL_SIZE, &channel_dev, NULL);
    pthread_create(&recv_thread, NULL, recv_thread_fun, NULL);
}

void nvbit_at_ctx_term(CUcontext ctx) {
    if (recv_thread_started) {
        recv_thread_started = false;
        pthread_join(recv_thread, NULL);
    }
}

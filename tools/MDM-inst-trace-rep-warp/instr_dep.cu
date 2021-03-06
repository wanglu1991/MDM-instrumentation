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
#include <vector>
/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* for channel */
#include "utils/channel.hpp"

/* Channel used to communicate from GPU to CPU receiving thread */
#define CHANNEL_SIZE (1l << 20)
static __managed__ ChannelDev channel_dev;
static ChannelHost channel_host;
/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;

/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;
int cache_line_size = 128;
int exclude_pred_off = 0;
int count_warp_level = 1;
/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;
__managed__ int kernel_id=0;
__managed__ int rep_warp[1000][1000]={-1};
/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;
__managed__ uint64_t counter = 0;
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
  int source_reg_3;
}instruction_t;







/* instrumentation function that we want to inject, please note the use of
 * 1. "extern "C" __device__ __noinline__" to prevent code elimination by the
 * compiler.
 * 2. NVBIT_EXPORT_FUNC(count_instrs) to notify nvbit the name of the function
 * we want to inject. This name must match exactly the function name */
extern "C" __device__ __noinline__ void dep_instrs(int predicate,
                                                     int count_warp_level,int offset,int opcode_id, int des_reg=10000,int reg_2=10000,int reg_3=10000,int reg_4=10000) {
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
    int push=0;
    for(int i=0;i<1000;i++)
    {if(rep_warp[kernel_id][i]==-1)
    	break;
     if(warp_id==rep_warp[kernel_id][i])
     {push=1;

      break;
      }
     else
    	i++;

    }
    if(push)
    {
    ta.offset = offset;
    ta.opcode_id = opcode_id;
    ta.des_reg = des_reg;
    if(reg_2<1000)
    	ta.source_reg_1=reg_2;
    if(reg_3<1000)
    	ta.source_reg_2=reg_3;
    if(reg_4<1000)
        ta.source_reg_3=reg_4;
    }
    /* only the first active thread will perform the atomic */
    //inst->print();
    if (first_laneid == laneid)
            {
        if (count_warp_level)
            {
            /* num threads can be zero when accounting for predicates off */
            if (num_threads > 0)
            {atomicAdd((unsigned long long *)&counter, 1);
            if(push)
            channel_dev.push(&ta, sizeof(instruction_t));

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
    //read file, select rep_warps in each path
       FILE * rep_warp_info = fopen("./bb_trace/rep_warp.txt","r");
       if(rep_warp_info!=NULL)
         {  int k;
            int warp_id;
            int i=0;
            while(fscanf(rep_warp_info,"%d,%d",&k,&warp_id)!=EOF)
            {rep_warp[k][i]=warp_id;
             i++;
           }

           }

           fclose(rep_warp_info);
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
        if (cnt < instr_begin_interval || cnt >= instr_end_interval)
            break;
            
        
        if (verbose) {
            instr->printDecoded();
        }

        if (opcode_to_id_map.find(instr->getOpcode()) ==
            opcode_to_id_map.end()) {
            int opcode_id = opcode_to_id_map.size();
            opcode_to_id_map[instr->getOpcode()] = opcode_id;
            printf("OPCODE %s MAPS TO ID %d\n",instr->getOpcode(),opcode_id);
            id_to_opcode_map[opcode_id] = std::string(instr->getOpcode());
        }
        instr->print();
        int opcode_id = opcode_to_id_map[instr->getOpcode()];
        int offset = instr->getOffset();

        /* instrument for instruction trace */

        {
        nvbit_insert_call(instr, "dep_instrs", IPOINT_BEFORE);
                  if (exclude_pred_off) {
                      /* pass predicate value */
                      nvbit_add_call_arg_pred_val(instr);
                  } else {
                      /* pass always true */
                      nvbit_add_call_arg_const_val32(instr, 1);
                  }

        /* add count warps option */
        nvbit_add_call_arg_const_val32(instr, count_warp_level);
        /* add instruction pc */
        nvbit_add_call_arg_const_val32(instr,offset);
        /* add opcode */
        nvbit_add_call_arg_const_val32(instr,opcode_id);
        //nvbit_add_call_arg_const_val64(instr, uint64_t(&rep_warp));
        //if(!i->isStore())
                 // {
        for (int j=0;j<instr->getNumOperands();j++)
         {const Instr::operand_t * op=instr->getOperand(j);/*get each operand*/
          //if((op->type==Instr::REG))
          nvbit_add_call_arg_const_val32(instr,op->value[0]);/* get register_id*/
          //else
         // {
           // if(j==0)
            // nvbit_add_call_arg_const_val32(instr,10000);
          // }
         }

        }

        /* instrument for memory trace */

        cnt++;
    }
}

__global__ void flush_channel() {
    /* push memory access with negative cta id to communicate the kernel is
     * completed */

    instruction_t ta;
    ta.warp_id=-1;
    channel_dev.push(&ta, sizeof(instruction_t));

       /* flush channel */
    channel_dev.flush();
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
   // std::map<int, std::vector<mem_access_t *>> per_warp_mem_trace;

    while (recv_thread_started) {
        uint32_t num_recv_bytes = 0;
        if (recv_thread_receiving &&
            (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) >
                0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                instruction_t *ta =
                    (instruction_t *)&recv_buffer[num_processed_bytes];

                /* when we get this cta_id_x it means the kernel has completed
                 */
                if (ta->warp_id == -1) {
                    recv_thread_receiving = false;
                    break;
                }
                int warp_id =ta->warp_id;
                //per_warp_mem_trace[warp_id].push_back(ma);
                char fn[100];

                snprintf(fn,sizeof(fn),"./instruction_trace_rep_warp_%d.txt",kernel_id);
                FILE * f = fopen(fn,"a");
                
                if(f!=NULL)
                {
                	fprintf(f,"%d,%d,%d,%d,%d,%d,%d\n",ta->warp_id,ta->offset,ta->opcode_id,ta->des_reg,ta->source_reg_1,ta->source_reg_2,ta->source_reg_3);


                }
                
                fclose(f);
                num_processed_bytes += sizeof(instruction_t);
            }
        }
    }
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

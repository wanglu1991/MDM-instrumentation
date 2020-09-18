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
static ChannelHost channel_host;

/* receiving thread and its control variables */
pthread_t recv_thread;
volatile bool recv_thread_started = false;
volatile bool recv_thread_receiving = false;
__managed__ uint64_t counter = 0;
/* skip flag used to avoid re-entry on the nvbit_callback when issuing
 * flush_channel kernel call */
bool skip_flag = false;
int cache_line_size = 128;
/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 1;
int kernel_id=0;

/* global control variables for this tool */
uint32_t ker_begin_interval = 0;
uint32_t ker_end_interval = UINT32_MAX;
int count_warp_level = 1;
int exclude_pred_off = 0;



/* opcode to id map and reverse map  */
std::map<std::string, int> opcode_to_id_map;
std::map<int, std::string> id_to_opcode_map;

/* information collected in the instrumentation function */
typedef struct {
    int cta_id_x;
    int bb_id;
    int warp_id;

} bb_t;

/* instrumentation function that we want to inject, please note the use of
 * 1. "extern "C" __device__ __noinline__" to prevent code elimination by the
 * compiler.
 * 2. NVBIT_EXPORT_FUNC(count_instrs) to notify nvbit the name of the function
 * we want to inject. This name must match exactly the function name */
extern "C" __device__ __noinline__ void count_instrs(int num_instrs,
                                                     int count_warp_level,int bb_id) {
    /* all the active threads will compute the active mask */
    const int active_mask = __ballot(1);
    /* each thread will get a lane id (get_lane_id is in utils/utils.h) */
    const int laneid = get_laneid();
    /* get the id of the first active thread */
    const int first_laneid = __ffs(active_mask) - 1;
    /* count all the active thread */
    const int num_threads = __popc(active_mask);
    int warp_id = get_global_warp_id();
    /* only the first active thread will perform the atomic */
    if (first_laneid == laneid) {
        if (count_warp_level) {
            atomicAdd((unsigned long long *)&counter, 1 * num_instrs);
            bb_t warp_bb;
            warp_bb.bb_id=bb_id;
            warp_bb.warp_id=warp_id;
            warp_bb.cta_id_x=get_ctaid().x;
            channel_dev.push(&warp_bb, sizeof(bb_t));

        } else {
            atomicAdd((unsigned long long *)&counter, num_threads * num_instrs);
        }
    }
}
NVBIT_EXPORT_FUNC(count_instrs);

extern "C" __device__ __noinline__ void count_pred_off(int predicate,
                                                       int count_warp_level) {
    const int active_mask = __ballot(1);

    const int laneid = get_laneid();

    const int first_laneid = __ffs(active_mask) - 1;

    const int predicate_mask = __ballot(predicate);

    const int mask_off = active_mask ^ predicate_mask;

    const int num_threads_off = __popc(mask_off);
    if (first_laneid == laneid) {
        if (count_warp_level) {
            /* if the predicate mask was off we reduce the count of 1 */
            if (predicate_mask == 0)
                atomicAdd((unsigned long long *)&counter, -1);
        } else {
            atomicAdd((unsigned long long *)&counter, -num_threads_off);
        }
    }
}
NVBIT_EXPORT_FUNC(count_pred_off)

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We
 * typically do initializations in this call. In this case for instance we get
 * some environment variables values which we use as input arguments to the tool
 */
void nvbit_at_init() {
    /* just make sure all managed variables are allocated on GPU */
    setenv("CUDA_MANAGED_FORCE_DEVICE_ALLOC", "1", 1);

    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
    GET_VAR_INT(ker_begin_interval, "KERNEL_BEGIN", 0,
                "Beginning of the kernel launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(
        ker_end_interval, "KERNEL_END", UINT32_MAX,
        "End of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(count_warp_level, "COUNT_WARP_LEVEL", 1,
                "Count warp level or thread level instructions");
    GET_VAR_INT(exclude_pred_off, "EXCLUDE_PRED_OFF", 0,
                "Exclude predicated off instruction from count");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

/* nvbit_at_function_first_load() is executed every time a function is loaded
 * for the first time. Inside this call-back we typically get the vector of SASS
 * instructions composing the loaded CUfunction. We can iterate on this vector
 * and insert call to instrumentation functions before or after each one of
 * them. */
void nvbit_at_function_first_load(CUcontext ctx, CUfunction func) {
    /* Get the static control flow graph of instruction */
    const CFG_t &cfg = nvbit_get_CFG(ctx, func);
    if (cfg.is_degenerate) {
        printf(
            "Warning: Function %s is degenerated, we can't compute basic "
            "blocks statically",
            nvbit_get_func_name(ctx, func));
    }

    if (verbose) {
        printf("Function %s\n", nvbit_get_func_name(ctx, func));
        /* print */
        int cnt = 0;
        for (auto &bb : cfg.bbs) {
            printf("Basic block id %d - num instructions %ld\n", cnt++,
                   bb->instrs.size());
            for (auto &i : bb->instrs) {
                i->print(" ");
            }
        }
    }

    if (verbose) {
        printf("inspecting %s - number basic blocks %ld\n",
               nvbit_get_func_name(ctx, func), cfg.bbs.size());
    }

    /* Iterate on basic block and inject the first instruction */
    int bb_id=0;
    for (auto &bb : cfg.bbs) {
        Instr *i = bb->instrs[0];
        /* inject device function */
        nvbit_insert_call(i, "count_instrs", IPOINT_BEFORE);
        /* add size of basic block in number of instruction */
        nvbit_add_call_arg_const_val32(i, bb->instrs.size());
        /* add count warp level option */
        nvbit_add_call_arg_const_val32(i, count_warp_level);
        nvbit_add_call_arg_const_val32(i,bb_id);
        bb_id++;
        if (verbose) {
            i->print("Inject count_instr before - ");
        }
    }

    if (exclude_pred_off) {
        /* iterate on instructions */
        for (auto i : nvbit_get_instrs(ctx, func)) {
            /* inject only if instruction has predicate */
            if (i->hasPred()) {
                /* inject function */
                nvbit_insert_call(i, "count_pred_off", IPOINT_BEFORE);
                /* add predicate as argument */
                nvbit_add_call_arg_pred_val(i);
                /* add count warp level option */
                nvbit_add_call_arg_const_val32(i, count_warp_level);
                if (verbose) {
                    i->print("Inject count_instr before - ");
                }
            }
        }
    }
}


__global__ void flush_channel() {
    /* push memory access with negative cta id to communicate the kernel is
     * completed */
    bb_t bb_s;
    bb_s.cta_id_x = -1;
    channel_dev.push(&bb_s, sizeof(bb_t));

    /* flush channel */
    channel_dev.flush();
}

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
    while (recv_thread_started) {
        uint32_t num_recv_bytes = 0;
        if (recv_thread_receiving &&
            (num_recv_bytes = channel_host.recv(recv_buffer, CHANNEL_SIZE)) >
                0) {
            uint32_t num_processed_bytes = 0;
            while (num_processed_bytes < num_recv_bytes) {
                bb_t *warp_bb =
                    (bb_t *)&recv_buffer[num_processed_bytes];

                /* when we get this cta_id_x it means the kernel has completed
                 */
                if (warp_bb->cta_id_x == -1) {
                    recv_thread_receiving = false;
                    break;
                }
                int warp_id =warp_bb->warp_id;
                int bb_id =warp_bb->bb_id;
                char fn[100];
                
                snprintf(fn,sizeof(fn),"./bb_trace_%d.txt",kernel_id);
                FILE * f = fopen(fn,"a");
                if(f!=NULL)
                {fprintf(f,"%d,%d\n",warp_id,bb_id);
                }
                fclose(f);
                
                num_processed_bytes += sizeof(bb_t);
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

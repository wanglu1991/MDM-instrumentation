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
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* nvbit utility functions */
#include "utils/utils.h"

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
int verbose = 0;

/* Instrumentation function that we want to inject, please note the use of
 * 1. extern "C" __device__ __noinline__
 *    To prevent "dead"-code elimination by the compiler.
 * 2. NVBIT_EXPORT_FUNC(dev_func)
 *    To notify nvbit the name of the function we want to inject.
 *    This name must match exactly the function name.
 */
extern "C" __device__ __noinline__ void mov_replace(int pred, int reg_dst_num,
                                                    int value_or_reg,
                                                    int is_op1_reg) {
    if (!pred) {
        return;
    }

    if (is_op1_reg) {
        /* read value of register source */
        int value = nvbit_read_reg(value_or_reg);
        /* write value in register destination */
        nvbit_write_reg(reg_dst_num, value);
    } else {
        /* immediate value, just write it in the register */
        nvbit_write_reg(reg_dst_num, value_or_reg);
    }
}
NVBIT_EXPORT_FUNC(mov_replace);

/* nvbit_at_init() is executed as soon as the nvbit tool is loaded. We typically
 * do initializations in this call. In this case for instance we get some
 * environment variables values which we use as input arguments to the tool */
void nvbit_at_init() {
    /* we get some environment variables that are going to be use to selectively
     * instrument (within a interval of kernel indexes and instructions). By
     * default we instrument everything. */
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

/* nvbit_at_function_first_load() is executed every time a function is loaded
 * for the first time. Inside this call-back we typically get the vector of SASS
 * instructions composing the loaded CUfunction. We can iterate on this vector
 * and insert call to instrumentation functions before or after each one of
 * them. */
void nvbit_at_function_first_load(CUcontext ctx, CUfunction func) {
    /* Get the vector of instruction composing the loaded CUFunction "func" */
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, func);

    /* If verbose we print function name and number of" static" instructions */
    if (verbose) {
        printf("inspecting %s - num instrs %ld\n",
               nvbit_get_func_name(ctx, func), instrs.size());
    }

    /* We iterate on the vector of instruction */
    for (auto instr : instrs) {
        /* Check if the instruction falls in the interval where we want to
         * instrument */
        if (instr->getIdx() < instr_begin_interval ||
            instr->getIdx() >= instr_end_interval) {
            continue;
        }

        /* If verbose we print which instruction we are instrumenting  */
        if (verbose) {
            instr->print();
        }

        std::string opcode = instr->getOpcode();
        /* match every MOV instruction */
        if (opcode.compare(0, 3, "MOV") == 0) {
            if (verbose) {
                instr->printDecoded();
            }
            /* assert MOV has really two arguments */
            assert(instr->getNumOperands() == 2);

            /* Insert a call to "mov_replace" before the instruction */
            nvbit_insert_call(instr, "mov_replace", IPOINT_BEFORE);

            /* Add predicate as argument to the instrumentation function */
            nvbit_add_call_arg_pred_val(instr);

            /* Add destination register number as argument (first operand
             * must be a register)*/
            const Instr::operand_t *op0 = instr->getOperand(0);
            assert(op0->type == Instr::REG);
            int reg_dst_num = (int)op0->value[0];
            nvbit_add_call_arg_const_val32(instr, reg_dst_num);

            /* add second operand */
            const Instr::operand_t *op1 = instr->getOperand(1);

            bool is_op1_reg = false;
            if (op1->type == Instr::REG) {
                is_op1_reg = true;
                /* register number as immediate */
                int reg_num = (int)op1->value[0];
                nvbit_add_call_arg_const_val32(instr, reg_num);

            } else if (op1->type == Instr::IMM) {
                /* Add immediate value */
                int value = (int)op1->value[0];
                nvbit_add_call_arg_const_val32(instr, value);

            } else if (op1->type == Instr::CBANK) {
                /* Add value from constant bank (passed as immediate to
                 * the mov_replace function) */
                int cbank_id = (int)op1->value[0];
                int cbank_offset = (int)op1->value[1];
                nvbit_add_call_arg_cbank_val(instr, cbank_id, cbank_offset);

            } else {
                printf("ERROR instrumenting MOV instruction\n");
                instr->printDecoded();
                exit(1);
            }

            /* Add flag to specify if value or register number */
            nvbit_add_call_arg_const_val32(instr, is_op1_reg);

            /* Remove original instruction */
            nvbit_remove_orig(instr);
        }
    }
}

void nvbit_at_cuda_event(CUcontext ctx, int is_exit, nvbit_api_cuda_t cbid,
                         const char *name, void *params, CUresult *pStatus) {
    /* Identify all the possible CUDA launch events */
    if (cbid == API_CUDA_cuLaunch || cbid == API_CUDA_cuLaunchKernel_ptsz ||
        cbid == API_CUDA_cuLaunchGrid || cbid == API_CUDA_cuLaunchGridAsync ||
        cbid == API_CUDA_cuLaunchKernel) {
        if (is_exit) {
            CUDA_SAFECALL(cudaDeviceSynchronize());
        }
    }
}


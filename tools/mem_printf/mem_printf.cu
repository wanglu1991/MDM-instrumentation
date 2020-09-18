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

/* every tool needs to include this once */
#include "nvbit_tool.h"

/* nvbit interface file */
#include "nvbit.h"

/* global control variables for this tool */
uint32_t instr_begin_interval = 0;
uint32_t instr_end_interval = UINT32_MAX;
uint32_t ker_begin_interval = 0;
uint32_t ker_end_interval = UINT32_MAX;
int verbose = 0;

/* opcode to id map */
std::map<std::string, int> opcode_to_id_map;

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
                                                       int32_t imm) {
    if (!pred) {
        return;
    }

    int64_t base_addr = (((uint64_t)reg_high) << 32) | ((uint64_t)reg_low);
    uint64_t addr = base_addr + imm;
    printf(" 0x%016lx - opcode_id %d\n", addr, opcode_id);
}
NVBIT_EXPORT_FUNC(instrument_mem);

void nvbit_at_init() {
    GET_VAR_INT(
        instr_begin_interval, "INSTR_BEGIN", 0,
        "Beginning of the instruction interval where to apply instrumentation");
    GET_VAR_INT(
        instr_end_interval, "INSTR_END", UINT32_MAX,
        "End of the instruction interval where to apply instrumentation");
    GET_VAR_INT(ker_begin_interval, "KERNEL_BEGIN", 0,
                "Beginning of the kernel launch interval where to apply "
                "instrumentation");
    GET_VAR_INT(
        ker_end_interval, "KERNEL_END", UINT32_MAX,
        "End of the kernel launch interval where to apply instrumentation");
    GET_VAR_INT(verbose, "TOOL_VERBOSE", 0, "Enable verbosity inside the tool");
    std::string pad(100, '-');
    printf("%s\n", pad.c_str());
}

/* instrument each memory instruction adding a call to the above
 * instrumentation function */
void nvbit_at_function_first_load(CUcontext ctx, CUfunction f) {
    const std::vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if (verbose) {
        printf("Inspecting function %s at address 0x%lx\n",
               nvbit_get_func_name(ctx, f), nvbit_get_func_addr(f));
    }
    /* iterate on all the static instructions in the function */
    for (auto instr : instrs) {
        if (instr->getIdx() < instr_begin_interval ||
            instr->getIdx() >= instr_end_interval ||
            instr->getMemOpType() == Instr::NONE) {
            continue;
        }
        if (verbose) {
            instr->print();
        }

        if (opcode_to_id_map.find(instr->getOpcode()) ==
            opcode_to_id_map.end()) {
            int opcode_id = opcode_to_id_map.size();
            opcode_to_id_map[instr->getOpcode()] = opcode_id;
            printf("OPCODE %s MAPS TO ID %d\n", instr->getOpcode(), opcode_id);
        }

        int opcode_id = opcode_to_id_map[instr->getOpcode()];
        /* iterate on the operands */
        for (int i = 0; i < instr->getNumOperands(); i++) {
            /* get the operand "i" */
            const Instr::operand_t *op = instr->getOperand(i);

            if (op->type == Instr::MREF) {
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
            }
        }
    }
}

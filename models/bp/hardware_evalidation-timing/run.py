#!/usr/bin/env python 
import os
import sys
def main(kernel_numbers):
    for i in range(1, int(kernel_numbers)+1):
        os.system('python ./generate_L2_access.py '+str(i))
        os.system('python ./interval_warp_model_E_noc_sensitivity.py '+str(i)+' > out_IPC_kernel_'+str(i))
       # os.system('python ./interval_warp_model_MDM_MSHR_noc_sensitivity.py '+str(i)+' > out_IPC_kernel_'+str(i))
        os.system('rm ./read_out_*')
        os.system('rm ./output_*')
        os.system('rm ./pc_*')
        os.system('rm ./L2_trace*')
if __name__== '__main__':
    main(sys.argv[1])

# MDM-instrumentation
This is the directory for the implementation of instrumentation-based MDM model.
We extend the NVbit binary instrumentation tool to collect instruction/memory trace.
The codes for extending NVbit are in tools directory. You may develope your own file to collect other information during the native 
execution of GPU-applications.

We also add an example directory to show how incorporate the instrumentation based tool and MDM model. The instruction/memory 
trace can be collected by runing the executed file ( run-instruction-trace). Run-trace is used to generate the memory instruction trace.
We also have an optimized implementation which only collect traces for warps in different workflow paths to save the storage and timing.

For other applications, you need to collect the trace by yourself.
In particular, the scrips to run the MDM model are in hardware evalidation directory. 


Generate_L2_access.py is used to run the cache simulation and 
collect miss information for L1/L2.
L1_cache/ L2_cache should be configured with your target platforms.
You can also develop your own cache-simulator but make sure the format of the memory traces, cache-simulator and the scripts are consistent.

interval_model_E.py is the script for the MDM model.
It includes the implementations for GPUMech and MDM.
This file is based on the equations in paper.

interval_warp_model_E_noc_sensitivity.py  is a script to run the NoC sensitivity analysis for GPUMEch/MDM-Queue/MDM.
interval_warp_model_MDM_MSHR_noc_sensitivity.py is a script to run the NoC sensitivity analysis for GPUMech+/MDM-MSHR.




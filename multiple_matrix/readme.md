# Tested on RTX 4070 TI and AMD Ryzen 7 7800X3D
## CUDA Toolkit 12.2  
## GPU INFO:  
Device Number: 0  
Device Name: NVIDIA GeForce RTX 4070 Ti  
Compute Capability: 8.9  
Clock Rate (KHz): 2625000  
Total Global Memory (bytes): 4288151552  
Total Constant Memory (bytes): 65536  
Multiprocessor Count: 60  
Shared Memory per Block (bytes): 49152  
Registers per Block: 65536  
Warp Size: 32  
Maximum Threads per Block: 1024  
Maximum Block Dimensions: (1024, 1024, 64)  
Maximum Grid Dimensions: (2147483647, 65535, 65535)  
Total Memory Pitch (bytes): 2147483647  
Texture Alignment: 512  
Concurrent Kernel Execution: Yes  
ECC Enabled: No  
TCC Driver: No  
Maximum threads per multiprocessor: 1536  
Shared memory per multiprocessor: 102400 bytes  
### Times:
**CPU**: 927.14s  
---
**GPU**  
---
*kernel_loop.cu* | wall time: 1.387000s  
*kernel.cu* | wall time: 155.864000s  
*kernel_reduction.cu* | wall time: 6.763000s  
*kernel_shared_mem.cu* | wall time: 7.277000s  

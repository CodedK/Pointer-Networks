import ctypes
import torch
import numpy as np
import datetime


def get_time():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H.%M.%S")


def write_array_txt(ar, fname='array.txt'):
    return np.savetxt(fname, ar, delimiter=',', fmt='%f')


# Save numpy array to file
def save_route_repr(route, prob_name, notes=''):
    """A function to save numpy array to a file"""
    # now = datetime.datetime.now()
    # zz = now.strftime("%Y-%m-%d_%H.%M.%S")
    arr_string = repr(route)
    fname = 'current_logs/sout_%s__%s__%s.out' % (prob_name, get_time(), notes)
    f = open(fname, 'w', encoding="utf-8")
    f.write(arr_string)
    f.close()


# time function execution time
def time_func(func):
    def wrapper(*args, **kwargs):
        start = ctypes.c_int64()
        end = ctypes.c_int64()
        ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(start))
        result = func(*args, **kwargs)
        ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(end))
        return result, (end.value - start.value) * 1e-9
    return wrapper

# print CUDA information
def print_cuda_info():
    print('CUDA is available! Training on GPU...')
    if torch.cuda.is_initialized():
        print('CUDA is initialized!')
    else:
        print('CUDA is not initialized! Initializing...')
        torch.cuda.init()
    # Clear CUDA cache
    print('Clearing CUDA cache...')
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Handling: Could not load Cudart error: CUDART_INVALID_VALUE
    print('Loading cudart64')
    hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\cudart64_110.dll") # Office
    # hllDll = ctypes.WinDLL("C:\\Python399\\Lib\\site-packages\\torch\\lib\\cudart64_110.dll") # 403 functions
    # hllDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\cudart64_110.dll") # 411 functions
    hllDll.cudaSetDevice(0)
    # cudaDeviceProp = ctypes.c_void_p(torch.cuda.current_device())
    # hllDll.cudaGetDeviceProperties(ctypes.byref(cudaDeviceProp), 0)
    # hllDll.cudaSetValidDevices(ctypes.byref(cudaDeviceProp), 0)

    # Handling: Could not load symbol cublasGetSmCountTarget from cublas64_11.dll. Error code 127
    print('Loading cublas64')
    cblDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\cublas64_11.dll") # Office
    # cblDll = ctypes.WinDLL("C:\\Python399\\Lib\\site-packages\\torch\\lib\\cublas64_11.dll") # 488 functions
    # cblDll = ctypes.WinDLL("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.5\\bin\\cublas64_11.dll") # 492 functions
    # C:\Python399\Lib\site-packages\torch\lib\cublas64_11.dll (488 functions. Missing functions: cublasGetSmCountTarget. Error code 127 - Dll version 11.4.2.10064 {PyTorch 2022.08.23})
    # C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.5\bin\cublas64_11.dll (492 functions. Total functions: 492. Dll version is newer 11.7.4.6)
    print('CUDA BLAS PyTorch version loaded!')

    # print("Memory summary: \n", torch.cuda.memory_summary())
    print('-~'*50)
    print('Available GPU devices: ', torch.cuda.device_count())
    print('Current GPU device name: {}, device id: {}'.format(torch.cuda.get_device_name(0), torch.cuda.current_device()))
    print('Current GPU device memory (Memory allocated): ', torch.cuda.memory_allocated(torch.cuda.current_device()))
    print('Current GPU device memory (Max allocated memory): ', torch.cuda.max_memory_allocated(torch.cuda.current_device()))
    print('Current GPU device memory (Memory cashed): ', torch.cuda.memory_reserved(torch.cuda.current_device()))
    print('Current GPU device memory (Max cashed memory): ', torch.cuda.max_memory_reserved(torch.cuda.current_device()))
    # Current GPU multiProcessorCount:  _CudaDeviceProperties(name='NVIDIA GeForce RTX 3090', major=8, minor=6, total_memory=24575MB, multi_processor_count=82)
    print('Current GPU multiProcessorCount: ', type(torch.cuda.get_device_properties(torch.cuda.current_device())))
    print('Device supported capability, {}.{}'.format(torch.cuda.get_device_capability(torch.cuda.current_device())[0], torch.cuda.get_device_capability(torch.cuda.current_device())[1]))
    print('CUDA BLAS cudart verion: ', cblDll.cublasGetCudartVersion(0))
    print('CUDA BLAS math mode: ', cblDll.cublasGetMathMode(0))
    print('CUDA BLAS version: ', cblDll.cublasGetVersion(0))
    print('CUDA BLAS Logger Callback: ', cblDll.cublasGetLoggerCallback(0))
    print('CUDA BLAS Error: ', cblDll.cublasGetError(0))
    print('Setting model to CUDA...')
    print('-~'*50)
    # Check if model.pth file exists and load it
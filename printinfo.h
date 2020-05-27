#ifndef PRINT_INFO_H
#define PRINT_INFO_H

#include "CL/cl.h"

void PrintOpenclInfo();

void SelectOpenCLDevice(cl_platform_id *platform_,
                        cl_device_id *device_);

#endif  // PRINT_INFO_H
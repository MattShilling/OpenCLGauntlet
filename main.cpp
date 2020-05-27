// 1. Program header

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <cstdlib>
#include <unistd.h>
#include <omp.h>
#include <iostream>

#include "CL/cl.h"

#include "printinfo.h"
#include "cl_rig.h"

#ifndef NMB
#define NMB 64
#endif

#define NUM_ELEMENTS NMB * 1024 * 1024

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 64
#endif

#define NUM_WORK_GROUPS NUM_ELEMENTS / LOCAL_SIZE

const float TOL = 0.0001f;

const double kBillion = 1000000000.0;

int LookAtTheBits(float);

int main(int argc, char *argv[]) {

    if (argc < 2) {
        fprintf(stderr, "Build string is a required option!\n");
        return 1;
    }

    size_t nmb = NMB;
    size_t local_size = LOCAL_SIZE;

    if(argc >= 3) {
        nmb = static_cast<size_t>(std::stoi(std::string(argv[2])));
    }

    if(argc >= 4) {
        local_size = static_cast<size_t>(std::stoi(std::string(argv[3])));
    }

    auto c2s = [](char c) {
        return std::string(1,c);
    };

    std::string build_string(argv[1]);
    std::string fcn_str = "kernel void auto_gen(";
    std::string op_str = "";
    std::string prm_str = "";
    char var = 'A';

    bool use_reduction = false;
    if (build_string[1] == ':') {
        // we want a reduction
        if (build_string[2] == '=') {
            use_reduction = true;
        }

    } else if (build_string[1] == '=') {
        // Just a regular =
        use_reduction = false;
        op_str += c2s(var) + "[gid] = ";
        prm_str += "global const float *" + c2s(var) + ",\n";
        var++;

    } else {
        fprintf(stderr, "Build string: Wrong format.");
        return 1;
    }

    std::string mlt = "*";
    std::string add = "+";

    int8_t add_count = 0;
    int8_t mlt_count = 0;
    int8_t variable_count = 1;

    size_t pos = 0;
    std::string token;
    std::string s = build_string;

    auto find_add = [](std::string s) {
        return s.find("+") != std::string::npos;
    };

    auto find_mlt = [](std::string s) {
        return s.find("*") != std::string::npos;
    };

    auto p_line = [](std::string s) {
        return "\t\t\t\t\t global const float *" + s;
    };

    while (find_add(s) || find_mlt(s)) {
        if (s.find("+") < s.find("*")) {
            // Found an add operation.
            pos = s.find(add);
            s.erase(0, pos + add.length());
            op_str += c2s(var) + "[gid] + ";
            prm_str += p_line(c2s(var)) + ",\n";
            var++;
        } else if (s.find("*") < s.find("+")) {
            // Found a multiply operation.
            pos = s.find(mlt);
            s.erase(0, pos + mlt.length());
            op_str += c2s(var) + "[gid] * ";
            prm_str += p_line(c2s(var)) + ",\n";
            var++;
        } 
    }

    op_str += std::string(1, var) + "[gid];";
    prm_str += p_line(c2s(var)) + ") {\n";

    std::cout << fcn_str << prm_str << op_str << std::endl;

    return 0;

    // Allocate the host memory buffers.
    float *hA = new float[NUM_ELEMENTS];
    float *hB = new float[NUM_ELEMENTS];
    float *hC = new float[NUM_ELEMENTS];

    // Fill the host memory buffers.
    for (int i = 0; i < NUM_ELEMENTS; i++) {
        hA[i] = static_cast<float>(sqrt(static_cast<double>(i)));
        hB[i] = hA[i];
    }

    size_t dataSize = NUM_ELEMENTS * sizeof(float);

    OpenCL cl("first.cl");
    cl.Init();

    // Allocate the device memory buffers.
    cl_mem dA, dB, dC;
    cl.CreateBuffer(&dA, dataSize);
    cl.CreateBuffer(&dB, dataSize);
    cl.CreateBuffer(&dC, dataSize);

    // Enqueue the 2 commands to write the data from the
    // host buffers to the device buffers.
    cl.EnqueueWriteBuffer(dA, dataSize, hA);
    cl.EnqueueWriteBuffer(dB, dataSize, hB);

    cl.Wait();

    // Compile and link the kernel code.
    cl.CreateProgram();
    std::string options = "";
    cl.BuildProgram(options);

    // Setup the arguments to the kernel object:
    cl.SetKernelArg(0, sizeof(cl_mem), &dA);
    cl.SetKernelArg(1, sizeof(cl_mem), &dB);
    cl.SetKernelArg(2, sizeof(cl_mem), &dC);

    // Enqueue the kernel object for execution.

    size_t globalWorkSize[3] = {NUM_ELEMENTS, 1, 1};
    size_t localWorkSize[3] = {LOCAL_SIZE, 1, 1};

    cl.Wait();

    cl_int status;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    cl.GetCommandQueue(&cmd_queue);
    cl.GetKernel(&kernel);

    double time0;
    time0 = omp_get_wtime();

    status = clEnqueueNDRangeKernel(cmd_queue,
                                    kernel,
                                    1,
                                    NULL,
                                    globalWorkSize,
                                    localWorkSize,
                                    0,
                                    NULL,
                                    NULL);

    cl.Wait();
    double time1 = omp_get_wtime();
    CHECK_CL(1, "clEnqueueNDRangeKernel failed.\n");

    // 12. read the results buffer back from the device to the
    // host:

    status = clEnqueueReadBuffer(
        cmd_queue, dC, CL_TRUE, 0, dataSize, hC, 0, NULL, NULL);

    CHECK_CL(status, "clEnqueueReadBuffer failed.\n");

    // did it work?

    for (int i = 0; i < NUM_ELEMENTS; i++) {
        float expected = hA[i] * hB[i];
        if (fabs(hC[i] - expected) > TOL) {
            printf("DID NOT WORK!\n");
            // fprintf( stderr, "%4d: %13.6f * %13.6f wrongly
            // produced %13.6f
            // instead
            // of %13.6f (%13.8f)\n",
            // i, hA[i], hB[i], hC[i], expected,
            // fabs(hC[i]-expected) );
            // fprintf( stderr, "%4d:    0x%08x *    0x%08x
            // wrongly produced
            // 0x%08x
            // instead of    0x%08x\n",
            // i, LookAtTheBits(hA[i]), LookAtTheBits(hB[i]),
            // LookAtTheBits(hC[i]),
            // LookAtTheBits(expected) );
        }
    }

    fprintf(stderr,
            "%8d\t%4d\t%10d\t%10.3lf GigaMultsPerSecond\n",
            NMB,
            LOCAL_SIZE,
            NUM_WORK_GROUPS,
            (double)NUM_ELEMENTS / (time1 - time0) / kBillion);

    // Step 13. clean everything up:
    clReleaseMemObject(dA);
    clReleaseMemObject(dB);
    clReleaseMemObject(dC);

    delete[] hA;
    delete[] hB;
    delete[] hC;

    return 0;
}

int LookAtTheBits(float fp) {
    int *ip = (int *)&fp;
    return *ip;
}

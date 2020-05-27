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

class KernelBuilder {
  public:
    KernelBuilder(std::string build_string) {
        std::string op_str = "\t";
        std::string prm_str = "";
        char var = 'A';
        bool use_reduction = false;

        auto c2s = [](char c) { return std::string(1, c); };

        if (build_string[1] == ':') {
            // We want a reduction!
            if (build_string[2] == '=') {
                use_reduction = true;
            }
            op_str += "rdc[t_num] = ";
            prm_str += "local float *rdc,\n";
            good_build_ = true;
        } else if (build_string[1] == '=') {
            // Just a regular assignment, no reduction.
            use_reduction = false;
            op_str += c2s(var) + "[gid] = ";
            prm_str += "global float *" + c2s(var) + ",\n";
            var++;
            good_build_ = true;
        } else {
            fprintf(stderr, "Build string: Wrong format.\n");
            good_build_ = false;
        }

        auto find_add = [](std::string s) {
            return s.find("+") != std::string::npos;
        };

        auto find_mlt = [](std::string s) {
            return s.find("*") != std::string::npos;
        };

        auto p_line = [](std::string s) {
            return "\t\t     global const float *" + s;
        };

        size_t pos = 0;
        std::string &s = build_string;
        while (find_add(s) || find_mlt(s)) {
            if (s.find("+") < s.find("*")) {
                // Found an add operation.
                pos = s.find("+");
                s.erase(0, pos + 1);
                op_str += c2s(var) + "[gid] + ";
                prm_str += p_line(c2s(var)) + ",\n";
                var++;
            } else if (s.find("*") < s.find("+")) {
                // Found a multiply operation.
                pos = s.find("*");
                s.erase(0, pos + 1);
                op_str += c2s(var) + "[gid] * ";
                prm_str += p_line(c2s(var)) + ",\n";
                var++;
            }
        }

        // Add the final parameter string.
        prm_str += p_line(c2s(var)) + ") {\n";

        // Add the last operation variable.
        op_str += c2s(var) + "[gid];\n";

        std::string fcn_str = "kernel void auto_gen(";
        std::string make_gid = "\tint gid = get_global_id(0);\n";
        std::string rdc_mask = "";
        if (use_reduction) {
            make_gid += "\tint n_items = get_local_size(0);\n";
            make_gid += "\tint t_num = get_local_id(0);\n";
            make_gid += "\tint work_group_num = get_group_id(0);\n";
        
            rdc_mask += "for (int ofst=1; ofst<n_items; ofst*=2) {\n"
                        "\tint msk=2*ofst-1;\n"
                        "\tbarrier(CLK_LOCAL_MEM_FENCE);\n"
                        "\tif ((t_num & msk)==0) {\n"
                        "\t\trdc[t_num]+=rdc"
        }
        source_code_ = fcn_str + prm_str + make_gid +
                       op_str + rdc_mask + "}";

        num_vars_ = static_cast<size_t>((var - 'A') + 1);
    }

    bool GoodBuild() {
        if (good_build_) {
            ASSERT_MSG(num_vars_ > 1,
                       "You need more than 1 variable!");
            return true;
        }
        return false;
    }

    std::string GetSourceCode() { return source_code_; }

    size_t GetNumVars() { return num_vars_; }

    std::string GetKernelName() { return "auto_gen"; }

  private:
    std::string source_code_;
    size_t num_vars_;
    bool good_build_;
};

int main(int argc, char *argv[]) {

    // Check for the build string.
    ASSERT_MSG(argc >= 2,
               "Build string is a required argument!");

    // Set up the number of elements and number of work groups
    // based on command line arguments if available, else just
    // used the ones provided in the #defines.
    size_t nmb = NMB;
    size_t local_size = LOCAL_SIZE;
    if (argc >= 3) {
        nmb = std::stoi(std::string(argv[2]));
        ASSERT_MSG(nmb >= 1,
                   "NMB must be greater or equal to one.");
    }
    if (argc >= 4) {
        local_size = std::stoi(std::string(argv[3]));
        ASSERT_MSG(
            local_size >= 8,
            "LOCAL_SIZE must be greater or equal to eight (8).");
    }
    size_t num_elements = nmb * 1024 * 1024;
    size_t num_work_groups = num_elements / local_size;

    // Here we will generate our kernel code from the provided
    // build string.
    std::cout << "Generating kernel code..." << std::endl;
    std::string build_string(argv[1]);
    KernelBuilder kb(build_string);

    // Check to make sure we didn't run into any generation
    // errors.
    ASSERT_MSG(kb.GoodBuild(),
               "Bad auto generation of OpenCL code!");

    // Output the
    std::cout << "Generated:\n" << kb.GetSourceCode() << "\n"
              << std::endl;
    ASSERT_MSG(!true, "TEST");
    // Create dynamic memory based on the number of variables
    // needed, which is indicated to us by our KernelBuilder.
    size_t num_vars = kb.GetNumVars();
    float **host_mem = new float *[num_vars];
    cl_mem *device_mem = new cl_mem[num_vars];
    // Create the second dimension.
    for (auto i = 0; i < num_vars; i++) {
        host_mem[i] = new float[num_elements];
    }

    // Fill the host memory buffers.
    for (auto i = 0; i < num_vars - 1; i++) {
        for (auto j = 0; j < num_elements; j++) {
            host_mem[i + 1][j] =
                static_cast<float>(sqrt(static_cast<double>(i)));
        }
    }

    size_t data_sz = num_elements * sizeof(float);

    OpenCL cl;
    ASSERT(cl.Init());

    // Allocate the device memory buffers.
    ASSERT(cl.CreateWriteBuffer(&(device_mem[0]), data_sz));
    ASSERT(cl.CreateWriteBuffer(&(device_mem[0]), data_sz));
    for (auto i = 0; i < num_vars - 1; i++) {
        ASSERT(
            cl.CreateReadBuffer(&(device_mem[i + 1]), data_sz));
    }

    // Enqueue the 2 commands to write the data from the
    // host buffers to the device buffers.
    for (auto i = 0; i < num_vars - 1; i++) {
        ASSERT(cl.EnqueueWriteBuffer(
            device_mem[i + 1], data_sz, host_mem[i + 1]));
    }

    ASSERT(cl.Wait());

    // Compile and link the kernel code.
    ASSERT(cl.CreateProgram(kb.GetSourceCode()));
    std::string options = "";
    ASSERT(cl.BuildProgram(options, kb.GetKernelName()));

    // Setup the arguments to the kernel object:
    for (auto i = 0; i < num_vars; i++) {
        ASSERT(cl.SetKernelArg(
            i, sizeof(cl_mem), &(device_mem[i])));
    }

    size_t globalWorkSize[3] = {num_elements, 1, 1};
    size_t localWorkSize[3] = {local_size, 1, 1};

    ASSERT(cl.Wait());

    cl_int status;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    cl.GetCommandQueue(&cmd_queue);
    cl.GetKernel(&kernel);

    double time0;
    time0 = omp_get_wtime();

    // Enqueue the kernel object for execution.
    status = clEnqueueNDRangeKernel(cmd_queue,
                                    kernel,
                                    1,
                                    NULL,
                                    globalWorkSize,
                                    localWorkSize,
                                    0,
                                    NULL,
                                    NULL);
    CHECK_CL(status, "clEnqueueNDRangeKernel failed.\n");
    ASSERT(cl.Wait());
    double time1 = omp_get_wtime();

    // Read the results buffer back from the device to the
    // host.
    status = clEnqueueReadBuffer(cmd_queue,
                                 device_mem[0],
                                 CL_TRUE,
                                 0,
                                 data_sz,
                                 host_mem[0],
                                 0,
                                 NULL,
                                 NULL);

    CHECK_CL(status, "clEnqueueReadBuffer failed.\n");

    fprintf(stderr,
            "%8d\t%4d\t%10d\t%10.3lf GigaMultsPerSecond\n",
            nmb,
            local_size,
            num_work_groups,
            (double)num_elements / (time1 - time0) / kBillion);

    // Clean everything up.
    for (auto i = 0; i < num_vars; i++) {
        clReleaseMemObject(device_mem[i]);
        delete[] host_mem[i];
    }

    delete[] host_mem;

    return 0;
}

int LookAtTheBits(float fp) {
    int *ip = (int *)&fp;
    return *ip;
}

/**
 *
 *     // did it work?

    for (int i = 0; i < num_elements; i++) {
        float expected = host_mem[1][i] * host_mem[2][i];
        if (fabs(host_mem[0][i] - expected) > TOL) {
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

    std::cout << "It works!" << std::endl;
    */

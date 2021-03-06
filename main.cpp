#include <cstdio>
#include <cmath>
#include <string>
#include <cstdlib>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <iomanip>

#include "CL/cl.h"

#include "printinfo.h"
#include "cl_rig.h"
#include "auto_gen.h"

#ifndef NMB
#define NMB 64
#endif

#define NUM_ELEMENTS NMB * 1024

#ifndef LOCAL_SIZE
#define LOCAL_SIZE 64
#endif

#define NUM_WORK_GROUPS NUM_ELEMENTS / LOCAL_SIZE

const float TOL = 0.0001f;

const double kBillion = 1000000000.0;

int LookAtTheBits(float);

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

    // Calculate number of elements to calculate and the
    // work groups needed.
    size_t num_elements = nmb * 1024;
    size_t num_work_groups = num_elements / local_size;

    // Here we will generate our kernel code from the provided
    // build string. Thanks, AutoGen!
    std::cout << "Generating kernel code..." << std::endl;
    std::string build_string(argv[1]);
    AutoGen ag(build_string);

    // Check to make sure we didn't run into any generation
    // errors.
    ASSERT_MSG(ag.GoodBuild(),
               "Bad auto generation of OpenCL code!");

    // Output the code because AutoGen makes it nice and pretty
    // for us. Also useful if we want to copy it from the
    // terminal or captured output.
    std::cout << "Generated:\n" << ag.GetSourceCode() << "\n"
              << std::endl;

    std::cout << "Op tag = " << ag.GetOpTag() << std::endl;

    // Create dynamic memory based on the number of variables
    // needed, which is indicated to us by our KernelBuilder.
    size_t num_vars = ag.GetNumVars();
    float **host_mem = new float *[num_vars];
    cl_mem *device_mem = new cl_mem[num_vars];
    // Create the second dimension.
    for (auto i = 0; i < num_vars; i++) {
        if (ag.UseReduction()) {
            // If we are using reduction.
            if (i == 0) {
                // The first variable should always contain
                // `number_work_groups` elements because
                // that is what each work group will be
                // reducing in to.
                host_mem[i] = new float[num_work_groups];
                continue;
            }
        }
        host_mem[i] = new float[num_elements];
    }

    // Fill the host memory buffers.
    // TODO: Now that we have autogenerated kernals, we should
    // find a way to take advantage of this. Pull data from
    // outside source?
    for (auto i = 0; i < num_vars - 1; i++) {
        for (auto j = 0; j < num_elements; j++) {
            host_mem[i + 1][j] =
                static_cast<float>(sqrt(static_cast<double>(j)));
        }
    }

    size_t data_sz = num_elements * sizeof(float);

    ClRig cl;
    ASSERT(cl.Init());

    // Allocate the device memory buffers for writing.
    if (ag.UseReduction()) {
        // As noted before, we need to create a differently sized
        // buffer for the variable we reduce into.
        size_t a_sz = num_work_groups * sizeof(float);
        ASSERT(cl.CreateWriteBuffer(&(device_mem[0]), a_sz));
    } else {
        ASSERT(cl.CreateWriteBuffer(&(device_mem[0]), data_sz));
    }

    // This for loop will skip the first device memory array.
    // The rest of the arrays need to be set to read.
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

    // Wait for our buffers to finish being created.
    ASSERT(cl.Wait());

    // Compile and link the kernel code.
    ASSERT(cl.CreateProgram(ag.GetSourceCode()));
    std::string options = "-cl-mad-enable";
    ASSERT(cl.BuildProgram(options, ag.GetKernelName()));

    // Setup the arguments to the kernel object:
    size_t offset = 0;
    if (ag.UseReduction()) {
        // We need to allocate the local variable in the kernel.
        ASSERT(cl.SetKernelArg(
            0, local_size * sizeof(float), nullptr));
        offset++;
    }
    // This for loop will skip the first kernal argument (0) if
    // we are using reduction. Otherwise, create kernel
    // arguments as usual.
    for (auto i = offset; i < num_vars + offset; i++) {
        ASSERT(cl.SetKernelArg(
            i, sizeof(cl_mem), &(device_mem[i - offset])));
    }

    // Setting work sizes.
    size_t globalWorkSize[3] = {num_elements, 1, 1};
    size_t localWorkSize[3] = {local_size, 1, 1};

    ASSERT(cl.Wait());

    cl_int status;
    cl_command_queue cmd_queue;
    cl_kernel kernel;
    ASSERT(cl.GetCommandQueue(&cmd_queue));
    ASSERT(cl.GetKernel(&kernel));

    double giga_ops = 0.0;

    for (int i = 0; i < 5; i++) {

        // Init and start the timer.
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

        // Check for any failures. Fingers crossed.
        CHECK_CL(status, "clEnqueueNDRangeKernel failed.\n");
        ASSERT(cl.Wait());

        // Stop timer.
        double time1 = omp_get_wtime();

        double giga_ops_new =
            num_elements / (time1 - time0) / kBillion;

        if (giga_ops_new > giga_ops) {
            giga_ops = giga_ops_new;
        }
    }

    // Read the results buffer back from the device to the
    // host.
    if (ag.UseReduction()) {
        // We need to alter the size of the data we read
        // back if we are using reduction methods.
        data_sz = num_work_groups * sizeof(float);
    }
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
    ASSERT(cl.Wait());

    // Check our reduction sum if needed.
    if (ag.UseReduction()) {
        float sum = 0;
        for (int i = 0; i < num_work_groups; i++) {
            sum += host_mem[0][i];
        }
        std::cout << "Reduction Sum = " << sum << std::endl;
    }

    // Print out the statistics for the run.
    fprintf(stderr,
            "%8d\t%4d\t%10d\t%10.3lf Giga(%s)PerSecond\n",
            num_elements,
            local_size,
            num_work_groups,
            giga_ops,
            ag.GetOpTag().c_str());

    std::string records_file =
        "records" + ag.GetOpTag() + ".csv";
    std::cout << "Writing to " << records_file << "..."
              << std::endl;
    std::ofstream outfile;
    outfile.open(records_file, std::ios_base::app);
    // Setting the precision for output.
    outfile << std::fixed;
    outfile << std::setprecision(3);
    outfile << num_elements << ", " << local_size << ", "
            << num_work_groups << ", " << giga_ops << std::endl;
    outfile.close();

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

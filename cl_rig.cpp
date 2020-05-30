#include "cl_rig.h"

#include <cstdio>
#include <string>

#include "printinfo.h"

bool ClRig::Init() {
    // Step 1: Get the platform id and the device id.
    SelectOpenCLDevice(&platform_, &device_);

    // Step 3: Create an OpenCL context.
    context_ = clCreateContext(
        nullptr, 1, &device_, nullptr, nullptr, &status_);

    CHECK_CL(status_, "clCreateContext failed\n");

    // Step 4: Create an OpenCL command queue.
    cmd_queue_ =
        clCreateCommandQueue(context_, device_, 0, &status_);

    CHECK_CL(status_, "clCreateCommandQueue failed\n");

    initialized_ = true;

    return true;
}

bool ClRig::AddProgramFile(const char *program_path) {
    fp_ = fopen(program_path, "r");
    if (fp_ == nullptr) {
        fprintf(stderr,
                "Cannot open OpenCL source file '%s'\n",
                program_path);
        return false;
    }
    return true;
}

bool ClRig::CreateReadBuffer(cl_mem *mem, size_t data_size) {
    ASSERT(context_ != nullptr);
    ASSERT(mem != nullptr);
    // Step 5: Allocate the device memory buffers.
    *mem = clCreateBuffer(
        context_, CL_MEM_READ_ONLY, data_size, NULL, &status_);

    CHECK_CL(status_, "clCreateBuffer failed\n");

    return true;
}

bool ClRig::CreateWriteBuffer(cl_mem *mem, size_t data_size) {
    ASSERT(context_ != nullptr);
    ASSERT(mem != nullptr);
    // Step 5: Allocate the device memory buffers.
    *mem = clCreateBuffer(
        context_, CL_MEM_WRITE_ONLY, data_size, NULL, &status_);

    CHECK_CL(status_, "clCreateBuffer failed\n");

    return true;
}

bool ClRig::EnqueueWriteBuffer(cl_mem d,
                                size_t data_size,
                                const void *ptr) {
    ASSERT(cmd_queue_ != nullptr);
    ASSERT(d != nullptr);
    ASSERT(ptr != nullptr);

    status_ = clEnqueueWriteBuffer(cmd_queue_,
                                   d,
                                   CL_FALSE,
                                   0,
                                   data_size,
                                   ptr,
                                   0,
                                   NULL,
                                   NULL);
    CHECK_CL(status_, "clEnqueueWriteBuffer failed\n");
    return true;
}

// Wait until all queued tasks have taken place:
bool ClRig::Wait() {
    cl_event wait;

    // NOTE: clEnqueueMarker() is depreciated.
    status_ = clEnqueueMarkerWithWaitList(
        cmd_queue_, 0, nullptr, &wait);
    CHECK_CL(status_, "Wait: clEnqueueMarker failed\n");

    status_ = clWaitForEvents(1, &wait);
    CHECK_CL(status_, "Wait: clWaitForEvents failed\n");

    return true;
}

// Step 7: Read the kernel code from a file.
bool ClRig::CreateProgram() {
    ASSERT(context_ != nullptr);

    fseek(fp_, 0, SEEK_END);
    size_t sz = ftell(fp_);
    rewind(fp_);
    std::string cl_program_text(sz, '\0');
    size_t n = fread(&cl_program_text[0], 1, sz, fp_);
    fclose(fp_);
    if (n != sz) {
        fprintf(stderr,
                "Expected to read %d bytes read from '%s' -- "
                "actually read %d.\n",
                sz,
                program_path_,
                n);
        return 1;
    }

    // Create the text for the kernel program.

    const char *strings[1] = {cl_program_text.c_str()};

    program_ = clCreateProgramWithSource(
        context_, 1, (const char **)strings, NULL, &status_);

    CHECK_CL(status_, "clCreateProgramWithSource failed\n");

    return true;
}

// Step 7: Read the kernel code from a file.
bool ClRig::CreateProgram(const std::string &source) {
    ASSERT(context_ != nullptr);

    // Create the text for the kernel program.
    const char *strings[1] = {source.c_str()};

    program_ = clCreateProgramWithSource(
        context_, 1, (const char **)strings, NULL, &status_);

    CHECK_CL(status_, "clCreateProgramWithSource failed\n");

    return true;
}

bool ClRig::BuildProgram(const std::string &options,
                          const std::string &k_name) {
    ASSERT(program_ != nullptr);
    ASSERT(device_ != nullptr);

    status_ = clBuildProgram(
        program_, 1, &device_, options.c_str(), NULL, NULL);

    if (status_ != CL_SUCCESS) {
        size_t size;
        clGetProgramBuildInfo(program_,
                              device_,
                              CL_PROGRAM_BUILD_LOG,
                              0,
                              NULL,
                              &size);
        cl_char *log = new cl_char[size];
        clGetProgramBuildInfo(program_,
                              device_,
                              CL_PROGRAM_BUILD_LOG,
                              size,
                              log,
                              NULL);
        fprintf(stderr, "clBuildProgram failed:\n%s\n", log);
        delete[] log;
        return false;
    }

    // 9. create the kernel object:
    kernel_ = clCreateKernel(program_, k_name.c_str(), &status_);
    CHECK_CL(status_, "clCreateKernel failed\n");

    return true;
}

bool ClRig::SetKernelArg(cl_uint arg_index,
                          size_t arg_size,
                          const void *arg) {
    ASSERT(kernel_ != nullptr);

    status_ = clSetKernelArg(kernel_, arg_index, arg_size, arg);
    CHECK_CL(status_, "clSetKernelArg failed.\n");

    return true;
}

bool ClRig::GetCommandQueue(cl_command_queue *cmd_queue) {
    ASSERT(cmd_queue_ != nullptr);
    ASSERT(cmd_queue != nullptr);

    *cmd_queue = cmd_queue_;

    return true;
}

bool ClRig::GetKernel(cl_kernel *kernel) {
    ASSERT(kernel_ != nullptr);
    ASSERT(kernel != nullptr);

    *kernel = kernel_;

    return true;
}
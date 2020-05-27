#ifndef CL_RIG_
#define CL_RIG_

#include "CL/cl.h"

#include <string>

#define ASSERT(X)         \
    do {                  \
        if ((X) == false) \
            return false; \
    } while (false);

#define CHECK_CL(X, ...)          \
    if (X != CL_SUCCESS) {        \
        fprintf(stderr,           \
                "At %s:%d -> %s", \
                __FILE__,         \
                __LINE__,         \
                __VA_ARGS__);     \
        return false;             \
    }

class OpenCL {
  public:
    OpenCL(const char *program_path)
        : program_path_(program_path), initialized_(false) {}

    ~OpenCL() {
        clReleaseKernel(kernel_);
        clReleaseProgram(program_);
        clReleaseCommandQueue(cmd_queue_);
    }

    bool Init();

    bool CreateBuffer(cl_mem *mem, size_t data_size);

    bool EnqueueWriteBuffer(cl_mem d,
                            size_t data_size,
                            const void *ptr);

    // Wait until all queued tasks have taken place:
    bool Wait();

    // Step 7: Read the kernel code from a file.
    bool CreateProgram();

    bool BuildProgram(const std::string &options);

    bool SetKernelArg(cl_uint arg_index,
                      size_t arg_size,
                      const void *arg);

    bool GetCommandQueue(cl_command_queue *cmd_queue);

    bool GetKernel(cl_kernel *kernel);

  private:
    const char *program_path_;
    cl_platform_id platform_;
    cl_device_id device_;
    FILE *fp_;
    cl_int status_;
    cl_context context_;
    cl_command_queue cmd_queue_;
    bool initialized_;
    cl_program program_;
    cl_kernel kernel_;
};

#endif  // CL_RIG_
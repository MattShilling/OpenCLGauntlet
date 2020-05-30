# OpenCLGauntlet

This project tests out different types of OpenCL kernels to benchmark them. Additionally, the main program
is able to create arbitrary kernels at run time and create host/device memory accordingly to meet a 
specified operation by a user. 

## Building

- `make clean` - Clean up after yourself.
- `make format` - Format all code using .clang-format rules.
- `make` - Build the project!

## Usage

- `./main <expression> [work size in KB] [local size]`

## Kernel Expressions

Currently there are three operations supported for creating dynamic kernels:

- Multiply: `./main a=b*c`
- Add: `./main a=b+c`
- Reduction `./main a:=b*c`

However, you can use any combination of the above operations to create your kernel. Example:

- `./main a:=b*c+d*e+f*g`

TODO: Add subtraction and division.

## Components

There are three main components to this project that assist the main program.

- ClRig: Contains class to manage and hold creating the necessary infrastructure to run a OpenCL kernel.
    - Additionally, it contains the useful `ASSERT()`, `ASSERT_MSG()`, and `CHECK_CL()` macros.
- AutoGen: This class is tasked with autogenerating the kernel code for the expression specified by the user and holds useful data for the main program to properly create the necessary memory for the kernel.
- printinfo.h/cpp: This implementation/header file contains functions used to get a good GPU device on the machine, which the main program will then use.

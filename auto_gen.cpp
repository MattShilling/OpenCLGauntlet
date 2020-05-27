
#include "auto_gen.h"

#include "cl_rig.h"

#include <string>
#include <cstdio>
#include <iostream>

AutoGen::AutoGen(std::string build_string) {
    std::string op_str = "\t";
    std::string prm_str = "";
    char var = 'A';
    use_reduction_ = false;

    /** Lambda function to make a single character string.
     * **/
    auto c2s = [](char c) { return std::string(1, c); };

    if (build_string[1] == ':') {
        // We want a reduction!
        if (build_string[2] == '=') {
            use_reduction_ = true;
            std::cout << "Using Reduction..." << std::endl;
        } else {
            fprintf(stderr, "AutoGen: Did you forget a '='?\n");
            good_build_ = false;
            return;
        }
        // Add the parameter for the local reduction
        // variable.
        prm_str += "local float *rdc, ";
        prm_str += "global float *" + c2s(var) + ",\n";
        // Add the ruduction variable assignment.
        op_str += "rdc[t_num] = ";
        // Move on to the next variable.
        var++;
        good_build_ = true;
    } else if (build_string[1] == '=') {
        // Just a regular assignment, no reduction.
        use_reduction_ = false;
        op_str += c2s(var) + "[gid] = ";
        prm_str += "global float *" + c2s(var) + ",\n";
        // Move on to the next variable.
        var++;
        good_build_ = true;
    } else {
        // There was something wrong with the format.
        // TODO: Make better formatting rules.
        fprintf(stderr, "AutoGen: Build string, bad format.\n");
        good_build_ = false;
    }

    // Lambda functions to see if there's add or
    // multiply delimeters.
    auto find_add = [](std::string s) {
        return s.find("+") != std::string::npos;
    };
    auto find_mlt = [](std::string s) {
        return s.find("*") != std::string::npos;
    };

    // Lambda function to return parameter line.
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
    if (use_reduction_) {
        make_gid += "\tint n_items = get_local_size(0);\n";
        make_gid += "\tint t_num = get_local_id(0);\n";
        make_gid += "\tint work_group_num = get_group_id(0);\n";

        rdc_mask +=
            "\tfor (int offset = 1; "
            "offset < n_items; "
            "offset *= 2) {\n"
            "\t\tint mask = 2 * offset - 1;\n"
            "\t\tbarrier(CLK_LOCAL_MEM_FENCE);\n"
            "\t\tif ((t_num & mask) == 0) {\n"
            "\t\t\trdc[t_num] += rdc[t_num + offset];\n"
            "\t\t}\n"
            "\t}\n"
            "\tbarrier(CLK_LOCAL_MEM_FENCE);\n"
            "\tif(t_num == 0) {\n"
            "\t\tA[work_group_num]=rdc[0];\n"
            "\t}\n";
    }
    source_code_ =
        fcn_str + prm_str + make_gid + op_str + rdc_mask + "}";

    // We can use `var` as a counter since it increases each
    // time
    // we add a new variable.
    num_vars_ = static_cast<size_t>((var - 'A') + 1);
}
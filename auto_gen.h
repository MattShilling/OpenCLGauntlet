#ifndef AUTO_GEN_
#define AUTO_GEN_

#include <string>
#include <cstdlib>
#include "cl_rig.h"

class AutoGen {
  public:
    AutoGen(std::string build_string);

    bool GoodBuild() {
        if (good_build_) {
            ASSERT_MSG(num_vars_ > 1,
                       "You need more than 1 variable!");
            return true;
        }
        return false;
    }

    bool UseReduction() { return use_reduction_; }

    std::string GetSourceCode() { return source_code_; }

    size_t GetNumVars() { return num_vars_; }

    std::string GetKernelName() { return "auto_gen"; }

    std::string GetOpTag() { return tag_; }

  private:
    std::string source_code_;
    size_t num_vars_;
    bool good_build_;
    bool use_reduction_;
    std::string tag_;
};

#endif  // AUTO_GEN_
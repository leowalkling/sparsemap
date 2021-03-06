#ifndef FACTOR_TREE_FAST
#define FACTOR_TREE_FAST

#include <iostream>

#include <ad3/examples/cpp/parsing/FactorTree.h>

/*
 * A simplified Tree factor
 * which takes a length x length-1 matrix and simply ignores the diagonal.
 *
 * This way we are able to pass it the output of something like dot(A, B).
 * */

namespace sparsemap {

class FactorTreeFast : public AD3::FactorTree {

 public:
  void Initialize(int length) {
    length_ = length;
    index_arcs_.assign(length, std::vector<int>(length, -1));
    int k = 0;
    for (int m = 1; m < length; m++) {
        for (int h = 0; h < length; ++h) {
            if (h != m) {
                index_arcs_[h][m] = k;
            }
            ++k;
        }
    }
  }

};

} // namespace sparsemap

#endif

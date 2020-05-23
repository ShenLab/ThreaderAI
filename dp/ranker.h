#ifndef _RANKER_H
#define _RANKER_H
#include "types.h"
#include <string>
#include <vector>

TypeVal calc_mac(TypeVal *score, TypePos template_len, TypePos query_len,
                 std::vector<AlignedPair> &aligned_pairs);

class Alignment {
public:
  Alignment(const std::string &name, TypePos template_len,
            TypeVal &aligned_score, std::vector<AlignedPair> &aligned_pairs) {
    template_name_ = name;
    template_len_ = template_len;
    aligned_score_ = aligned_score;
    aligned_pairs_ = aligned_pairs;
  }

public:
  std::string template_name_;
  TypePos template_len_;
  TypeVal aligned_score_;
  std::vector<AlignedPair> aligned_pairs_;
};

#endif

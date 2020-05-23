#ifndef _TYPES_H
#define _TYPES_H

typedef float TypeVal;
typedef unsigned int TypePos;

enum AlignState { MATCH, DELETE, INSERT, ZERO };

const TypeVal INFMIN = -100000.0;

struct AlignedPair {
  TypePos template_pos;
  TypePos query_pos;

  TypeVal score;
};

#endif

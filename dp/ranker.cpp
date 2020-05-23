#include <gflags/gflags.h>
#include <glog/logging.h>

#include "ranker.h"

DEFINE_double(alpha, 0.3, "alpha");
DEFINE_double(beta, 0.5, "beta");

TypeVal calc_mac(TypeVal *score, TypePos template_len, TypePos query_len,
                 std::vector<AlignedPair> &aligned_pairs) {

  TypeVal dp_table[1000][1000];
  AlignState trace_table[1000][1000];

  TypeVal alpha = FLAGS_alpha, beta = FLAGS_beta;
  TypeVal *row_score = score;
  TypeVal best_score = INFMIN;
  TypePos best_template_pos, best_query_pos;
  for (TypePos template_pos = 0; template_pos < template_len; template_pos++) {
    for (TypePos query_pos = 0; query_pos < query_len; query_pos++) {
      TypeVal max_score = row_score[query_pos] - alpha;
      AlignState trace_state = ZERO;
      if (template_pos > 0 && query_pos > 0 &&
          dp_table[template_pos - 1][query_pos - 1] > 0) {
        max_score += dp_table[template_pos - 1][query_pos - 1];
        trace_state = MATCH;
      }
      if (template_pos > 0) {
        TypeVal del = dp_table[template_pos - 1][query_pos] - alpha * beta;
        if (del > max_score) {
          max_score = del;
          trace_state = DELETE;
        }
      }
      if (query_pos > 0) {
        TypeVal insert = dp_table[template_pos][query_pos - 1] - alpha * beta;
        if (insert > max_score) {
          max_score = insert;
          trace_state = INSERT;
        }
      }

      trace_table[template_pos][query_pos] = trace_state;
      dp_table[template_pos][query_pos] = max_score;
      if (max_score > best_score) {
        best_score = max_score;
        best_template_pos = template_pos;
        best_query_pos = query_pos;
      }
    }
    row_score += query_len;
  }

  if (best_score <= 0.0) {
    return 0.0;
  }

  TypePos template_pos = best_template_pos, query_pos = best_query_pos;
  while (template_pos >= 0 && query_pos >= 0) {
    if (trace_table[template_pos][query_pos] == ZERO) {
      aligned_pairs.push_back({template_pos, query_pos,
                               score[template_pos * query_len + query_pos]});
      break;
    }

    if (trace_table[template_pos][query_pos] == MATCH) {
      aligned_pairs.push_back({template_pos, query_pos,
                               score[template_pos * query_len + query_pos]});
      template_pos--;
      query_pos--;
    } else if (trace_table[template_pos][query_pos] == DELETE) {
      template_pos--;
    } else {
      query_pos--;
    }
  }

  return best_score;
}

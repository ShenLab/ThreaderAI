#include <gflags/gflags.h>
#include <glog/logging.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <queue>

#include "cnpy.h"
#include "ranker.h"
#include "types.h"

static bool ValidateNotEmpty(const char *flagname, const std::string &value) {
  if (value != "") {
    return true;
  }
  printf("--%s can not be empty\n", flagname);
  return false;
}

DEFINE_string(template_list, "", "Input file of template list");
DEFINE_validator(template_list, &ValidateNotEmpty);

DEFINE_string(score_dir, "", "Directory of scores");
DEFINE_validator(score_dir, &ValidateNotEmpty);

DEFINE_string(output, "", "Output file");
DEFINE_validator(output, &ValidateNotEmpty);

int read_template_list(const char *list_path,
                       std::vector<std::string> &template_list);

struct compareAlignment {
  bool operator()(const Alignment &left, const Alignment &right) {
    return left.aligned_score_ > right.aligned_score_;
  }
};

TypeVal calc_alignment(const std::string &template_name, TypePos &template_len,
                       std::vector<AlignedPair> &aligned_pairs) {
  std::string score_path = FLAGS_score_dir + "/" + template_name + ".npy";
  cnpy::NpyArray arr = cnpy::npy_load(score_path.c_str());
  TypeVal *score = arr.data<TypeVal>();
  template_len = int(arr.shape[0]);
  TypePos t = template_len;
  TypePos query_len = int(arr.shape[1]);
  // std::cout << template_name << " " << template_len << " " << query_len
  //           << std::endl;
  TypeVal aligned_score =
      calc_mac(score, template_len, query_len, aligned_pairs);
  return aligned_score;
}

int main(int argc, char **argv) {
  gflags::SetVersionString("1.0.0.0");
  // gflags::SetUsageMessage("Usage : ./app");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  google::InitGoogleLogging(argv[0]);

  std::vector<std::string> template_list;

  read_template_list(FLAGS_template_list.c_str(), template_list);
  LOG(INFO) << "The size of template lib is " << template_list.size()
            << std::endl;
  std::cout << template_list.size() << std::endl;
  int count = 0;

  int TOP = 200;
  std::priority_queue<Alignment, std::vector<Alignment>, compareAlignment> Q;
  for (auto &template_name : template_list) {
    std::string score_path = FLAGS_score_dir + "/" + template_name + ".npy";
    std::vector<AlignedPair> aligned_pairs;
    TypePos template_len;
    TypeVal aligned_score =
        calc_alignment(template_name, template_len, aligned_pairs);
    if (Q.size() < TOP) {
      Q.push({template_name, template_len, aligned_score, aligned_pairs});
    } else {
      if (aligned_score > Q.top().aligned_score_) {
        Q.pop();
        Q.push({template_name, template_len, aligned_score, aligned_pairs});
      }
    }
    count++;
    if (count % 1000 == 0) {
      std::cout << count << std::endl;
    }
  }

  std::vector<Alignment> res;
  while (!Q.empty()) {
    // std::cout << Q.top().template_name_ << " " << Q.top().aligned_score_ <<
    // std::endl;
    res.push_back(Q.top());
    Q.pop();
  }
  std::ofstream fout(FLAGS_output);
  for (TypePos i = 0; i < res.size(); i++) {
    auto &cur = res[res.size() - i - 1];
    fout << i + 1 << "\t" << cur.template_name_ << "\t" << cur.template_len_
         << "\t" << cur.aligned_score_ << "\t";
    for (auto &p : cur.aligned_pairs_) {
      fout << p.template_pos << "," << p.query_pos << "," << p.score << " ";
    }
    fout << std::endl;
  }
  return 0;
}

int read_template_list(const char *list_path,
                       std::vector<std::string> &template_list) {

  std::ifstream fin;
  fin.open(list_path);
  std::string line;
  while (getline(fin, line)) {
    template_list.emplace_back(line);
  }
  return 0;
}

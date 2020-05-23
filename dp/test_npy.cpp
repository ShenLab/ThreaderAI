#include "cnpy.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

typedef float Type;

int test() {
  std::string score_dir = "/data/hz2529/zion/casp/CASP13/results_theraderai/"
                          "2020-01-26-15-18-47-epoch14/T0961-D1/score";
  std::ifstream fin;
  fin.open("/data/hz2529/zion/pdbdata/list");
  std::string line;
  std::vector<std::string> template_list;
  while (getline(fin, line)) {
    template_list.emplace_back(line);
  }
  int cnt = 0;
  for (auto &template_name : template_list) {
    std::string score_path = score_dir + "/" + template_name + ".npy";
    cnpy::NpyArray arr = cnpy::npy_load(score_path.c_str());
    cnt += 1;
    if (cnt % 1000 == 0) {
      std::cout << "loaded " << cnt << std::endl;
      std::cout << arr.shape[0] << "\t" << arr.shape[1] << std::endl;
      std::cout << arr.word_size << " " << sizeof(double) << " "
                << sizeof(float) << std::endl;
      Type *data = arr.data<Type>();
      double sum = 0.0;
      for (int i = 0; i < arr.shape[0]; i++) {
        Type *row = data + i * arr.shape[1];
        for (int j = 0; j < arr.shape[1]; j++) {
          sum += row[j];
        }
      }
      std::cout << template_name << " " << sum << std::endl;
    }
    // std::<<double> > *loaded = rr.data<std::com<double>>();
  }
  std::cout << "template size " << template_list.size() << std::endl;
  return 0;
}

int main() {
  test();
  return 0;
}

#ifndef RANDOMIZER_H
#define RANDOMIZER_H

#include <vector>
#include "datum.h"

class Randomizer {
 public:
  Randomizer();
  void SetDataSize(int s) {_data_size = s;}
  void SetUpBatchIndices(int batch_size);
  int GetStartIndex(int);
  int GetEndIndex(int);
  ~Randomizer();
 private:
  int _data_size;
  int _group_num;
  vector<int> _start_indices;
  vector<int> _end_indices;
};

#endif

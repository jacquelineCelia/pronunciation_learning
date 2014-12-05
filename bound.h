#ifndef BOUND_H
#define BOUND_H

#include <vector>

#include "config.h"

using namespace std;

class Bound {
 public:
  Bound(Config*);
  int dim() {return _dim;}
  int frame_num() {return _frame_num;}
  int label() {return _label;}
  int start_frame() {return _start_frame;}
  int end_frame() {return _end_frame;}
  bool is_labeled() {return _is_labeled;}
  bool is_boundary() {return _is_boundary;}
  void set_is_labeled(bool value) {_is_labeled = value;}
  void set_is_boundary(bool value) {_is_boundary = value;}
  void set_label(int label) {_label = label;}
  void set_start_frame(int start_frame) {_start_frame = start_frame;}
  void set_end_frame(int end_frame) {_end_frame = end_frame;}
  void set_data(float** data, int frame_num); 
  void print_data();
  vector<float*> data() {return _data_array;}
  ~Bound();
 private:
  float** _data;
  int _frame_num;
  int _dim;
  int _label;
  int _start_frame;
  int _end_frame;
  bool _is_labeled;
  bool _is_boundary;
  Config* _config;
  vector<float*> _data_array;
};

#endif

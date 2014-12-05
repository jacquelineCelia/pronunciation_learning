#ifndef SEGMENT_H
#define SEGMENT_H

#include <vector>
#include "bound.h"
#include "config.h"

using namespace std;

class Segment {
 public:
  Segment() {};
  Segment(Config* config, int id) {_config = config; _id = id;}
  int id() {return _id;}
  int frame_num() {return _data.size();}
  vector<int> state_seq() {return _state_seq;}
  vector<int> mix_seq() {return _mix_seq;}
  vector<float*> data() {return _data;}
  void set_state_seq(vector<int>& state_seq) {_state_seq = state_seq;} 
  void set_mix_seq(vector<int>& mix_seq) {_mix_seq = mix_seq;}
  void set_data(vector<float*> data) {_data = data;}
  void set_data_index(vector<int> data_index) {_data_index = data_index;}
  void set_id(int id) {_id = id;}
  float* frame(int i) {return _data[i];}
  int frame_index(int i) {return _data_index[i];}
  void push_back(Bound*);
  ~Segment() {};
 private:
  int _id;
  int _start_frame;
  int _end_frame;
  vector<int> _state_seq;
  vector<int> _mix_seq;
  vector<int> _data_index;
  vector<float*> _data;
  Config* _config;
};

#endif

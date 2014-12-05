#ifndef COUNTER_H
#define COUNTER_H

#include "config.h"
#include "cluster.h"
#include "l2s.h"
#include <vector>
#include <map>

using namespace std;

class Counter {
 public:
  Counter() {};
  Counter(Config*);
  vector<Cluster*>& clusters() {return _cluster_counter;}
  map<vector<int>, L2S*>& l2s() {return _l2s_counter;}
  Counter& operator+= (Counter&);
  int weak_limit() {return _weak_limit;}
  void SetSpecialL2S(vector<int> label);
  void ShowL2S();
  void CreateCountsForNonLeaves();
  void Initialize(Config*);
  void Reset();
  ~Counter();
 private:
  vector<Cluster*> _cluster_counter;
  map<vector<int>, L2S*> _l2s_counter;
  Config* _config; 
  int _weak_limit;
  int _num_chars; 
};

#endif

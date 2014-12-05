#ifndef CLUSTER_H
#define CLUSTER_H

#include <vector>
#include <iostream>
#include <fstream>

#include "bound.h"
#include "segment.h"
#include "prob_list.h"
#include "gmm.h"
#include "config.h"
#include "toolkit.h"

using namespace std;

class Cluster {
 public:
  Cluster() {};
  Cluster(Config*);
  Cluster(Config*, int);
  Cluster(Config*, int, int, int);
  Cluster(Cluster&);
  Cluster& operator+= (Cluster&);
  // Access functions
  void set_transition_probs(vector<vector<float> >&);
  void set_emission(GMM&, int);
  void set_emissions(vector<GMM>&);
  void set_id(int id) {_id = id;}
  void set_is_fixed(bool is_fixed) {_is_fixed = is_fixed;}
  int id() {return _id;}
  int state_num() {return _state_num;}
  float transition_prob(int i, int j) {return _transition_probs[i][j];}
  bool is_fixed() const {return _is_fixed;}
  vector<vector<float> > transition_probs() {return _transition_probs;}
  GMM& emission(int index) {return _emissions[index];}
  Config* config() {return _config;}
  // Tool
  float** ConstructSegProbTable(vector<Bound*>&);
  float ComputeSegmentProb(Segment*);
  ProbList<int>** MessageBackwardForASegment(Segment*);
  void Plus(Segment*);
  void Minus(Segment*);
  void Save(ofstream&);
  void Load(ifstream&);
  void PreCompute(float**, int);
  void Reset();
  ~Cluster();
 protected:
  bool _is_fixed;
  int _id;
  int _state_num;
  // Save in Log
  vector<vector<float> > _transition_probs;
  vector<GMM> _emissions;
  Config* _config;
};

#endif

#ifndef L2S_H
#define L2S_H

#include <iostream>
#include <fstream>
#include <vector>
#include <map>

#include "config.h"
#include "bound.h"
#include "prob_list.h"
#include "toolkit.h"

using namespace std;

class L2S {
 public:
  L2S() {};
  L2S(Config*, const vector<int>&);
  L2S(L2S&);
  L2S& operator= (L2S&);
  void set_mapping_likelihood(vector<vector<vector<float> > >&);
  void set_length_prior(vector<float>&);
  void set_label(vector<int>&);
  void set_parent(L2S*);
  void set_length_alpha(vector<float>);
  void set_mapping_alpha(vector<float>&);
  void set_parent_label();
  void set_updated(bool updated) {_updated = updated;}
  void set_length_prior_fixed(bool value) {_is_length_prior_fixed = value;}
  void set_mapping_likelihood_fixed(bool value) {_is_mapping_likelihood_fixed = value;}
  void set_collapsed(bool v) {_is_collapsed = v;}
  void set_is_leaf(bool v) {_is_leaf = v;}
  void print_length_prior();
  void print_mapping_likelihood();
  L2S* parent() {return _parent;}
  Config* config() {return _config;}
  vector<int> label() {return _label;}
  vector<int> parent_label() {return _parent_label;}
  vector<float> length_prior() const {return _length_prior;}
  vector<float> length_prior() {return _length_prior;}
  vector<vector<vector<float> > > mapping_likelihood() const {return _mapping_likelihood;}
  vector<float> mapping_likelihood(int i, int j) {return _mapping_likelihood[i][j];}
  vector<float>& length_alpha() {return _length_alpha;}
  vector<float>& mapping_alpha() {return _mapping_alpha;}
  float Prior(int);
  float Likelihood(vector<int>);
  // Retrive likelihood by specifying
  // 1) length of the phone sequence
  // 2) position in the phone sequence that is going to be queried
  // 3) the phone id
  float Likelihood(int, int, int);
  int N() const {return _N;}
  int center_label() {return _label[(_label.size() - 1) / 2];}
  ProbList<int>**** ConstructIndividualSegProbTable( \
          vector<Bound*>&, float***);
  ProbList<int>** ConstructSegProbTable(vector<Bound*>&, \
                                        ProbList<int>****); 
  bool updated() {return _updated;}
  bool is_mapping_likelihood_fixed() {return _is_mapping_likelihood_fixed;}
  bool is_length_prior_fixed() {return _is_length_prior_fixed;}
  bool is_collapsed() {return _is_collapsed;}
  bool is_leaf() {return _is_leaf;}
  L2S& operator+= (const L2S& rhs);
  L2S& operator-= (const L2S& lhs);
  // For collapsed inference
  float GetCollapsedLikelihood(int, int, int);
  float mapping_likelihood_ijk(int i, int j, int k) {return _mapping_likelihood[i][j][k];}
  void Plus(vector<int>, int count = 1);
  void Minus(vector<int>, int count = 1);
  void set_N(int N) {_N = N;}
  void Reset();
  void Save(ofstream&);
  void Load(ifstream&);
  void print_label();
  ~L2S() {};

 protected:
  Config* _config;
  vector<int> _label;
  vector<int> _parent_label;
  vector<float> _length_alpha;
  float _total_length_alpha;
  vector<float> _mapping_alpha;
  vector<float> _length_prior;
  vector<vector<vector<float> > > _mapping_likelihood;
  int _N;
  bool _updated;
  bool _is_length_prior_fixed;
  bool _is_mapping_likelihood_fixed;
  bool _is_collapsed;
  bool _is_leaf;
  L2S* _parent;
};

#endif

#ifndef CONFIG_H
#define CONFIG_H

#include "gaussian_seed.h"

#include <vector>
#include <string>

using namespace std;

class Config {
 public:
  Config();
  ~Config();
  bool Load(string& fn, string& fn_gaussian);
  bool LoadGaussian(string& fn_gaussian);
  bool parallel() {return _parallel;}
  bool precompute() {return _precompute;} 
  int emission_type() {return _emission_type;}
  int state_num() {return _state_num;}
  int n_context() {return _n_context;}
  int max_num_epsilons() {return _max_num_epsilons;} 
  int max_duration() {return _max_duration;}
  int max_num_units() {return _max_num_units;}
  int cluster_num() {return _cluster_num;}
  int n_ngram() {return _n_ngram;}
  int mix_num() {return _mix_num;}
  int dim() {return _dim;}
  int weak_limit() {return _weak_limit;}
  int num_chars() {return _num_chars;}
  int num_sil_mix() {return _num_sil_mix;}
  int num_sil_states() {return _num_sil_states;}
  int num_gaussian_seed() {return _num_mixtures;}
  float sil_self_trans_prob() {return _sil_self_trans_prob;}
  float transition_alpha() {return _transition_alpha;}
  float mix_alpha() {return _mix_alpha;}
  vector<float> length_alpha(int, vector<int>);
  float gaussian_a0() {return _gaussian_a0;}
  float gaussian_k0() {return _gaussian_k0;}
  float ngram_weight() {return _ngram_weight;}
  bool is_collapsed() {return _is_collapsed;}
  bool LoadSeedingMixtures(string&);
  bool UseSilence() {return _use_silence;}
  vector<float> gaussian_b0() {return _gaussian_b0;}
  vector<float> gaussian_u0() {return _gaussian_u0;}
  vector<float> mapping_alpha(int i) {return _mapping_alpha[i];}
  vector<float> boundary_distribution() {return _boundary_distribution;}
  void print();
  void set_use_silence(bool t) {_use_silence = t;}
  GaussianSeed mixture(int i) {return _mixtures[i];}
 private:
  int _emission_type;
  int _state_num;
  int _n_context;
  int _max_num_epsilons;
  int _max_duration;
  int _max_num_units;
  int _cluster_num;
  int _n_ngram;
  int _mix_num;
  int _dim;
  int _weak_limit;
  int _num_chars;
  int _num_sil_mix;
  int _num_sil_states;
  int _num_mixtures;
  float _sil_self_trans_prob;
  float _transition_alpha;
  float _mix_alpha;
  float _gaussian_a0;
  float _gaussian_k0;
  float _ngram_weight;
  bool _is_collapsed;
  bool _parallel;
  bool _precompute;
  bool _use_silence;
  vector<vector<float> > _mapping_alpha;
  vector<float> _boundary_distribution;
  vector<float> _gaussian_b0;
  vector<float> _gaussian_u0;
  vector<vector<float> > _normal_length_alpha;
  vector<float> _sil_length_alpha;
  vector<float> _space_length_alpha;
  vector<GaussianSeed> _mixtures;
};

#endif

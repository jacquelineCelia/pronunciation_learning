#ifndef MODEL_H
#define MODEL_H

#include "config.h"
#include "cluster.h"
#include "counter.h"
#include "l2s.h"
#include <vector>
#include <map>

using namespace std;

class Model {
 public:
  Model() {};
  Model(Config*);
  Model(Model& rhs);
  Model& operator= (Model& rhs);
  vector<Cluster*>& clusters() {return _clusters;}
  map<vector<int>, L2S*>& l2s() {return _l2s;}
  Config* config() const {return _config;}
  int weak_limit() const {return _weak_limit;}
  int num_chars() const {return _num_chars;}
  void AddSilenceCluster(Cluster* cluster); 
  void Initialize();
  void SetSpecialL2S(vector<int>);
  void SetSpecialL2SMappingLikelihoodFixed(vector<int>);
  void SetParentOfSpecialL2S();
  void Save(const string&);
  void LoadSnapshot(const string&);
  void PreCompute(float**, int);
  void CopyL2SCounts(Counter&);
  void ShowL2S();
  ~Model();
 private:
  vector<Cluster*> _clusters;
  map<vector<int>, L2S*> _l2s;
  Config* _config;
  int _weak_limit;
  int _num_chars;
};

#endif

#ifndef SAMPLER_H
#define SAMPLER_H

#include <map>
#include <string>
#include <mkl.h>

#include "config.h"
#include "datum.h"
#include "rvg.h"
#include "toolkit.h"
#include "cluster.h"
#include "prob_list.h"
#include "model.h"
#include "counter.h"
#include "l2s.h"
#include "rvg.h"

using namespace std;
using namespace boost;
using namespace boost::random;

class Sampler {
 public:
  Sampler() {};
  Sampler(Config*);
  // Model-wise
  void SampleModelParams(Model*, Counter* = NULL);
  void SampleClusterParams(Cluster*, Cluster* = NULL);
  void SampleL2SParams(L2S*, L2S* = NULL);
  // Data-wise
  void SampleSegments(Datum*, \
          vector<Cluster*>&, map<vector<int>, L2S*>&, \
          vector<Cluster*>&, map<vector<int>, L2S*>&);
  void SampleStateMixtureSeq(Segment*, Cluster*);
  // Data-assist 
  // Note that current implementation still considers words without mapped to any sounds
  void MessageBackward(Datum*, map<vector<int>, \
          ProbList<int>**>&, \
          map<vector<int>, L2S*>&, ProbList<int>**, ProbList<int>**);
  void SampleForward(Datum*, \
          map<vector<int>, ProbList<int>**>&, \
          map<vector<int>, ProbList<int>****>&, \
          ProbList<int>**, ProbList<int>**);
  void ComputeSegProbGivenCluster(Datum*, vector<Cluster*>&, float***);
  void ComputeSegProbGivenLetter(Datum*, map<vector<int>, L2S*>&, \
          float***, map<vector<int>, \
          ProbList<int>**>&, map<vector<int>, ProbList<int>****>&);
  // Tool-wise
  void RemoveL2SAlignment(Datum*, map<vector<int>, L2S*>&, map<vector<int>, L2S*>&);
  void RemoveClusterAssignment(Datum*, vector<Cluster*>&);
  void AddL2SAlignment(Datum*, map<vector<int>, L2S*>&, map<vector<int>, L2S*>&);
  void AddClusterAssignment(Datum*, vector<Cluster*>&); 
  int SampleIndexFromLogDistribution(vector<float>);
  int SampleIndexFromDistribution(vector<float>);
  // vector<float> SampleGamma(int, float);
  vector<float> SampleDirFromGamma(int, float*, float = -70000000);
  vector<float> SampleGaussianMean(vector<float>, \
                                   vector<float>, float, int, bool);
  vector<float> SampleGaussianPre(vector<float>, vector<float>, \
          float, int, bool);
  float UpdateGammaRate(float, float, float, float, float);
  vector<int> FindRootInL2S(vector<int>, map<vector<int>, L2S*>&);
  void EstablishRootInL2S(vector<int>, \
                                 map<vector<int>, L2S*>&);
  void EstablishLeafInL2S(vector<int>, map<vector<int>, L2S*>&);
  void CopyL2SCounts(L2S*, L2S*);
  void Initialize(Config*);
  ~Sampler();
 private:
  RandomVarGen rvg;
  int _n_context;
  int _max_num_epsilons;
  Config* _config;
  VSLStreamStatePtr stream; 
};

#endif

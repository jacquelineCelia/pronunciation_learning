#ifndef MANAGER_H
#define MANAGER_H

#include <string>
#include <iostream>
#include <fstream>
#include "config.h"
#include "datum.h"
#include "model.h"
#include "randomizer.h"
#include "bound.h"

class Manager {
 public:
  Manager(Config*);
  bool LoadData(string& fn_data_list);
  void LoadDataIntoMatrix(vector<Datum*>&);
  void LoadBounds(Datum*, ifstream&, ifstream&);
  void LoadTranscriptions(Datum*, ifstream&);
  bool LoadSilenceModel(string&);
  void Inference(int, int, const string&);
  void ParallelInference(int, int, const string&);
  void SerielInference(int, int, const string&);
  void InitializeModel();
  void InitializeModel(const string&);
  void SaveData(const string&);
  void SaveModel(const string&);
  void ShowModel();
  string GetTag(const string&);
  ~Manager();
 private:
  vector<Datum*> _datum;
  Model* _model;
  Config* _config;
  int _total_frame_num;
  vector<float*> _features;
  Randomizer randomizer;
};

#endif

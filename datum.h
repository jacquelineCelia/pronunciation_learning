// File Name: datum.h
// Date: Nov. 19, 2012

#ifndef DATUM_H
#define DATUM_H

#include <vector>

#include "config.h"
#include "letter.h"
#include "segment.h"
#include "bound.h"

using namespace std;

// A datum represents one training data point, which 
// includes the hidden letter sequence and the observed
// speech data. 
// A datum only keeps the relationship of words
// and letters, segments and bounds.
// It does not do any actions.
// This should enable parallel computation.

class Datum {
 public:
  Datum(Config*);
  // Access Functions:
  vector<Letter*>& letters() {return _letters;}
  Letter* letter(int i) {return _letters[i];}
  vector<Segment*>& segments() {return _segments;}
  vector<Bound*>& bounds() {return _bounds;}
  string tag() {return _tag;}
  bool transcribed() {return _transcribed;}
  bool corrupted() {return _corrupted;}
  void set_bounds(vector<Bound*>& bounds) {_bounds = bounds;}
  void set_segments(vector<Segment*>& segments) {_segments = segments;}
  void set_transcribed(bool transcribed) {_transcribed = transcribed;}
  void set_letters(vector<Letter*>& letters) {_letters = letters;}
  void set_tag(const string& tag) {_tag = tag;}
  void set_corrupted(bool value) {_corrupted = value;}
  void set_corrupted_type(int type) {_corrupted_type.push_back(type);}

  void ClearSegs(); 
  void Save(const string&);
  void CheckLetterBoundNumber();
  vector<int> letter_seq();
  vector<int> Context(int i);
  ~Datum();
 private:
  vector<Letter*> _letters;
  vector<Segment*> _segments;
  vector<Bound*> _bounds;
  string _tag;
  bool _transcribed;
  bool _corrupted;
  vector<int> _corrupted_type;
  Config* _config;
};

#endif

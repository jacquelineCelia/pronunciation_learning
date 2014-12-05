#ifndef LETTER_H
#define LETTER_H

#include <vector>

using namespace std;

class Letter {
 public:
  Letter() {};
  Letter(int);
  Letter(int, vector<int>);
  Letter(const Letter&);
  void set_word_boundary(bool value) {_word_boundary = value;}
  bool is_word_boundary() const {return _word_boundary;}
  void set_sentence_boundary(bool value) {_sentence_boundary = value;}
  bool is_sentence_boundary() const {return _sentence_boundary;}
  void set_label(int label) {_label = label;}
  int label() const {return _label;}
  vector<int> phones() const {return _phones;}
  void set_phones(vector<int> phones) {_phones = phones;}
  int phone_size() {return _phones.size();}
  ~Letter() {};
 private:
  bool _word_boundary;
  bool _sentence_boundary;
  int _label;
  vector<int> _phones;
};

#endif

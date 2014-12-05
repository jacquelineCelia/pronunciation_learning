#include <cmath>
#include <fstream>
#include <iostream>
#include "datum.h"

Datum::Datum(Config* config) {
    _config = config;
    _transcribed = false;
    _corrupted = false;
}

vector<int> Datum::letter_seq() {
    vector<int> seq;
    vector<Letter*>::iterator l_iter = _letters.begin();
    for (; l_iter != _letters.end(); ++l_iter) {
        seq.push_back((*l_iter) -> label());
    }
    return seq;
}

vector<int> Datum::Context(int i) {
    vector<int> context;
    if (_letters[i] -> label() < 0) {
        context.push_back(_letters[i] -> label());
    }
    else {
        int start = i - _config -> n_context();
        int end = i + _config -> n_context();
        for (int j = start; j <= end; ++j) {
            if (j < 0) {
                context.push_back(-1);
            }
            else if (j >= (int) _letters.size()) {
                context.push_back(-2);
            }
            else {
                context.push_back(_letters[j] -> label());
            }
        }
    }
    return context;
}

void Datum::ClearSegs() {
    vector<Segment*>::iterator s_iter = _segments.begin();
    for (; s_iter != _segments.end(); ++s_iter) {
        delete *s_iter;
    }
    _segments.clear();
}

void Datum::Save(const string& dir) {
    string filename = dir + _tag;
    // cout << filename << endl;
    ofstream fout(filename.c_str());
    if (!_corrupted) {
        vector<Letter*>::iterator l_iter = _letters.begin();
        int offset = 0;
        int segment_ptr = 0;
        // cout << "total number of letters: " << _letters.size() << endl;
        // cout << "total number of segments: " << _segments.size() << endl; 
        for (; l_iter != _letters.end(); ++l_iter) {
            // cout << (*l_iter) -> label() << ": " << (*l_iter) -> phone_size() << endl;
            if ((*l_iter) -> phone_size() == 0) {
                fout << "- - - " << (*l_iter) -> label() << endl; 
            }
            else {
                for (int i = 0; i < (*l_iter) -> phone_size(); ++i, ++segment_ptr) {
                    // cout << "segment ptr: " << segment_ptr << endl;
                    fout << offset << " " << offset + _segments[segment_ptr] -> frame_num() - 1 \
                        << " " << _segments[segment_ptr] -> id() << " ";
                    if (i == 0) {
                        fout << (*l_iter) -> label() << endl; 
                    } 
                    else {
                        fout << "-" << endl;
                    }
                    offset += _segments[segment_ptr] -> frame_num(); 
                }
            }
        }
    }
    else {
        fout << "Datum corrupted. Corrupted types are: " << endl;
        for (unsigned int i = 0; i < _corrupted_type.size(); ++i) {
            fout << _corrupted_type[i] << endl;
        }
    }
    fout.close();
}

void Datum::CheckLetterBoundNumber() {
    int number_of_normal_chars = 0;
    int minB = 0; 
    for (unsigned int i = 1; i < _letters.size(); ++i) {
        if (_letters[i] -> label() >= 0) {
             ++number_of_normal_chars;   
        }
        else {
            minB += number_of_normal_chars - \
                    floor((number_of_normal_chars + 1) / \
                    (_config -> max_num_units() + 1)) * (_config -> max_num_units() + 1);
            number_of_normal_chars = 0;
        }
    }
    bool transcribed_bounds = true;
    int total_frame_num = 0;
    for (unsigned int i = 0; i < _bounds.size(); ++i) {
        if (!(_bounds[i] -> is_boundary())) {   
            transcribed_bounds = false;
        }
        total_frame_num += _bounds[i] -> frame_num();
    }
    if (transcribed_bounds) {
        int b = _bounds.size();
        int l = _letters.size();
        int maxB = (_config -> max_num_units()) * l;
        if (b > maxB) {
            _corrupted = true;
            _corrupted_type.push_back(1);
        } 
        if (b < minB) {
            _corrupted = true;
            _corrupted_type.push_back(2);
        }
    }
    else {
        int l = _letters.size();
        int maxFrames = (l * (_config -> max_num_units()) * (_config -> max_duration())); 
        int minFrames = l; // not precise but can serve as a lower bound
        if (total_frame_num > maxFrames) {
            _corrupted = true;
            _corrupted_type.push_back(3);
        }
        if (total_frame_num < minFrames) {
            _corrupted = true;
            _corrupted_type.push_back(4);
        }
    }
}

Datum::~Datum() {
   // delete memory allocated for Letter objects
   vector<Letter*>::iterator l_iter;
   l_iter = _letters.begin();
   for (; l_iter != _letters.end(); ++l_iter) {
       delete *l_iter;
   }
   _letters.clear();
   // delete memory allocated for segment objects
   vector<Segment*>::iterator s_iter;
   s_iter = _segments.begin();
   for (; s_iter != _segments.end(); ++s_iter) {
       delete *s_iter;
   }
   _segments.clear();
   // delete memory allocated for Bound objects
   vector<Bound*>::iterator b_iter;
   b_iter = _bounds.begin();
   for (; b_iter != _bounds.end(); ++b_iter) {
       delete *b_iter;
   }
   _bounds.clear();
}

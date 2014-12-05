#include <cstdlib>
#include <iostream>
#include "segment.h"

void Segment::push_back(Bound* bound) {
    vector<float*> data = bound -> data();
    // need to fix the copy thing
    int ptr = _data.size();
    _data.resize(ptr + data.size());
    copy(data.begin(), data.end(), _data.begin() + ptr); 
    for (int i = bound -> start_frame(); i <= bound -> end_frame(); ++i) {
        _data_index.push_back(i);
    } 
}

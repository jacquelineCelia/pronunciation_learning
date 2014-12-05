#include <iostream>
#include "bound.h"

Bound::Bound(Config* config) {
    _config = config;
    _data = NULL;
    _frame_num = 0;
    _dim = _config -> dim();
    _label = -2;
    _start_frame = 0;
    _end_frame = 0;
    _is_labeled = false;
    _is_boundary = false;
}

void Bound::set_data(float** data, int frame_num) {
    _data = data;
    _frame_num = frame_num;
    for(int i = 0; i < _frame_num; ++i) {
        _data_array.push_back(_data[i]);
    }
}

void Bound::print_data() {
    for (int i = 0; i < (int) _data_array.size(); ++i) {
        for (int j = 0; j < _dim; ++j) {
            cout << _data_array[i][j] << " ";
        }
        cout << endl;
    }
}

Bound::~Bound() {
    for (int i = 0; i < _frame_num; ++i) {
        delete[] _data[i];
    }
    delete[] _data;
}

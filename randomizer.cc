#include <cstdlib>
#include <iostream>
#include "randomizer.h"

using namespace std;

Randomizer::Randomizer() {
    _data_size = 0;
    _group_num = 0;
}

void Randomizer::SetUpBatchIndices(int batch_size) {
    if (!_data_size) {
        cerr << "No data at all!" << endl;
        exit(0);
    }
    _group_num = _data_size / batch_size;
    _group_num = _data_size % batch_size ? _group_num + 1 : _group_num;
    _start_indices.resize(_group_num, 0);
    _end_indices.resize(_group_num, 0);
    for (int i = 0; i < _group_num; ++i) {
        _start_indices[i] = i * batch_size;
        _end_indices[i] = i == _group_num - 1 ? \
            _data_size - 1 : batch_size * (i + 1) - 1; 
    }
}

int Randomizer::GetStartIndex(int i) {
    return _start_indices[i % _group_num];
}

int Randomizer::GetEndIndex(int i) {
    return _end_indices[i % _group_num];
}

Randomizer::~Randomizer() {
}

#include <cmath>
#include <cstring>
#include <iostream>

#include "mixture.h"

Mixture::Mixture(Config* config) {
    _config = config;
    _dim = _config -> dim();
    _mean.resize(_dim, 0);
    _pre.resize(_dim, 0);
    _det = 0;
}

Mixture::Mixture(const Mixture& rhs) {
    _mean = rhs.mean();
    _pre = rhs.pre();
    _config = rhs.config();
    _dim = rhs.dim();
    _det = rhs.det();
    _likelihood = rhs.likelihood();
}

Mixture& Mixture::operator = (const Mixture& rhs) {
    _mean = rhs.mean();
    _pre = rhs.pre();
    _config = rhs.config();
    _dim = rhs.dim();
    _det = rhs.det();
    _likelihood = rhs.likelihood();
    return *this;
}

void Mixture::set_mean(vector<float>& mean) {
    _mean = mean;
}

void Mixture::set_pre(vector<float>& pre) {
    _pre = pre;
}

void Mixture::set_det(float det) {
    _det = det;
}

void Mixture::set_det() {
    _det = 0;
    for (int i = 0; i < _dim; ++i) {
        _det += log(_pre[i]);
    }
    _det *= 0.5;
    _det -= 0.5 * _dim * 1.83787622175;
    // log(2*3.1415926) = 1.83787622175
}

float Mixture::likelihood(float* data) {
    float likelihood = 0;
    for (int i = 0; i < _dim; ++i) {
        likelihood += (data[i] - _mean[i]) * (data[i] - _mean[i]) * _pre[i];
    }
    likelihood *= -0.5;
    return _det + likelihood;
}

float Mixture::likelihood(int i) {
    return _likelihood[i]; 
}

void Mixture::Plus(float* data) {
    for(int i = 0 ; i < _dim; ++i) {
        _mean[i] += data[i];
        _pre[i] += data[i] * data[i];
    }
}

void Mixture::Minus(float* data) {
    for (int i = 0; i < _dim; ++i) {
        _mean[i] -= data[i];
        _pre[i] -= data[i] * data[i];
    }
}

void Mixture::PreCompute(float** data, int frame_num) {
}

Mixture& Mixture::operator+= (Mixture& rhs) {
    vector<float> rhs_mean = rhs.mean();
    vector<float> rhs_pre = rhs.pre();
    for (int i = 0; i < _dim; ++i) {
        _mean[i] += rhs_mean[i];
        _pre[i] += rhs_pre[i];
    }
    return *this;
}

void Mixture::Reset() {
    _mean.resize(_dim, 0);
    _pre.resize(_dim, 0);
}

Mixture::~Mixture() {
}

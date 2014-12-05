#include "gmm.h"

GMM::GMM(Config* config) {
    _config = config;
    _mix_num = _config -> mix_num();
    _weight.resize(_mix_num, 0);
    for (int i = 0; i < _mix_num; ++i) {
        Mixture mixture(_config);
        _mixtures.push_back(mixture);
    }
}

GMM::GMM(Config* config, int mix_num) {
    _config = config;
    _mix_num = mix_num;
    _weight.resize(_mix_num, 0);
    for (int i = 0; i < _mix_num; ++i) {
        Mixture mixture(_config);
        _mixtures.push_back(mixture);
    }
}

GMM::GMM(const GMM& rhs) {
    _config = rhs.config();
    _mix_num = rhs.mix_num();
    _mixtures = rhs.mixtures(); 
    _weight = rhs.weight();
}

GMM& GMM::operator= (const GMM& rhs) {
    _config = rhs.config();
    _mix_num = rhs.mix_num();
    _mixtures = rhs.mixtures();
    _weight = rhs.weight();
    return *this;
}

void GMM::set_mixture(Mixture& mixture, int index) {
    _mixtures[index] = mixture;
}

void GMM::set_mixtures(vector<Mixture>& mixtures) {
    _mixtures = mixtures;
}

void GMM::set_weight(vector<float> weight) {
    _weight = weight; 
}

void GMM::Minus(float* data, int index) {
    --_weight[index];
    _mixtures[index].Minus(data);
}

void GMM::Plus(float* data, int index) {
    ++_weight[index];
    _mixtures[index].Plus(data);
}

vector<float> GMM::ComponentLikelihood(float* data) {
    vector<float> likelihood;
    for (int i = 0; i < _mix_num; ++i) {
        likelihood.push_back(_weight[i] + _mixtures[i].likelihood(data));
    } 
    return likelihood;
}

vector<float> GMM::ComponentLikelihood(int index) {
    vector<float> likelihood;
    for (int i = 0; i < _mix_num; ++i) {
        likelihood.push_back(_weight[i] + _mixtures[i].likelihood(index));
    } 
    return likelihood;
}

float GMM::ComputeLikehood(float* data) {
    vector<float> likelihood;
    for (int i = 0; i < _mix_num; ++i) {
        likelihood.push_back(_weight[i] + _mixtures[i].likelihood(data));
    } 
    return ToolKit::SumLogs(likelihood);
}

float GMM::ComputeLikehood(int index) {
    vector<float> likelihood;
    for (int i = 0; i < _mix_num; ++i) {
        likelihood.push_back(_weight[i] + _mixtures[i].likelihood(index));
    }
    return ToolKit::SumLogs(likelihood);
}

void GMM::ComputeLikehood(vector<float*> data, float* likelihood) {
    for (int i = 0; i < (int) data.size(); ++i) {
        likelihood[i] = ComputeLikehood(data[i]);
    }
}

void GMM::ComputeLikehood(int start_frame, int end_frame, float* likelihood) {
    for (int i = start_frame; i <= end_frame; ++i) {
        likelihood[i - start_frame] = ComputeLikehood(i);
    }
}

GMM& GMM::operator+= (GMM& rhs) {
    vector<float> rhs_weight = rhs.weight();
    for (int i = 0; i < _mix_num; ++i) {
        _weight[i] += rhs_weight[i];
        _mixtures[i] += rhs.mixture(i);
    }
    return *this;
}

void GMM::PreCompute(float** data, int frame_num) {
    for (int i = 0; i < _mix_num; ++i) {
        _mixtures[i].PreCompute(data, frame_num);
    }
}

void GMM::Save(ofstream& fout) {
    fout.write(reinterpret_cast<char*> (&_mix_num), sizeof(int));
    fout.write(reinterpret_cast<char*> (&_weight[0]), sizeof(float) * _mix_num);
    for (int m = 0; m < _mix_num; ++m) {
       float det = mixture(m).det();
       vector<float> mean = mixture(m).mean();
       vector<float> pre = mixture(m).pre();
       fout.write(reinterpret_cast<char*> (&det), sizeof(float));
       fout.write(reinterpret_cast<char*> (&mean[0]), sizeof(float) * mean.size());
       fout.write(reinterpret_cast<char*> (&pre[0]), sizeof(float) * pre.size()); 
    }
}

void GMM::Load(ifstream& fin) {
    fin.read(reinterpret_cast<char*> (&_mix_num), sizeof(int));
    fin.read(reinterpret_cast<char*> (&_weight[0]), sizeof(float) * \
            _mix_num);
    for (int m = 0; m < _mix_num; ++m) {
        float det;
        vector<float> mean(_config -> dim(), 0);  
        vector<float> pre(_config -> dim(), 0);
        fin.read(reinterpret_cast<char*> (&det), sizeof(float));
        fin.read(reinterpret_cast<char*> (&mean[0]), sizeof(float) * \
                _config -> dim());
        fin.read(reinterpret_cast<char*> (&pre[0]), sizeof(float) * \
                _config -> dim());
        _mixtures[m].set_det(det);
        _mixtures[m].set_mean(mean);
        _mixtures[m].set_pre(pre);
    }
}

void GMM::Reset() {
    _weight.resize(_mix_num, 0);
    for (size_t i = 0; i < _mixtures.size(); ++i) {
        _mixtures[i].Reset();
    }
}

GMM::~GMM() {
}

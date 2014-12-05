#include <iostream>
#include <fstream>
#include <string>
#include "config.h"

Config::Config() {
    _use_silence = false;
    _num_mixtures = 0;
}

bool Config::Load(string& fn, string& fn_gaussian) {
    ifstream fin(fn.c_str(), ios::in);
    // Emission type
    // 0: GMM
    if (!fin.good()) {return false;}
    fin >> _num_sil_states;
    //cout << "1. num silence states: " << _num_sil_states << endl;
    if (!fin.good()) {return false;}
    fin >> _num_sil_mix;
    //cout << "2. num silence mixture: " << _num_sil_mix << endl;
    if (!fin.good()) {return false;}
    fin >> _sil_self_trans_prob;
    //cout << "3. self trans prob: " << _sil_self_trans_prob << endl;
    if (!fin.good()) {return false;}
    fin >> _emission_type;
    //cout << "4. emission type: " << _emission_type << endl;
    if (!fin.good()) {return false;}
    fin >> _state_num;
    //cout << "5. state num: " << _state_num << endl;
    if (!fin.good()) {return false;}
    fin >> _n_context;
    //cout << "6. n_context: " << _n_context << endl;
    if (!fin.good()) {return false;}
    fin >> _max_num_epsilons; // max num of epsilons in a row
    //cout << "7. max num epsilons: " << _max_num_epsilons << endl;
    if (!fin.good()) {return false;}
    fin >> _max_duration; // stored in frame number
    //cout << "8. max duration: " << _max_duration << endl;
    if (!fin.good()) {return false;}
    fin >> _max_num_units; // num of units that a phone can map to
    //cout << "9. max num units: " << _max_num_units << endl;
    if (!fin.good()) {return false;}
    fin >> _cluster_num;
    //cout << "10. cluster num: " << _cluster_num << endl;
    if (!fin.good()) {return false;}
    fin >> _n_ngram;
    //cout << "11. ngram: " << _n_ngram << endl;
    if (!fin.good()) {return false;}
    fin >> _mix_num;
    //cout << "12. mixture num: " << _mix_num << endl;
    if (!fin.good()) {return false;}
    fin >> _dim;
    //cout << "13. dim: " << _dim << endl;
    if (!fin.good()) {return false;}
    fin >> _weak_limit;
    if (_weak_limit != _cluster_num) {
        //cout << "Unmatched cluster number and weak limit" << endl;
        return false;
    }
    //cout << "14. weak limit: " << _weak_limit << endl;
    if (!fin.good()) {return false;}
    fin >> _num_chars;
    //cout << "15. num chars: " << _num_chars << endl;
    if (!fin.good()) {return false;} 
    fin >> _transition_alpha; // n_expected / _num_states
    //cout << "16. transition alpha: " << _transition_alpha << endl;
    if (!fin.good()) {return false;}
    fin >> _mix_alpha;
    //cout << "17. mix alpha: " << _mix_alpha << endl;
    if (!fin.good()) {return false;}
    fin >> _gaussian_a0;
    //cout << "18. gaussian a0: " << _gaussian_a0 << endl;
    if (!fin.good()) {return false;}
    fin >> _gaussian_k0;
    //cout << "19. gaussian k0: " << _gaussian_k0 << endl;
    if (!fin.good()) {return false;}
    fin >> _ngram_weight; // don't store in log. Normal type.
    //cout << "20. ngram weight: " << _ngram_weight << endl;
    if (!fin.good()) {return false;}
    int collapsed_type;
    fin >> collapsed_type;
    if (collapsed_type == 1) {
        _is_collapsed = true;
    }
    else if (collapsed_type == 0) {
        _is_collapsed = false;
    }
    else {
        //cout << "Undefined collapse type" << endl;
        return false;
    }
    //cout << "21. collapsed type: " << _is_collapsed << endl;
    if (!fin.good()) {return false;}
    int parallel_type;
    fin >> parallel_type;
    if (parallel_type == 1) {
        _parallel = true;
    }
    else if (parallel_type == 0) {
        _parallel = false;
    }
    else {
        //cout << "Undefined parallel type. Must be either 0 or 1." << endl;
        return false;
    }
    //cout << "22. parallel: " << _parallel << endl;
    if (!fin.good()) {return false;}
    int precompute_type;
    fin >> precompute_type;
    if (precompute_type == 1) {
        _precompute = true;
    }
    else if (precompute_type == 0) {
        _precompute = false;
    }
    else {
        cout << "Undefined precompute type. Must be either 0 or 1." << endl;
        return false;
    }
    //cout << "23. precompute: " << _precompute << endl;
    if (!fin.good()) {return false;}
    //cout << "mapping alpha:" << endl;
    for (int i = 0; i <= _n_context + 1; ++i) {
        //cout << "for " << i << " context: " << endl;
        vector<float> alpha;
        for (int j = 0; j < _max_num_units; ++j) {
            float a;
            fin >> a;
            //cout << a << " ";
            alpha.push_back(a);
        }
        //cout << endl;
        _mapping_alpha.push_back(alpha);
    }
    if (!fin.good()) {return false;}
    int max_bound_num;
    fin >> max_bound_num;
    //cout << "max bound num: " << max_bound_num << endl;
    //cout << "boundary distribution: " << endl;
    for (int i = 0; i < max_bound_num; ++i) {
        float num_prob;
        fin >> num_prob;
        _boundary_distribution.push_back(num_prob);
        //cout << num_prob << " ";
    }
    //cout << endl;
    // read silence length alpha
    if (!fin.good()) {return false;}
    //cout << "Silence length alpha: " << endl;
    for (int i = 0; i <= _max_num_units; ++i) {
        float alpha;
        fin >> alpha;
        _sil_length_alpha.push_back(alpha);
        //cout << alpha << " ";
    }
    //cout << endl;
    // read space length alpha
    if (!fin.good()) {return false;}
    //cout << "Space length alpha: " << endl;
    for (int i = 0; i <= _max_num_units; ++i) {
        float alpha;
        fin >> alpha;
        _space_length_alpha.push_back(alpha);
        //cout << alpha << " ";
    }
    //cout << endl;
    // read normal length alpha
    if (!fin.good()) {return false;}
    //cout << "normal length alpha: " << endl;
    for (int j = 0; j <= _n_context + 1; ++j) {
        //cout << "for " << j << " context: " << endl;
        vector<float> normal_length_alpha;
        for (int i = 0; i <= _max_num_units; ++i) {
            float alpha;
            fin >> alpha;
            normal_length_alpha.push_back(alpha);
            //cout << alpha << " ";
        }
        //cout << endl;
        _normal_length_alpha.push_back(normal_length_alpha);
    }
    fin.close();
    return LoadGaussian(fn_gaussian);
}

bool Config::LoadGaussian(string& fn_gaussian) {
    ifstream fgaussian(fn_gaussian.c_str(), ios::binary);
    if (!fgaussian.good()) {
        cout << "Cannot load Gaussian Prior" << endl;
        return false;
    }
    cout << "Loading Gaussian" << endl;
    float weight;
    fgaussian.read(reinterpret_cast<char*> (&weight), sizeof(float));
    float mean[_dim];
    float pre[_dim];
    fgaussian.read(reinterpret_cast<char*> (mean), sizeof(float) * _dim);
    fgaussian.read(reinterpret_cast<char*> (pre), sizeof(float) * _dim);
    _gaussian_u0.assign(mean, mean + _dim);
    _gaussian_b0.assign(pre, pre + _dim);
    for (int i = 0; i < _dim; ++i) {
        _gaussian_b0[i] = _gaussian_a0 / _gaussian_b0[i];
    }
    fgaussian.close(); 
    return true;
}

vector<float> Config::length_alpha(int index, vector<int> label) {
    if (index == -3) {
        return _space_length_alpha;
    }
    else if (index == -10 || index == -20) {
        return _sil_length_alpha;
    }
    else {
        int context_length = label.size() == 1 && label[0] == -100 ? \
                             0 : (label.size() + 1) / 2;
        return _normal_length_alpha[context_length];
    }
}

void Config::print() {
    cout << "Silence state: " << _num_sil_states << endl;
    cout << "Silence mix: " << _num_sil_mix << endl;
    cout << "Silence self trans: " << _sil_self_trans_prob << endl;
    cout << "Emission type: " << _emission_type << endl;
    cout << "State number: " << _state_num << endl;
    cout << "N context: " << _n_context << endl;
    cout << "Max num of epsilons: " << _max_num_epsilons << endl;
    cout << "Max duration: " << _max_duration << endl;
    cout << "Max num units: " << _max_num_units << endl;
    cout << "Cluster num: " << _cluster_num << endl;
    cout << "Ngram num: " << _n_ngram << endl;
    cout << "Mix num: " << _mix_num << endl;
    cout << "Dim: " << _dim << endl;
    cout << "Weak limit: " << _weak_limit << endl;
    cout << "Num of chars: " << _num_chars << endl;
    cout << "Transition alaph: " << _transition_alpha << endl;
    cout << "Mix alpha: " << _mix_alpha << endl;
    cout << "Gaussian alpha: " << _gaussian_a0 << endl;
    cout << "Gaussian kappa: " << _gaussian_k0 << endl;
    cout << "Ngram weight: " << _ngram_weight << endl;
    cout << "Collapsed type: " << _is_collapsed << endl;
    cout << "Parallel: " << _parallel << endl;
    cout << "Precompute: " << _precompute << endl;
    cout << "Mapping alpha: " << endl;
    for (int i = 0; i <= _n_context + 1; ++i) {
        for (int j = 0; j < _max_num_units; ++j) {
            cout << _mapping_alpha[i][j] << " ";
        }
        cout << endl;
    }
    cout << "Boundary distribution: " << endl;
    for (int i = 0; i < (int) _boundary_distribution.size(); ++i) {
        cout << _boundary_distribution[i] << " ";
    }
    cout << endl;
    cout << "Silence length alpha: " << endl;
    for (int i = 0; i <= _max_num_units; ++i) {
        cout << _sil_length_alpha[i] << " ";
    }
    cout << endl;
    cout << "Space length alpha: " << endl;
    for (int i = 0; i <= _max_num_units; ++i) {
        cout << _space_length_alpha[i] << " ";
    }
    cout << endl;
    cout << "Normal length alpha: " << endl;
    for (int j = 0; j <= _n_context + 1; ++j) {
        vector<float> normal_length_alpha;
        for (int i = 0; i <= _max_num_units; ++i) {
            cout << _normal_length_alpha[j][i] << " ";
        }
        cout << endl;
    }
    cout << "Gaussian mean: " << endl;
    for (int i = 0; i < _dim; ++i) {
        cout << _gaussian_u0[i] << " ";
    }
    cout << endl;
    cout << "Gaussian pre: " << endl;
    for (int i = 0; i < _dim; ++i) {
        cout << _gaussian_b0[i] << " ";
    }
    cout << endl;
}

bool Config::LoadSeedingMixtures(string& fn_mixtures) {
    cout << "Must load the regular config file before loading gaussians!" << endl;
    ifstream fmixture(fn_mixtures.c_str(), ios::binary);
    if (!fmixture.good()) {
        return false;
    }
    fmixture.seekg(0, fmixture.end);
    int length = fmixture.tellg();
    fmixture.seekg(0, fmixture.beg);
    // mean/pre vectors + weight counted in bytes
    int size_per_mixture = (_dim * 2 + 1) * sizeof(float);
    if (length % size_per_mixture) {
        cout << "Input format may not match" << endl;
        return false;
    }
    _num_mixtures = length / size_per_mixture;
    for (int i = 0; i < _num_mixtures; ++i) {
        GaussianSeed gs(_dim); 
        float weight;
        fmixture.read(reinterpret_cast<char*> (&weight), sizeof(float));
        vector<float> mean(_dim, 0);
        vector<float> pre(_dim, 0);
        fmixture.read(reinterpret_cast<char*> (&mean[0]), sizeof(float) * _dim);
        fmixture.read(reinterpret_cast<char*> (&pre[0]), sizeof(float) * _dim);
        gs.set_mean(mean);
        for (int d = 0; d < _dim; ++d) {
            pre[d] = _gaussian_a0 / pre[d];
        }
        gs.set_pre(pre);
        _mixtures.push_back(gs);
    }
    fmixture.close();
    return true; 
}

Config::~Config() {
}

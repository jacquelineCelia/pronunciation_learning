#include <iostream>
#include <cmath>
#include <math.h>
#include <cstdlib>
#include "l2s.h"

#define DEBUG false 
#define SEG_DEBUG false 
#define MIN_PROB_VALUE -70000000

L2S::L2S(Config* config, const vector<int>& label) {
    _config = config;
    _label = label;
    set_parent_label();
    int central_label = label[(label.size() - 1) / 2];
    // Set length alpha
    _length_alpha = _config -> length_alpha(central_label, label);
    _total_length_alpha = 0;
    for (int i = 0; i <= _config -> max_num_units(); ++i) {
        _total_length_alpha += _length_alpha[i];
    }
    // Set mapping alpha
    if (_label[0] == -100) {
        _mapping_alpha = _config -> mapping_alpha(0);
    }
    else {
        _mapping_alpha = config -> mapping_alpha((_label.size() + 1) / 2);
    }
    // Create container for probabilities
    _length_prior.resize(_config -> max_num_units() + 1, 0);
    for (int i = 0; i < _config -> max_num_units(); ++i) {
        vector<vector<float> > mapping_likelihood_i;
        for (int j = 0; j <= i; ++j) {
            vector<float> mapping_likelihood_ij(_config -> cluster_num(), 0);
            mapping_likelihood_i.push_back(mapping_likelihood_ij);
        }
        _mapping_likelihood.push_back(mapping_likelihood_i);
    }
    _N = 0;
    _updated = false;
    _is_mapping_likelihood_fixed = false;
    _is_length_prior_fixed = false;
    _is_leaf = central_label >= 0 && (int) (label.size() - 1) / 2 == _config -> n_context() ? true : false;
    _is_collapsed = _config -> is_collapsed() && _is_leaf ? true : false;
    _parent = NULL;
}

L2S::L2S(L2S& rhs) {
    _config = rhs.config();
    _label = rhs.label();
    _parent_label = rhs.parent_label();
    _length_alpha = rhs.length_alpha();
    _total_length_alpha = 0;
    for (int i = 0; i <= _config -> max_num_units(); ++i) {
        _total_length_alpha += _length_alpha[i];
    }
    _mapping_alpha = rhs.mapping_alpha();
    _length_prior = rhs.length_prior();
    _mapping_likelihood = rhs.mapping_likelihood();
    _N = rhs.N();
    _updated = rhs.updated();
    _is_mapping_likelihood_fixed = rhs.is_mapping_likelihood_fixed(); 
    _is_length_prior_fixed = rhs.is_length_prior_fixed();
    _is_leaf = rhs.is_leaf();
    _is_collapsed = rhs.is_collapsed();
    _parent = NULL;
}

L2S& L2S::operator= (L2S& rhs) {
    _config = rhs.config();
    _label = rhs.label();
    _parent_label = rhs.parent_label();
    _length_alpha = rhs.length_alpha();
    _total_length_alpha = 0;
    for (int i = 0; i <= _config -> max_num_units(); ++i) {
        _total_length_alpha += _length_alpha[i];
    }
    _mapping_alpha = rhs.mapping_alpha();
    _length_prior = rhs.length_prior();
    _mapping_likelihood = rhs.mapping_likelihood();
    _N = rhs.N();
    _updated = rhs.updated();
    _is_mapping_likelihood_fixed = rhs.is_mapping_likelihood_fixed();
    _is_length_prior_fixed = rhs.is_length_prior_fixed();
    _is_leaf = rhs.is_leaf();
    _is_collapsed = rhs.is_collapsed();
    // need to deal with old parent (delete?)
    // don't know when this operator will be called.
    // It should not be called actually.
    _parent = NULL;
    return *this;
}

void L2S::set_mapping_likelihood(vector<vector<vector<float> > >& \
                            mapping_likelihood) {
    _mapping_likelihood = mapping_likelihood;
}

void L2S::set_length_prior(vector<float>& length_prior) {
    _length_prior = length_prior;
}

void L2S::set_label(vector<int>& label) {
    _label = label;
}

void L2S::set_parent(L2S* parent) {
    _parent = parent;
}

void L2S::set_length_alpha(vector<float> length_alpha) {
    _length_alpha = length_alpha;
}

void L2S::set_mapping_alpha(vector<float>& mapping_alpha) {
    _mapping_alpha = mapping_alpha;
}

void L2S::set_parent_label() {
    if (_parent_label.size() == 0) {
        if (_label.size() == 1) {
            if (center_label() >= 0) {
                _parent_label.push_back(-100);
            }
        }
        else {
            for (int i = 1; i < (int) _label.size() - 1; ++i) {
               _parent_label.push_back(_label[i]);
            }
        }
    }
}

float L2S::Prior(int len) {
    if (!_is_collapsed) {
        return _length_prior[len];
    }
    else {
        if (_is_length_prior_fixed) {
            return _length_prior[len];
        }
        else {
            float total_N = 0;
            for (size_t i = 0; i < _length_prior.size(); ++i) {
                total_N += _length_prior[i];
            }
            if (total_N != _N) {
                cerr << "Mismatched total_N and N: " << total_N << " " << _N << endl;
                exit(1);
            }
            return log(_length_prior[len] + _length_alpha[len]) \
                    - log(total_N + _total_length_alpha);
        }
    }
}

float L2S::Likelihood(vector<int> cluster_seq) {
    if (!_is_collapsed) {
        int len = cluster_seq.size();
        float likelihood = 0;
        for (int i = 0; i < len; ++i) {
            likelihood += _mapping_likelihood[len - 1][i][cluster_seq[i]];
        }
        return likelihood;
    }
    else {
        float likelihood = 0;
        int len = cluster_seq.size(); 
        if (_is_mapping_likelihood_fixed) {
           for (int i = 0; i < len; ++i) {
               likelihood += _mapping_likelihood[len - 1][i][cluster_seq[i]];
           } 
        }
        else {
           for (int i = 0 ; i < len; ++i) {
               likelihood += \
                   GetCollapsedLikelihood(len - 1, i, cluster_seq[i]);
           }
        }
        return likelihood;
    }
}

float L2S::Likelihood(int len, int j, int k) {
    if (!_is_collapsed) {
        return _mapping_likelihood[len - 1][j][k]; 
    }
    else {
        float likelihood = 0;
        if (_is_mapping_likelihood_fixed) {
            likelihood = _mapping_likelihood[len - 1][j][k];
        }
        else {
            likelihood = GetCollapsedLikelihood(len - 1, j, k);
        }
        return likelihood;
    }
}

void L2S::Plus(vector<int> phones, int count) {
    int len = phones.size();
    if (!_is_mapping_likelihood_fixed) {
        for (int i = 0 ; i < len; ++i) {
            _mapping_likelihood[len - 1][i][phones[i]] += count;
        }
    }
    if (!_is_length_prior_fixed) {
        _length_prior[len] += count;
    }
    _N += count;
}

void L2S::Minus(vector<int> phones, int count) {
    int len = phones.size();
    if (!_is_mapping_likelihood_fixed) {
        for (int i = 0 ; i < len; ++i) {
            _mapping_likelihood[len - 1][i][phones[i]] -= count;
        }
    }
    if (!_is_length_prior_fixed) {
        _length_prior[len] -= count;
    }
    _N -= count;
}

float L2S::GetCollapsedLikelihood(int i, int j, int k) {
    int len = i + 1;
    if (_parent != NULL) {
        float parent_prob;
        if (_parent -> is_collapsed()) {
            parent_prob = _parent -> GetCollapsedLikelihood(i, j, k);
        }
        else {
            parent_prob = _parent -> mapping_likelihood_ijk(i, j, k);
        }
        return ToolKit::SumLogs(log(_mapping_likelihood[i][j][k]), \
                log(_mapping_alpha[i]) + parent_prob) \
            - log(_length_prior[len] + _mapping_alpha[i]);
    }
    else {
        if (_is_collapsed) {
            return log(_mapping_likelihood[i][j][k] + _mapping_alpha[i]) - \
                    log(_length_prior[len] + (_config -> weak_limit()) * _mapping_alpha[i]); 
        }
        else {
            return _mapping_likelihood[i][j][k];
        }
    }
}

ProbList<int>****
L2S::ConstructIndividualSegProbTable(vector<Bound*>& bounds, float*** seg_prob_given_cluster) {
    const int b = bounds.size();
    const int max_duration = _config -> max_duration();
    const int max_num_units = _config -> max_num_units();
    const int cluster_num = _config -> cluster_num();
    ProbList<int>**** prob_container = new ProbList<int>*** [max_num_units];
    for (int i = 0; i < max_num_units; ++i) {
        prob_container[i] = new ProbList<int>** [i + 1];
        for (int j = 0; j <= i; ++j) {
            prob_container[i][j] = new ProbList<int>* [b];  
            for (int k = 0; k < b; ++k) {
                prob_container[i][j][k] = new ProbList<int> [b];
            }
        }
    }
    int total_frame_num = 0;
    vector<int> accumulated_frame_num(b, 0);
    for (int i = 0; i < b; ++i) {
        total_frame_num += bounds[i] -> frame_num();
        accumulated_frame_num[i] = total_frame_num;
    }
    vector<vector<vector<float> > > prob_k_l;
    for (int i = 0; i < max_num_units; ++i) {
        vector<vector<float> > prob_k_fixed_i;
        for (int j = 0; j <= i; ++j) {
            vector<float> prob_k_fixed_i_j(cluster_num, 0);
            for (int k = 0; k < cluster_num; ++k) {
                prob_k_fixed_i_j[k] = Likelihood(i + 1, j, k);
                if (isnan(prob_k_fixed_i_j[k])) {
                    cerr << "Found nan in ConstructIndividualSegProbTable. L2S label: " << endl;
                    for (size_t t = 0; t < _label.size(); ++t) {
                        cerr << _label[t] << " ";
                    }
                    cerr << endl;
                }
            }
            prob_k_fixed_i.push_back(prob_k_fixed_i_j);
        }
        prob_k_l.push_back(prob_k_fixed_i);
    }
    for (int i = 0; i < max_num_units; ++i) {
        for (int j = 0; j <= i; ++j) {
            for (int b1 = 0; b1 < b; ++b1) {
                int start_frame = b1 == 0 ? 0 : accumulated_frame_num[b1 - 1];
                for (int b2 = b1; b2 < b; ++b2) {
                    int duration = accumulated_frame_num[b2] - start_frame;
                    if (duration <= max_duration || b2 == b1) {
                        vector<float> prob_k_l_mul_b_k = prob_k_l[i][j]; 
                        for (int k = 0; k < cluster_num; ++k) {
                            prob_k_l_mul_b_k[k] += seg_prob_given_cluster[k][b1][b2];
                        }
                        prob_container[i][j][b1][b2].assign(prob_k_l_mul_b_k);
                    }
                }
            }
        }
    }
    return prob_container;
}

ProbList<int>** L2S::ConstructSegProbTable(vector<Bound*>& bounds, \
                        ProbList<int>**** individual_prob_table) {
    const int b = bounds.size();
    const int max_num_units = _config -> max_num_units();
    const int max_duration = _config -> max_duration();
    vector<int> accumulated_frame_num;
    int total_frame_num = 0;
    for (int i = 0 ; i < b; ++i) {
        total_frame_num += bounds[i] -> frame_num();
        accumulated_frame_num.push_back(total_frame_num);
    }
    ProbList<int>** prob_table = new ProbList<int>* [b];
    for (int i = 0; i < b; ++i) {
        prob_table[i] = new ProbList<int> [b];
    }
    const float prior_1 = Prior(1);
    const float prior_2 = Prior(2);
    for (int b1 = 0; b1 < b; ++b1) {
        int start_frame = b1 == 0 ? 0 : accumulated_frame_num[b1 - 1];
        for (int b2 = b1; b2 < b; ++b2) {
            int duration = accumulated_frame_num[b2] - start_frame;
            if (duration <= max_num_units * max_duration || b2 - b1 <= 1) {
                vector<float> probs;
                probs.reserve(b2 - b1 + 1);
                vector<int> index;
                index.reserve(b2 - b1 + 1);
                if (duration <= max_duration || b2 == b1) {
                    probs.push_back(prior_1 + \
                            individual_prob_table[0][0][b1][b2].value());
                    index.push_back(b2);
                }
                if (b2 - b1 >= 1) {
                    for (int p = b1; p < b2; ++p) {
                        probs.push_back(prior_2 + \
                                individual_prob_table[1][0][b1][p].value() + \
                                individual_prob_table[1][1][p + 1][b2].value());
                        index.push_back(p);
                    }
                }
                prob_table[b1][b2].assign(probs, index);
            }
        }
    }
    return prob_table;
}

L2S& L2S::operator-= (const L2S& rhs) {
    vector<float> rhs_length_prior = rhs.length_prior();
    vector<vector<vector<float> > > rhs_mapping_likelihood = rhs.mapping_likelihood();
    _N -= rhs.N();
    if ((int) rhs_length_prior.size() != (_config -> max_num_units() + 1)) {
        exit(1);
    }
    else {
        for (size_t i = 0 ; i < _length_prior.size(); ++i) {
            _length_prior[i] -= rhs_length_prior[i];
        }
    }
    if ((int) rhs_mapping_likelihood[0][0].size() != _config -> cluster_num()) {
        exit(1);
    }
    else {
        for (int i = 0; i < _config -> max_num_units(); ++i) {
            for (int j = 0; j <= i; ++j) {
                for (int k = 0; k < _config -> cluster_num(); ++k) {
                    _mapping_likelihood[i][j][k] -= rhs_mapping_likelihood[i][j][k];
                }
            }
        }
    }
    return *this;
}

L2S& L2S::operator+= (const L2S& rhs) {
    vector<float> rhs_length_prior = rhs.length_prior();
    vector<vector<vector<float> > > rhs_mapping_likelihood = rhs.mapping_likelihood();
    _N += rhs.N();
    if (rhs_length_prior.size() != _length_prior.size()) {
        cout << "Lenth_prior size not matched" << endl;
        exit(1);
    }
    else {
        for (size_t i = 0 ; i < _length_prior.size(); ++i) {
            _length_prior[i] += rhs_length_prior[i];
        }
    }
    if (rhs_mapping_likelihood[0][0].size() != _mapping_likelihood[0][0].size()) {
        cout << "Mapping_likelihood size not matched" << endl;
        exit(1);
    }
    else {
        for (size_t i = 0; i < _length_prior.size() - 1; ++i) {
            for (size_t j = 0; j <= i; ++j) {
                for (size_t k = 0; k < _mapping_likelihood[0][0].size(); ++k) {
                    _mapping_likelihood[i][j][k] += rhs_mapping_likelihood[i][j][k];
                }
            }
        }
    }
    return *this;
}

void L2S::Reset() {
    _length_prior.clear();
    _mapping_likelihood.clear();
    for (int i = 0 ; i <= _config -> max_num_units(); ++i) {
        _length_prior.push_back(0);
    }
    for (int i = 0; i < _config -> max_num_units(); ++i) {
        vector<vector<float> > mapping_counter;
        for (int j = 0 ; j <= i; ++j) {
           vector<float> buckets(_config -> cluster_num(), 0); 
           mapping_counter.push_back(buckets);
        }
        _mapping_likelihood.push_back(mapping_counter);
    }
    _N = 0;
}

void L2S::Save(ofstream& fout) {
    int label_length = _label.size();
    fout.write(reinterpret_cast<char*> (&label_length), sizeof(int));
    fout.write(reinterpret_cast<char*> (&_label[0]), sizeof(int) * _label.size());
    int length_prior_length = _length_prior.size();
    fout.write(reinterpret_cast<char*> (&length_prior_length), sizeof(int));
    fout.write(reinterpret_cast<char*> (&_length_prior[0]), sizeof(float) * _length_prior.size());
    for (int i = 0; i < _config -> max_num_units(); ++i) {
        for (int j = 0; j <= i; ++j) {
            fout.write(reinterpret_cast<char*> (&_mapping_likelihood[i][j][0]), sizeof(float) * _mapping_likelihood[i][j].size());
        }
    }
    int length_alpha_length = _length_alpha.size();
    fout.write(reinterpret_cast<char*> (&length_alpha_length), sizeof(float));
    fout.write(reinterpret_cast<char*> (&_length_alpha[0]), sizeof(float) * length_alpha_length);
    int mapping_alpha_length = _mapping_likelihood.size();
    fout.write(reinterpret_cast<char*> (&mapping_alpha_length), sizeof(int));
    fout.write(reinterpret_cast<char*> (&_mapping_alpha[0]), sizeof(float) * _mapping_alpha.size());
    fout.write(reinterpret_cast<char*> (&_N), sizeof(int));
    int length_prior_fixed = _is_length_prior_fixed ? 1 : 0;
    int mapping_likelihood_fixed = _is_mapping_likelihood_fixed ? 1 : 0;
    int collapsed = _is_collapsed ? 1 : 0;
    int leaf = _is_leaf ? 1 : 0;
    fout.write(reinterpret_cast<char*> (&length_prior_fixed), sizeof(int));
    fout.write(reinterpret_cast<char*> (&mapping_likelihood_fixed), sizeof(int));
    fout.write(reinterpret_cast<char*> (&collapsed), sizeof(int));
    fout.write(reinterpret_cast<char*> (&leaf), sizeof(int));
}

void L2S::Load(ifstream& fin) {
    int length_prior_length;
    fin.read(reinterpret_cast<char*> (&length_prior_length), sizeof(int));
    fin.read(reinterpret_cast<char*> (&_length_prior[0]), sizeof(float) * _length_prior.size());
    for (int i = 0; i < _config -> max_num_units(); ++i) {
        for (int j = 0; j <= i; ++j) {
            fin.read(reinterpret_cast<char*> (&_mapping_likelihood[i][j][0]), sizeof(float) * _mapping_likelihood[i][j].size());
        }
    }
    /*
    int buffer[1];
    fin.read(reinterpret_cast<char*> (buffer), sizeof(int) * 1);
    */
    int length_alpha_length;
    fin.read(reinterpret_cast<char*> (&length_alpha_length), sizeof(int));
    fin.read(reinterpret_cast<char*> (&_length_alpha[0]), sizeof(float) * length_alpha_length);
    int mapping_alpha_length;
    fin.read(reinterpret_cast<char*> (&mapping_alpha_length), sizeof(int));
    fin.read(reinterpret_cast<char*> (&_mapping_alpha[0]), sizeof(float) * mapping_alpha_length);
    fin.read(reinterpret_cast<char*> (&_N), sizeof(int));
    int length_prior_fixed;
    int mapping_likelihood_fixed;
    int collapsed;
    int leaf;
    fin.read(reinterpret_cast<char*> (&length_prior_fixed), sizeof(int));
    _is_length_prior_fixed = length_prior_fixed == 1 ? true : false;
    fin.read(reinterpret_cast<char*> (&mapping_likelihood_fixed), sizeof(int));
    _is_mapping_likelihood_fixed = mapping_likelihood_fixed == 1 ? true : false;
    fin.read(reinterpret_cast<char*> (&collapsed), sizeof(int));
    _is_collapsed = collapsed == 1 ? true : false;
    fin.read(reinterpret_cast<char*> (&leaf), sizeof(int));
    _is_leaf = leaf == 1 ? true : false;
}

void L2S::print_length_prior() {
    for (unsigned int i = 0; i < _length_prior.size(); ++i) {
        cout << _length_prior[i] << " ";
    }
    cout << endl;
}

void L2S::print_mapping_likelihood() {
    for (int i = 0; i < _config -> max_num_units(); ++i){
        for (int j = 0; j <= i; ++j) {
            for (int k = 0; k < _config -> cluster_num(); ++k) {
                if (_is_collapsed) {
                    cout << _mapping_likelihood[i][j][k] << " ";
                }
                else {
                    cout << _length_prior[i + 1] + _mapping_likelihood[i][j][k] << " ";
                }
            }
            cout << endl;
        }
    }
}

void L2S::print_label() {
    cout << "Label: ";
    for (unsigned int i = 0; i < _label.size(); ++i) {
        cout << _label[i] << " ";
    }
    cout << endl;
}

#include <cstdlib>
#include <set>

#include "counter.h"
#include "rvg.h"

Counter::Counter(Config* config) {
    _config = config;
    _weak_limit = _config -> weak_limit();
    _num_chars = _config -> num_chars();
    for (int i = 0; i < _weak_limit; ++i) {
        Cluster* c = new Cluster(_config, i);
        _cluster_counter.push_back(c);
    }
    _cluster_counter[0] -> set_is_fixed(true);

    vector<int> sen_beginning(1, -10);
    vector<int> sen_ending(1, -20);
    vector<int> space(1, -3);
    SetSpecialL2S(sen_beginning);
    SetSpecialL2S(sen_ending);
    SetSpecialL2S(space);
}

void Counter::Initialize(Config* config) {
    _config = config;
    _weak_limit = _config -> weak_limit();
    _num_chars = _config -> num_chars();
    for (int i = 0; i < _weak_limit; ++i) {
        Cluster* c = new Cluster(_config, i);
        _cluster_counter.push_back(c);
    }
    _cluster_counter[0] -> set_is_fixed(true);

    vector<int> sen_beginning(1, -10);
    vector<int> sen_ending(1, -20);
    vector<int> space(1, -3);
    SetSpecialL2S(sen_beginning);
    SetSpecialL2S(sen_ending);
    SetSpecialL2S(space);
}

Counter& Counter::operator+= (Counter& rhs) {
    vector<Cluster*> rhs_cluster_counter = rhs.clusters();
    map<vector<int>, L2S*> rhs_l2s_counter = rhs.l2s();
    // Sum ClusterCounters
    if (rhs.weak_limit() != _weak_limit) {
        exit(3);
    }
    else {
        for (int i = 0 ; i < _weak_limit; ++i) {
            (*_cluster_counter[i]) += *rhs_cluster_counter[i];
        }
    }
    // Sum L2SCounters
    map<vector<int>, L2S*>::iterator l_iter = rhs_l2s_counter.begin();
    for (; l_iter != rhs_l2s_counter.end(); ++l_iter) {
        if (_l2s_counter.find(l_iter -> first) == _l2s_counter.end()) {
            L2S* l2s = new L2S(_config, l_iter -> first); 
            _l2s_counter[l_iter -> first] = l2s;
        }
        (*_l2s_counter[l_iter -> first]) += (*l_iter -> second);
    }
    return *this;
}

void Counter::SetSpecialL2S(vector<int> label) {
    L2S* special_l2scounter = new L2S(_config, label);
    _l2s_counter[label] = special_l2scounter;
    special_l2scounter -> set_parent(NULL);
}

void Counter::CreateCountsForNonLeaves() {
    set<L2S*> to_be_processed; 
    map<vector<int>, L2S*>::iterator l_iter = _l2s_counter.begin();
    for (; l_iter != _l2s_counter.end(); ++l_iter) {
        if (l_iter -> second -> is_leaf()) {
            to_be_processed.insert(l_iter -> second);
        }
        else {
            // Don't reset -10, -20, -3
            if (l_iter -> second -> center_label() >= 0 || l_iter -> second -> center_label() == -100) {
                l_iter -> second -> Reset();
            }
        }
    }
    RandomVarGen rvg;
    while (to_be_processed.size()) {
        set<L2S*> parents_to_be_processed;
        set<L2S*>::iterator iter = to_be_processed.begin();
        for (; iter != to_be_processed.end(); ++iter) {
            vector<int> parent_label = (*iter) -> parent_label();
            if (_l2s_counter.find(parent_label) == _l2s_counter.end()) {
                L2S* new_l2s_counter = new L2S(_config, parent_label);
                _l2s_counter[parent_label] = new_l2s_counter;
            }
            if (_l2s_counter[parent_label] -> center_label() >= 0) {
                parents_to_be_processed.insert(_l2s_counter[parent_label]);
            }
            vector<float> counter_length_prior = (*iter) -> length_prior();
            vector<vector<vector<float> > > counter_mapping_likelihood = (*iter) -> mapping_likelihood();
            L2S pseudo_counter(_config, parent_label);
            vector<float> length_prior = pseudo_counter.length_prior();
            vector<vector<vector<float> > > mapping_likelihood = pseudo_counter.mapping_likelihood();
            int N = 0;
            for (int i = 0; i <= _config -> max_num_units(); ++i) {
                if (!i) {
                    float alpha0 = ((*iter) -> length_alpha())[0];
                    for (int n = 0; n < counter_length_prior[i]; ++n) {
                        length_prior[i] += rvg.GetUniformSample() <= (alpha0 / (n + alpha0));
                    }
                }
                else {
                    float alpha = (*iter) -> mapping_alpha()[i - 1];
                    for (int j = 0; j < i; ++j) {
                        for (int k = 0; k < _config -> cluster_num(); ++k) {
                            for (int n = 0; n < counter_mapping_likelihood[i - 1][j][k]; ++n) {
                                float value = rvg.GetUniformSample() <= (alpha / (n + alpha));
                                mapping_likelihood[i - 1][j][k] += value;
                                if (!j) {
                                    length_prior[i] += value;
                                }
                            }
                        }
                    }
                }
                N += length_prior[i];
            }
            pseudo_counter.set_length_prior(length_prior);
            pseudo_counter.set_mapping_likelihood(mapping_likelihood);
            pseudo_counter.set_N(N);
            *_l2s_counter[parent_label] += pseudo_counter;
        }
        to_be_processed = parents_to_be_processed;
    }
}

void Counter::ShowL2S() {
    map<vector<int>, L2S*>::iterator l_iter = _l2s_counter.begin();
    for (; l_iter != _l2s_counter.end(); ++l_iter) {
        l_iter -> second -> print_label();
        l_iter -> second -> print_length_prior();
        l_iter -> second -> print_mapping_likelihood();
    }
}

void Counter::Reset() {
    vector<Cluster*>::iterator iter = _cluster_counter.begin();
    for (; iter != _cluster_counter.end(); ++iter) {
        (*iter) -> Reset();
    }
    map<vector<int>, L2S*>::iterator l_iter = _l2s_counter.begin();
    for (; l_iter != _l2s_counter.end(); ++l_iter) {
        l_iter -> second -> Reset();
    }
}

Counter::~Counter() {
    vector<Cluster*>::iterator c_iter = _cluster_counter.begin();
    for (; c_iter != _cluster_counter.end(); ++c_iter) {
        delete *c_iter;
    }
    map<vector<int>, L2S*>::iterator l_iter = _l2s_counter.begin();
    for (; l_iter != _l2s_counter.end(); ++l_iter) {
        delete l_iter -> second;
    }
}

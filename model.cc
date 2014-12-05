#include <fstream>
#include <iostream>
#include <cstdlib>
#include "model.h"

#define MIN_PROB_VALUE -70000000

Model::Model(Config* config) {
    _config = config;
    _weak_limit = _config -> weak_limit();
    _num_chars = _config -> num_chars();
}

Model::Model(Model& rhs) {
    _config = rhs.config();
    _weak_limit = rhs.weak_limit();
    _num_chars = rhs.num_chars();
    vector<Cluster*> rhs_clusters = rhs.clusters();
    for (int i = 0 ; i < _weak_limit; ++i) {
        Cluster* c = new Cluster(*rhs_clusters[i]);
        _clusters.push_back(c);
    }
    map<vector<int>, L2S*> rhs_l2s = rhs.l2s();
    vector<int> no_context_label(1, -100);
    L2S* no_context_l2s = new L2S(*rhs_l2s[no_context_label]);
    _l2s[no_context_label] = no_context_l2s;
    map<vector<int>, L2S*>::iterator l_iter = rhs_l2s.begin();
    for (; l_iter != rhs_l2s.end(); ++l_iter) {
        if (_l2s.find(l_iter -> first) == _l2s.end()) {
            L2S* l2s = new L2S(*(l_iter -> second));
            vector<int> label = l2s -> label();
            vector<int> parent_label = l2s -> parent_label();
            _l2s[label] = l2s;
            while (_l2s.find(parent_label) == _l2s.end()) {
                if (rhs_l2s.find(parent_label) == rhs_l2s.end()) {
                    cout << "Parent path does not exist!" << endl;
                    exit(-1);
                }
                L2S* parent_l2s = new L2S(*rhs_l2s[parent_label]);
                _l2s[label] -> set_parent(parent_l2s);
                _l2s[parent_label] = parent_l2s;
                label = parent_label;
                parent_label = parent_l2s -> parent_label();
            }
            _l2s[label] -> set_parent(_l2s[parent_label]);
        }
    }
    SetParentOfSpecialL2S();
}

Model& Model::operator= (Model& rhs) {
    // Remove previous model setup
    for (int i = 0; i < (int) _clusters.size(); ++i) {
        delete _clusters[i];
    }
    _clusters.clear();
    map<vector<int>, L2S*>::iterator l_iter = _l2s.begin();
    for (; l_iter != _l2s.end(); ++l_iter) {
        delete (l_iter -> second);
    } 
    _l2s.clear();
    // Assign values of rhs
    _config = rhs.config();
    _weak_limit = rhs.weak_limit();
    _num_chars = rhs.num_chars();
    vector<Cluster*> rhs_clusters = rhs.clusters();
    for (int i = 0 ; i < _weak_limit; ++i) {
        Cluster* c = new Cluster(*rhs_clusters[i]);
        _clusters.push_back(c);
    }
    map<vector<int>, L2S*> rhs_l2s = rhs.l2s();
    l_iter = rhs_l2s.begin();
    for (; l_iter != rhs_l2s.end(); ++l_iter) {
        L2S* l2s = new L2S(*(l_iter -> second));
        _l2s[l_iter -> first] = l2s; 
    }
    l_iter = _l2s.begin();
    for (; l_iter != _l2s.end(); ++l_iter) {
        if (l_iter -> second -> center_label() >= 0) {
            vector<int> parent_label = l_iter -> second -> parent_label();
            if (_l2s.find(parent_label) == _l2s.end()) {
                cout << "Cannot find parent" << endl;
                exit(-1);
            }
            l_iter -> second -> set_parent(_l2s[parent_label]);
        }
    }
    SetParentOfSpecialL2S();
    return *this;
}

void Model::CopyL2SCounts (Counter& rhs) {
    // Remove previous model setup
    map<vector<int>, L2S*>::iterator l_iter = _l2s.begin();
    for (; l_iter != _l2s.end(); ++l_iter) {
        delete (l_iter -> second);
    } 
    _l2s.clear();
    // Assign values of rhs
    map<vector<int>, L2S*> rhs_l2s = rhs.l2s();
    l_iter = rhs_l2s.begin();
    for (; l_iter != rhs_l2s.end(); ++l_iter) {
        L2S* l2s = new L2S(*(l_iter -> second));
        _l2s[l_iter -> first] = l2s; 
    }
    l_iter = _l2s.begin();
    for (; l_iter != _l2s.end(); ++l_iter) {
        if ((l_iter -> first)[0] != -100) {
            vector<int> parent_label = l_iter -> second -> parent_label();
            if (_l2s.find(parent_label) == _l2s.end()) {
                cout << "Cannot find parent in CopyL2SCounts" << endl;
                exit(-1);
            }
            l_iter -> second -> set_parent(_l2s[parent_label]);
        }
    }
    vector<int> sen_beginning(1, -10);
    vector<int> sen_ending(1, -20);
    vector<int> space(1, -3);
    SetSpecialL2SMappingLikelihoodFixed(sen_beginning);
    SetSpecialL2SMappingLikelihoodFixed(sen_ending);
    SetSpecialL2SMappingLikelihoodFixed(space);
    SetParentOfSpecialL2S();
}

void Model::SetSpecialL2SMappingLikelihoodFixed(vector<int> label) {
    vector<vector<vector<float> > > fixed_mapping_prob;
    for (int i = 0; i < _config -> max_num_units(); ++i) {
        vector<vector<float> > fixed_mapping_prob_i;
        for (int j = 0; j <= i; ++j) {
            vector<float> fixed_mapping_prob_ij(_config -> cluster_num(), MIN_PROB_VALUE);
            fixed_mapping_prob_ij[0] = 0;
            fixed_mapping_prob_i.push_back(fixed_mapping_prob_ij);
        }
        fixed_mapping_prob.push_back(fixed_mapping_prob_i);
    }
    _l2s[label] -> set_mapping_likelihood(fixed_mapping_prob);
    _l2s[label] -> set_mapping_likelihood_fixed(true); 
    _l2s[label] -> set_collapsed(true);
    _l2s[label] -> set_parent(NULL);
}

void Model::SetParentOfSpecialL2S() {
    vector<int> sen_beginning(1, -10);
    vector<int> sen_ending(1, -20);
    vector<int> space(1, -3);
    _l2s[sen_beginning] -> set_parent(NULL);
    _l2s[sen_ending] -> set_parent(NULL);
    _l2s[space] -> set_parent(NULL);
}

void Model::AddSilenceCluster(Cluster* cluster) {
    _clusters.insert(_clusters.begin(), cluster);
}

void Model::SetSpecialL2S(vector<int> label) {
    vector<vector<vector<float> > > fixed_mapping_prob;
    for (int i = 0; i < _config -> max_num_units(); ++i) {
        vector<vector<float> > fixed_mapping_prob_i;
        for (int j = 0; j <= i; ++j) {
            vector<float> fixed_mapping_prob_ij(_config -> cluster_num(), MIN_PROB_VALUE);
            fixed_mapping_prob_ij[0] = 0;
            fixed_mapping_prob_i.push_back(fixed_mapping_prob_ij);
        }
        fixed_mapping_prob.push_back(fixed_mapping_prob_i);
    }
    L2S* special_l2s = new L2S(_config, label);
    special_l2s -> set_mapping_likelihood(fixed_mapping_prob);
    special_l2s -> set_mapping_likelihood_fixed(true); 
    special_l2s -> set_collapsed(true);
    special_l2s -> set_parent(NULL);
    _l2s[label] = special_l2s; 
}

void Model::Initialize() {
    for (int i = _clusters.size() ; i < _weak_limit; ++i) {
        Cluster* c = new Cluster(_config, i);
        _clusters.push_back(c);
    }
    vector<int> sen_beginning(1, -10);
    vector<int> sen_ending(1, -20);
    vector<int> space(1, -3);
    SetSpecialL2S(sen_beginning);
    SetSpecialL2S(sen_ending);
    SetSpecialL2S(space);

    vector<int> no_context_label(1, -100);
    L2S* no_context_l2s = new L2S(_config, no_context_label);
    _l2s[no_context_label] = no_context_l2s;
    for (int i = 0; i < _num_chars; ++i) {
        vector<int> label(1, i);
        L2S* l2s = new L2S(_config, label);
        _l2s[label] = l2s;
        l2s -> set_parent(no_context_l2s);
    }
}

void Model::Save(const string& path) {
    ofstream fout(path.c_str(), ios::binary);
    int num_clusters = _clusters.size(); 
    fout.write(reinterpret_cast<char*> (&num_clusters), sizeof(int));
    for (int i = 0; i < num_clusters; ++i) {
        _clusters[i] -> Save(fout);
    }
    int num_l2s = _l2s.size();
    fout.write(reinterpret_cast<char*> (&num_l2s), sizeof(int));
    map<vector<int>, L2S*>::iterator iter = _l2s.begin();
    for (; iter != _l2s.end(); ++iter) {
        iter -> second -> Save(fout);
    }
    fout.close();
}

void Model::LoadSnapshot(const string& fn_snapshot) {
    ifstream fsnapshot(fn_snapshot.c_str(), ios::binary);
    int num_clusters;
    fsnapshot.read(reinterpret_cast<char*> (&num_clusters), sizeof(int));
    cout << "Number of clusters: " << num_clusters << endl;
    for (int i = 0; i < num_clusters; ++i) {
        Cluster* c = new Cluster(_config);
        _clusters.push_back(c);
        c -> Load(fsnapshot);
    }
    int num_l2s;
    fsnapshot.read(reinterpret_cast<char*> (&num_l2s), sizeof(int));
    cout << "Number of L2S: " << num_l2s << endl;
    for (int i = 0; i < num_l2s; ++i) {
        int label_length;
        fsnapshot.read(reinterpret_cast<char*> (&label_length), \
                sizeof(int));
        vector<int> label(label_length, 0);
        fsnapshot.read(reinterpret_cast<char*> (&label[0]), \
                sizeof(int) * label_length);
        _l2s[label] = new L2S(_config, label);
        _l2s[label] -> Load(fsnapshot);
    }
    // cout << "Setting up parent relationship" << endl;
    // set parent relationship
    map<vector<int>, L2S*>::iterator l_iter = _l2s.begin();
    for (; l_iter != _l2s.end(); ++l_iter) {
        if ((l_iter -> first)[0] != -100) {
            vector<int> parent_label = l_iter -> second -> parent_label();
            if (_l2s.find(parent_label) == _l2s.end()) {
                cout << "Cannot find parent" << endl;
                exit(-1);
            }
            l_iter -> second -> set_parent(_l2s[parent_label]);
        }
    }
    SetParentOfSpecialL2S();
    fsnapshot.close();
}

void Model::PreCompute(float** _features, int frame_num) {
    vector<Cluster*>::iterator c_iter = _clusters.begin();
    for (; c_iter != _clusters.end(); ++c_iter) {
        (*c_iter) -> PreCompute(_features, frame_num);
    }
}

void Model::ShowL2S() {
    int counter = 0;
    map<vector<int>, L2S*>::iterator l_iter = _l2s.begin();
    for (; l_iter != _l2s.end(); ++l_iter) {
        cout << "Showing " << counter << "th L2S" << endl;
        ++counter;
        l_iter -> second -> print_label();
        cout << "Printing length prior" << endl;
        l_iter -> second -> print_length_prior();
        cout << "Printing mapping likelihood" << endl;
        l_iter -> second -> print_mapping_likelihood();
    }
}

Model::~Model() {
    vector<Cluster*>::iterator c_iter = _clusters.begin();
    for (; c_iter != _clusters.end(); ++c_iter) {
        delete *c_iter;
    }
    _clusters.clear();
    map<vector<int>, L2S*>::iterator l_iter = _l2s.begin();
    for (; l_iter != _l2s.end(); ++l_iter) {
        // l_iter -> second -> print_label();
        delete l_iter -> second;
    }
    _l2s.clear();
    // Language model should be deleted outside
}

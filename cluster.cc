#include <cstdlib>
#include <cstring>
#include <math.h>
#include "cluster.h"

#define DEBUG false 
#define SEG_DEBUG true
#define MIN_PROB_VALUE -70000000

using namespace std;

Cluster::Cluster(Config* config) {
    _config = config;
    _state_num = _config -> state_num();
    _is_fixed = false;
    _id = -1;
    for (int i = 0; i < _state_num; ++i) {
        GMM emission(_config);
        _emissions.push_back(emission);
        vector<float> trans_prob(_state_num + 1, 0);
        _transition_probs.push_back(trans_prob);
    }
}

Cluster::Cluster(Config* config, int state_num, int mix_num, int id) {
    _config = config;
    _state_num = state_num;
    _id = id;
    _is_fixed = false;
    for (int i = 0; i < _state_num; ++i) {
        GMM emission(_config, mix_num);
        _emissions.push_back(emission);
        vector<float> trans_prob(_state_num + 1, 0);
        _transition_probs.push_back(trans_prob);
    }
}

Cluster::Cluster(Config* config, int id) {
    _config = config;
    _state_num = _config -> state_num();
    _id = id;
    _is_fixed = false;
    for (int i = 0; i < _state_num; ++i) {
        GMM emission(_config);
        _emissions.push_back(emission);
        vector<float> trans_prob(_state_num + 1, 0);
        _transition_probs.push_back(trans_prob);
    }
}

Cluster::Cluster(Cluster& rhs) {
    _id = rhs.id();
    _state_num = rhs.state_num();
    _config = rhs.config();
    _transition_probs = rhs.transition_probs();
    _is_fixed = rhs.is_fixed();
    for (int i = 0; i < _state_num; ++i) {
        _emissions.push_back(rhs.emission(i));
    }
}

void Cluster::set_transition_probs(vector<vector<float> >& trans_prob) {
    _transition_probs = trans_prob;
}

void Cluster::set_emission(GMM& rhs, int index) {
    _emissions[index] = rhs;
}

void Cluster::set_emissions(vector<GMM>& rhs) {
    _emissions = rhs;
}

float** Cluster::ConstructSegProbTable(vector<Bound*>& bounds) {
    int b = bounds.size();
    int total_frame_num = 0;
    vector<int> accumulated_frame_nums(b, 0);
    int start_frame = bounds[0] -> start_frame();
    int end_frame = bounds[b - 1] -> end_frame();
    vector<float*> frames(end_frame - start_frame + 1, NULL); 
    for (int i = 0 ; i < b; ++i) {
        if (!(_config -> precompute())) {
            vector<float*> bound_data = bounds[i] -> data();
            memcpy(&frames[total_frame_num], &bound_data[0], \
                    (bounds[i] -> frame_num()) * sizeof(float*));
        }
        total_frame_num += bounds[i] -> frame_num();
        accumulated_frame_nums[i] = total_frame_num;
    }
    if (end_frame - start_frame + 1 != total_frame_num) {
        cerr << "Miss matched total frame number!" << endl;
        exit(1);
    }
    float** frame_prob_for_each_state;
    frame_prob_for_each_state = new float* [_state_num];
    for (int i = 0 ; i < _state_num; ++i) {
        frame_prob_for_each_state[i] = new float[total_frame_num];
        if (!(_config -> precompute())) {
            _emissions[i].ComputeLikehood(frames, frame_prob_for_each_state[i]);
            if (DEBUG) {
                for (int j = 0; j < total_frame_num; ++j) {
                    cout << frame_prob_for_each_state[i][j] << " ";
                }
                cout << endl;
            }
        }
        else {
            _emissions[i].ComputeLikehood(start_frame, end_frame, frame_prob_for_each_state[i]);
        }
    }
    float** prob_table;
    prob_table = new float* [b];
    for (int i = 0 ; i < b; ++i) {
        prob_table[i] = new float [b];
        for (int k = 0; k < b; ++k) {
            prob_table[i][k] = MIN_PROB_VALUE;
        }
    }
    int max_duration = _config -> max_duration();
    for (int i = 0 ; i < b; ++i) {
        if (bounds[i] -> is_labeled()) {
            if (bounds[i] -> label() == _id) {
                // Start computing probs by extending the vad result
                int start_ptr = i;
                while (start_ptr - 1 >= 0 && !(bounds[start_ptr - 1] -> is_labeled())) {
                    start_frame = start_ptr - 1 == 0 ? 0 : accumulated_frame_nums[start_ptr - 1];
                    if ((accumulated_frame_nums[i] - start_frame <= max_duration * 2) || start_ptr == i) {
                        --start_ptr;
                    }
                    else {
                        break;
                    }
                }
                int end_ptr = i;
                while (end_ptr + 1 < b && !(bounds[end_ptr + 1] -> is_labeled())) {
                    start_frame = i == 0 ? 0 : accumulated_frame_nums[i - 1];
                    if ((accumulated_frame_nums[end_ptr + 1] - start_frame <= max_duration * 2) || end_ptr == i) {
                        ++end_ptr;
                    } 
                    else {
                        break;
                    }
                }
                for (int b1 = start_ptr; b1 <= i; ++b1) {
                    start_frame = b1 == 0 ? 0 : accumulated_frame_nums[b1 - 1];
                    for (int b2 = i; b2 <= end_ptr; ++b2) {
                        end_frame = accumulated_frame_nums[b2] - 1;
                        vector<float> cur_prob(_state_num, MIN_PROB_VALUE);
                        for (int ptr = start_frame; ptr <= end_frame; ++ptr) {
                            if (ptr == start_frame) {
                                cur_prob[0] = frame_prob_for_each_state[0][ptr];
                            }
                            else {
                                vector<float> next_prob(_state_num, 0);
                                for (int k = 0; k < _state_num; ++k) {
                                    vector<float> summands;
                                    for (int l = 0; l <= k; ++l) {
                                        summands.push_back(cur_prob[l] + _transition_probs[l][k]);
                                    }
                                    next_prob[k] = ToolKit::SumLogs(summands) + frame_prob_for_each_state[k][ptr];
                                }
                                cur_prob = next_prob;
                            }
                        }
                        vector<float> next_prob;
                        for (int k = 0; k < _state_num; ++k) {
                            next_prob.push_back(cur_prob[k] + _transition_probs[k][_state_num]);
                        }
                        prob_table[b1][b2] = ToolKit::SumLogs(next_prob);
                    }
                }
            }
            else {
                prob_table[i][i] = MIN_PROB_VALUE; 
            }
        }
        else {
            int j = i;
            start_frame = i == 0 ? 0 : accumulated_frame_nums[i - 1];
            int duration = accumulated_frame_nums[i] - start_frame; 
            int ptr = start_frame;
            vector<float> cur_prob(_state_num, MIN_PROB_VALUE);
            while (ptr < start_frame + duration && j < b \
                    && (duration <= max_duration || (int) i == j) && !(bounds[j] -> is_labeled())) {
                if (ptr == start_frame) {
                    cur_prob[0] = frame_prob_for_each_state[0][ptr];
                }
                else {
                    vector<float> next_prob(_state_num, 0);
                    for (int k = 0; k < _state_num; ++k) {
                        vector<float> summands(k + 1, 0);
                        for (int l = 0; l <= k; ++l) {
                            summands[l] = cur_prob[l] + _transition_probs[l][k]; 
                        }
                        next_prob[k] = ToolKit::SumLogs(summands) + frame_prob_for_each_state[k][ptr];
                    }
                    cur_prob = next_prob;
                }
                if (ptr == accumulated_frame_nums[j] - 1) {
                    vector<float> next_prob(_state_num, 0);
                    for (int k = 0; k < _state_num; ++k) {
                        next_prob[k] = cur_prob[k] + _transition_probs[k][_state_num];
                    }
                    prob_table[i][j] = ToolKit::SumLogs(next_prob);
                    if (++j < b) {
                        duration = accumulated_frame_nums[j] - start_frame;
                    }
                }
                ++ptr;
            }
        }
    }
    for (int i = 0; i < b; ++i) {
        for (int j = 0; j < b; ++j) {
            if (isnan(prob_table[i][j])) {
               cerr << "Found NaN in  ConstructSegProbTable. Cluster id: " << _id << endl;
            }
        }
    }
    for (int i = 0 ; i < _state_num; ++i) {
        delete[] frame_prob_for_each_state[i];
    }
    delete [] frame_prob_for_each_state;
    return prob_table;
}

ProbList<int>** Cluster::MessageBackwardForASegment(Segment* segment) {
    int frame_num = segment -> frame_num();
    ProbList<int>** B;
    B = new ProbList<int>* [_state_num + 1];
    for (int i = 0 ; i <= _state_num; ++i) {
        B[i] = new ProbList<int> [frame_num + 1];
    }
    // Initialization [need to check what the initial value should be!]
    for (int i = 1; i <= _state_num; ++i) {
        B[i][frame_num].push_back(_transition_probs[i - 1][_state_num], -1);
    }
    // Message Backward
    for (int j = frame_num - 1; j > 0; --j) {
       float* data = segment -> frame(j);
       int data_index = segment -> frame_index(j);
       vector<float> emit_probs(_state_num);
       for (int k = 0; k < _state_num; ++k) {
           float emit_prob = _config -> precompute() ? \
                   _emissions[k].ComputeLikehood(data_index) : \
                   _emissions[k].ComputeLikehood(data);
           emit_probs[k] = emit_prob;
       }
       for (int i = 1; i <= _state_num; ++i) {
           for (int k = i; k <= _state_num; ++k) {
               B[i][j].push_back(_transition_probs[i - 1][k - 1] + \
                   emit_probs[k - 1] + B[k][j + 1].value(), k);
           }
       } 
    }
    float emit_prob = _config -> precompute() ? \
        _emissions[0].ComputeLikehood(segment -> frame_index(0)) : \
        _emissions[0].ComputeLikehood(segment -> frame(0));
    B[0][0].push_back(emit_prob + B[1][1].value(), 1); 
    return B;
}

float Cluster::ComputeSegmentProb(Segment* segment) {
    int frame_num = segment -> frame_num();
    vector<float> prob_s(_state_num, 0);
    for (int i = 0; i < _state_num; ++i) {
        prob_s[i] = _transition_probs[i][_state_num];
    }
    for (int j = frame_num - 2; j >= 0; --j) {
        float* data = segment -> frame(j + 1);
        float data_index = segment -> frame_index(j + 1);
        vector<float> emit_probs(_state_num, 0);
        for (int k = 0; k < _state_num; ++k) {
            float emit_prob = _config -> precompute() ? \
                _emissions[k].ComputeLikehood(data_index) : \
                _emissions[k].ComputeLikehood(data);
            emit_probs[k] = emit_prob;
        }
        vector<float> prob_s_1;
        for (int s_1 = 0; s_1 < _state_num; ++s_1) {
            vector<float> to_sum;
            for (int s = s_1; s < _state_num; ++s) {
                to_sum.push_back(_transition_probs[s_1][s] + \
                        emit_probs[s] + prob_s[s]);
            }
            prob_s_1.push_back(ToolKit::SumLogs(to_sum));
        }
        prob_s = prob_s_1;
    }
    if (_config -> precompute()) {
        return prob_s[0] + _emissions[0].ComputeLikehood(segment -> frame_index(0));
    }
    else {
        return prob_s[0] + _emissions[0].ComputeLikehood(segment -> frame(0));
    }
}

void Cluster::Plus(Segment* segment) {
    if (_is_fixed) {
        return;
    }
    const vector<int> state_seq = segment -> state_seq(); 
    const vector<int> mix_seq = segment -> mix_seq();
    const vector<float*> data = segment -> data();
    if (state_seq.size() != data.size()) {
        cout << "In ClusterCounter::Plus, state_seq and data have different sizes." << endl;
        exit(2);
    }
    else if (mix_seq.size() != data.size()) {
        cout << "In ClusterCounter::Plus, mix_seq and data have different sizes." << endl;
        exit(2);
    }
    else {
        for (size_t i= 0 ; i < state_seq.size(); ++i) {
            int cur_state = state_seq[i];
            int next_state = i == state_seq.size() - 1 ? \
                             _state_num : state_seq[i + 1];
            ++_transition_probs[cur_state][next_state];
            _emissions[cur_state].Plus(data[i], mix_seq[i]);
        }
    }
}

void Cluster::Minus(Segment* segment) {
    if (_is_fixed) {
        return;
    }
    vector<int> state_seq = segment -> state_seq();
    vector<int> mix_seq = segment -> mix_seq();
    vector<float*> data = segment -> data();
    if (state_seq.size() != data.size()) {
        cout << "In ClusterCounter::Minus, state_seq and data have different sizes." << endl;
        exit(2);
    }
    else if (mix_seq.size() != data.size()) {
        cout << "In ClusterCounter::Minus, mix_seq and data have different sizes." << endl;
        exit(2);
    }
    else {
        for (size_t i = 0; i < state_seq.size(); ++i) {
            int cur_state = state_seq[i];
            int next_state = i == state_seq.size() - 1 ? \
                             _state_num : state_seq[i + 1];
            --_transition_probs[cur_state][next_state];
            _emissions[cur_state].Minus(data[i], mix_seq[i]);
        }
    }
}

Cluster& Cluster::operator+= (Cluster& rhs) {
    vector<vector<float> > rhs_transition_probs = rhs.transition_probs();
    for (int i = 0 ; i < _state_num; ++i) {
        for (int j = 0; j <= _state_num; ++j) {
            _transition_probs[i][j] += rhs_transition_probs[i][j];
        }
    }
    for (int i = 0; i < _state_num; ++i) {
        _emissions[i] += rhs.emission(i);
    }
    return *this;
}

void Cluster::Save(ofstream& fout) {
    fout.write(reinterpret_cast<char*> (&_id), sizeof(int));
    int fixed = _is_fixed ? 1 : 0;
    fout.write(reinterpret_cast<char*> (&fixed), sizeof(int));
    fout.write(reinterpret_cast<char*> (&_state_num), sizeof(int));
    for (int i = 0; i < _state_num; ++i) {
        fout.write(reinterpret_cast<char*> (&_transition_probs[i][0]), sizeof(float) * (_state_num + 1));
    }
    for (int i = 0 ; i < _state_num; ++i) {
        _emissions[i].Save(fout);
    }
}

void Cluster::Load(ifstream& fin) {
    fin.read(reinterpret_cast<char*> (&_id), sizeof(int));
    int fixed;
    fin.read(reinterpret_cast<char*> (&fixed), sizeof(int));
    _is_fixed = fixed == 1 ? true : false;
    fin.read(reinterpret_cast<char*> (&_state_num), sizeof(int));
    // Initialize space for _transition_probs and _emissions
    for (int i = 0; i < _state_num; ++i) {
        GMM emission(_config);
        _emissions.push_back(emission);
        vector<float> trans_prob(_state_num + 1, 0);
        _transition_probs.push_back(trans_prob);
    }
    for (int i = 0; i < _state_num; ++i) {
        fin.read(reinterpret_cast<char*> (&_transition_probs[i][0]), \
                sizeof(float) * (_state_num + 1));
        for (int j = 0; j <= _state_num; ++j) {
        }
    }
    for (int i = 0; i < _state_num; ++i) {
        _emissions[i].Load(fin);
    }
}

void Cluster::PreCompute(float** data, int frame_num) {
    for (int i = 0; i < _state_num; ++i) {
        _emissions[i].PreCompute(data, frame_num);
    }
}

void Cluster::Reset() {
    for (size_t i = 0; i < _emissions.size(); ++i) {
        _emissions[i].Reset();
    }
}

Cluster::~Cluster() {
}

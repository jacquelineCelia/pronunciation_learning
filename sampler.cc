#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <math.h>
#include <mkl_vml.h>
#include <queue>
#include <cstring>

#include "sampler.h"
#include "letter.h"
#include "bound.h"

#define BRNG VSL_BRNG_MT19937 
#define GAMMA_METHOD VSL_RNG_METHOD_GAMMA_GNORM
#define UNIFORM_METHOD VSL_RNG_METHOD_UNIFORM_STD
#define GAUSSIAN_METHOD VSL_RNG_METHOD_GAUSSIAN_ICDF 

#define MIN_PROB_VALUE -70000000

#define DEBUG false 
#define SEG_DEBUG false 

Sampler::Sampler(Config* config) {
    _config = config;
    _n_context = config -> n_context();
    _max_num_epsilons = config -> max_num_epsilons();    
    unsigned int SEED = time(0);
    vslNewStream(&stream, BRNG,  SEED);
}

void Sampler::Initialize(Config* config) {
    _config = config;
    _n_context = config -> n_context();
    _max_num_epsilons = config -> max_num_epsilons();    
}

void Sampler::SampleModelParams(Model* model, Counter* counter) {
    // Sample Cluster parameters
    vector<Cluster*> clusters = model -> clusters();
    map<vector<int>, L2S*> l2s = model -> l2s();
    if (counter != NULL) {
        vector<Cluster*> cluster_counter = counter -> clusters();
        map<vector<int>, L2S*> l2s_counter = counter -> l2s();
        if (clusters.size() != cluster_counter.size()) {
            cout << "Clusters don't have the right number of ClusterCounters" << endl;
            exit(4);
        }
        else {
            for (int  i = 0 ; i < (int) clusters.size(); ++i) {
                SampleClusterParams(clusters[i], cluster_counter[i]);
            }
        }
        // Sample new L2S parameters
        map<vector<int>, L2S*>::iterator l_counter_iter = l2s_counter.begin();
        for (; l_counter_iter != l2s_counter.end(); ++l_counter_iter) {
            if (l2s.find(l_counter_iter -> first) == l2s.end()) {
                L2S* new_l2s = new L2S (_config, l_counter_iter -> first);
                (model -> l2s())[l_counter_iter -> first] = new_l2s;
            }
        }
        l2s = model -> l2s();
        map<vector<int>, L2S*>::iterator l_iter = l2s.begin();
        for (; l_iter != l2s.end(); ++l_iter) {
            vector<int> label = l_iter -> first;
            if (!(label.size() == 1 && label[0] < 0)) {
                vector<int> parent_label = l_iter -> second -> parent_label();
                l_iter -> second -> set_parent(l2s[parent_label]);
            }
        }
        vector<int> sen_beginning(1, -10);
        if (l2s[sen_beginning] -> is_collapsed()) {
            CopyL2SCounts(l2s[sen_beginning], l2s_counter[sen_beginning]);
        }
        else {
            SampleL2SParams(l2s[sen_beginning], l2s_counter[sen_beginning]);
        }
        vector<int> sen_ending(1, -20);
        if (l2s[sen_ending] -> is_collapsed()) {
            CopyL2SCounts(l2s[sen_ending], l2s_counter[sen_ending]);
        }
        else {
            SampleL2SParams(l2s[sen_ending], l2s_counter[sen_ending]);
        }
        vector<int> space(1, -3);
        if (l2s[space] -> is_collapsed()) {
            CopyL2SCounts(l2s[space], l2s_counter[space]);
        }
        else {
            SampleL2SParams(l2s[space], l2s_counter[space]);
        }
        vector<int> label(1, -100);
        if (l2s[label] -> is_collapsed()) {
            CopyL2SCounts(l2s[label], l2s_counter[label]);
        }
        else {
            SampleL2SParams(l2s[label], l2s_counter[label]);
        }
        l_iter = l2s.begin();
        for (; l_iter != l2s.end(); ++l_iter) {
            if (!(l_iter -> second -> updated())) {
                vector<L2S*> to_be_processed;
                to_be_processed.push_back(l_iter -> second);
                vector<int> parent_label = l_iter -> second -> parent_label();
                while (!(l2s[parent_label] -> updated())) {
                    to_be_processed.push_back(l2s[parent_label]);
                    parent_label = l2s[parent_label] -> parent_label(); 
                }
                for (int p = to_be_processed.size() - 1; p >= 0; --p) {
                    if (!(to_be_processed[p] -> is_collapsed())) {
                        if (l2s_counter.find(to_be_processed[p] -> label()) == l2s_counter.end()) {
                            L2S pseudo_counter(_config, to_be_processed[p] -> label());
                            SampleL2SParams(to_be_processed[p], &pseudo_counter);
                        }
                        else {
                            SampleL2SParams(to_be_processed[p], l2s_counter[to_be_processed[p] -> label()]);
                        }
                    }
                    else {
                        if (l2s_counter.find(to_be_processed[p] -> label()) == l2s_counter.end()) {
                            L2S pseudo_counter(_config, to_be_processed[p] -> label());
                            CopyL2SCounts(to_be_processed[p], &pseudo_counter);
                        }
                        else {
                            CopyL2SCounts(to_be_processed[p], l2s_counter[to_be_processed[p] -> label()]);
                        }
                    }
                }
            }
        }
        l_iter = l2s.begin();
        for (; l_iter != l2s.end(); ++l_iter) {
            l_iter -> second -> set_updated(false);
        }
    }
    else {
        for (int i = 0; i < (int) clusters.size(); ++i) {
            SampleClusterParams(clusters[i]);
        }
        vector<int> sen_beginning(1, -10);
        if (!(l2s[sen_beginning] -> is_collapsed())) {
            SampleL2SParams(l2s[sen_beginning]);
        }
        vector<int> sen_ending(1, -20);
        if (!(l2s[sen_ending] -> is_collapsed())) {
            SampleL2SParams(l2s[sen_ending]);
        }
        vector<int> space(1, -3);
        if (!(l2s[space] -> is_collapsed())) {
            SampleL2SParams(l2s[space]);
        }
        vector<int> label(1, -100);
        if (!(l2s[label] -> is_collapsed())) {
            SampleL2SParams(l2s[label]);
        }
        map<vector<int>, L2S*>::iterator l_iter = l2s.begin();
        for (; l_iter != l2s.end(); ++l_iter) {
            if (!(l_iter -> second -> updated())) {
                if (!(l_iter -> second -> is_collapsed())) {
                    SampleL2SParams(l_iter -> second);
                }
            }
        }
        l_iter = l2s.begin();
        for (; l_iter != l2s.end(); ++l_iter) {
            l_iter -> second -> set_updated(false);
        }
    }
}

void Sampler::CopyL2SCounts(L2S* destination, L2S* source) {
    if (!(destination -> is_length_prior_fixed())) {
        vector<float> a = source -> length_prior();
        destination -> set_length_prior(a);
    } 
    if (!(destination -> is_mapping_likelihood_fixed())) {
        vector<vector<vector<float> > > a = source -> mapping_likelihood();
        destination -> set_mapping_likelihood(a);
    } 
    destination -> set_N(source -> N());
    destination -> set_updated(true);
}

void Sampler::SampleL2SParams(L2S* l2s, L2S* l2s_counter) {
    if (l2s_counter == NULL) {
        int num_clusters = _config -> cluster_num();
        // Sample 1) length_prior 
        if (!(l2s -> is_length_prior_fixed())) {
            vector<float> length_alpha = l2s -> length_alpha();
            vector<float> length_prior = SampleDirFromGamma(_config -> max_num_units() + 1, \
                                                 &length_alpha[0], -10); 
            l2s -> set_length_prior(length_prior);
        }
        // Sample 2) mapping_likelihood [make it uniform at the beginning] 
        if (!(l2s -> is_mapping_likelihood_fixed())) {
            vector<vector<vector<float> > > mapping_likelihood;
            for (int i = 0; i < _config -> max_num_units(); ++i) {
                vector<vector<float> > mapping_likelihood_i;
                vector<float> mapping_alpha(num_clusters, l2s -> mapping_alpha()[i]);
                for (int j = 0; j <= i; ++j) {
                    vector<float> mapping_likelihood_ij = SampleDirFromGamma(num_clusters, &mapping_alpha[0], -10);
                    mapping_likelihood_i.push_back(mapping_likelihood_ij);
                }
                mapping_likelihood.push_back(mapping_likelihood_i);
            } 
            l2s -> set_mapping_likelihood(mapping_likelihood);
        }
    }
    else {
        // Sample 1) length_prior
        if (!(l2s -> is_length_prior_fixed())) {
            vector<float> length_alpha = l2s_counter -> length_prior();
            for (int i = 0; i < _config -> max_num_units() + 1; ++i) {
                length_alpha[i] += (l2s -> length_alpha())[i];
            } 
            vector<float> length_prior = SampleDirFromGamma(_config -> max_num_units() + 1, &length_alpha[0], -10);
            l2s -> set_length_prior(length_prior);
        }
        // Sample 2) mapping_likelihood
        if (!(l2s -> is_mapping_likelihood_fixed())) {
            vector<vector<vector<float> > > mapping_likelihood;
            for (int i = 0; i < _config -> max_num_units(); ++i) {
                vector<vector<float> > mapping_likelihood_i;
                float mapping_alpha = log(l2s -> mapping_alpha()[i]);
                for (int j = 0; j <= i; ++j) {
                    vector<float> parent_prior;
                    if (l2s -> parent() == NULL) {
                        parent_prior.resize(_config -> cluster_num(), (l2s -> mapping_alpha())[i]);
                        vector<float> data_counts = l2s_counter -> mapping_likelihood(i, j);
                        for (int k = 0; k < _config -> cluster_num(); ++k) {
                            parent_prior[k] += data_counts[k];
                        }
                    }
                    else {
                        // This only supports the case that parents are non-collapsed.
                        // Otherwise, parent_prior will be counts instead of probabilities.
                        parent_prior = l2s -> parent() -> mapping_likelihood(i, j);
                        vector<float> data_counts = l2s_counter -> mapping_likelihood(i, j);
                        for (int k = 0; k < _config -> cluster_num(); ++k) {
                            float value = exp(mapping_alpha + parent_prior[k]);
                            parent_prior[k] = value > 0 ? value + data_counts[k] : 0.00001 + data_counts[k];
                        }
                    } 
                    vector<float> mapping_likelihood_ij = SampleDirFromGamma(_config -> cluster_num(), &parent_prior[0], -10);
                    mapping_likelihood_i.push_back(mapping_likelihood_ij);
                }
                mapping_likelihood.push_back(mapping_likelihood_i);
            }
            l2s -> set_mapping_likelihood(mapping_likelihood);
        }
    }
    l2s -> set_updated(true);
}

vector<float> Sampler::SampleDirFromGamma(int n, float* alpha, float min) {
    vector<float> samples(n, 0);
    bool need_to_set_sparse = true;
    for (int i = 0; i < n; ++i) {
        if (vsRngGamma(GAMMA_METHOD, stream, 1, &samples[i], alpha[i], 0, 1) != VSL_STATUS_OK) {
           cout << "Error when calling SampleDirFromGamma" << endl;
           cout << "The parameters are:" << endl;
           cout << "an: " << alpha[i] << " bn: 1" << endl; 
           exit(1);
        }
        samples[i] = samples[i] == 0 ? min : log(samples[i]);
        if (samples[i] < min) {
            samples[i] = min;
        }
        if (samples[i] > min) {
            need_to_set_sparse = false;
        }
    }
    if (need_to_set_sparse) {
        samples[0] = 0;
    }
    float sum = ToolKit::SumLogs(samples); 
    for (int i = 0; i < n; ++i) {
        samples[i] -= sum; 
    }
    return samples;
}

void Sampler::SampleClusterParams(Cluster* cluster, Cluster* counter) {
    if (cluster -> is_fixed()) {
        return;
    }
    int state_num_ = _config -> state_num();
    int mix_num_ = _config -> mix_num();
    vector<vector<float> > trans_probs;
    if (counter == NULL) {
        for (int i = 0; i < state_num_; ++i) {
            vector<float> alpha(state_num_ + 1 - i, _config -> transition_alpha());
            vector<float> trans_prob = SampleDirFromGamma(state_num_ + 1 - i, &alpha[0]);
            for (int j = 0; j < i; ++j) {
                trans_prob.insert(trans_prob.begin(), MIN_PROB_VALUE);
            }
            trans_probs.push_back(trans_prob);
        }  
        cluster -> set_transition_probs(trans_probs);
        // Sample each GMM from prior (mixture weight and Gaussian)
        for (int i = 0; i < state_num_; ++i) {
            vector<float> alpha(mix_num_, _config -> mix_alpha());
            vector<float> mix_weight = SampleDirFromGamma(mix_num_, &alpha[0], -5); 
            for (int j = 0; j < mix_num_; ++j) {
                vector<float> mean_count(_config -> dim(), 0);
                vector<float> pre_count(_config -> dim(), 0);
                vector<float> pre = SampleGaussianPre(mean_count, pre_count, 0, cluster -> id(), false);
                vector<float> mean = SampleGaussianMean(pre, mean_count, 0, cluster -> id(), false);
                (cluster -> emission(i)).mixture(j).set_mean(mean);
                (cluster -> emission(i)).mixture(j).set_pre(pre);
                (cluster -> emission(i)).mixture(j).set_det();
            }
            (cluster -> emission(i)).set_weight(mix_weight);
        }
    }
    else {
        vector<vector<float> > trans_probs;
        vector<vector<float> > trans_counts = counter -> transition_probs();
        for (int i = 0; i < state_num_; ++i) {
            for (int j = i; j < state_num_ + 1; ++j) {
                trans_counts[i][j] += _config -> transition_alpha();
            }
            vector<float> trans_prob = SampleDirFromGamma(state_num_ + 1 - i, &trans_counts[i][i]);
            for (int j = 0; j < i; ++j) {
                trans_prob.insert(trans_prob.begin(), MIN_PROB_VALUE);
            }
            trans_probs.push_back(trans_prob);
        }
        cluster -> set_transition_probs(trans_probs);
        for (int i = 0; i < state_num_; ++i) {
            vector<float> weight_count = (counter -> emission(i)).weight();
            for (int j = 0; j < mix_num_; ++j) {
                // Sample Gaussian
                vector<float> mean_count = (counter -> emission(i)).mixture(j).mean();
                vector<float> pre_count = (counter -> emission(i)).mixture(j).pre();
                vector<float> pre = SampleGaussianPre(mean_count, pre_count, weight_count[j], cluster -> id(), false);
                vector<float> mean = SampleGaussianMean(pre, mean_count, weight_count[j], cluster -> id(), false);
                (cluster -> emission(i)).mixture(j).set_mean(mean);
                (cluster -> emission(i)).mixture(j).set_pre(pre);
                (cluster -> emission(i)).mixture(j).set_det();
                weight_count[j] += _config -> mix_alpha();
            }
            vector<float> weight = SampleDirFromGamma(mix_num_, &weight_count[0], -5);
            (cluster -> emission(i)).set_weight(weight);
        }
    }
}

float Sampler::UpdateGammaRate(float b0, float x2, float x, float n, float u0) {
    float k0 = _config -> gaussian_k0();
    if (n == 0) {
        return b0;
    }
    else {
        return b0 + 0.5 * (x2 - x * x / n) + (k0 * n * (x / n - u0) * (x / n - u0))/(2 * (k0 + n));
    }
}

vector<float> Sampler::SampleGaussianPre(vector<float> mean_count, \
        vector<float> pre_count, float n, int id, bool strong_seeding) {
    vector<float> gaussian_b0;
    vector<float> gaussian_u0;
    vector<float> new_pre(_config -> dim(), 1);
    gaussian_b0 = _config -> gaussian_b0();
    gaussian_u0 = _config -> gaussian_u0();    
    for (int i = 0; i < _config -> dim(); ++i) {
        float bn = UpdateGammaRate(\
                gaussian_b0[i], pre_count[i], mean_count[i], \
                n, gaussian_u0[i]);
        float an = _config -> gaussian_a0() + n / 2;
        if (vsRngGamma(GAMMA_METHOD, stream, 1, &new_pre[i], an, 0, 1 / bn) != VSL_STATUS_OK) {
            cout << "Error when calling SampleGaussianPre" << endl;
            cout << "The parameters are: " << endl;
            cout << "an: " << an << " bn: " << bn << endl;
            exit(1);
        } 
    }
    return new_pre;
}

vector<float> Sampler::SampleGaussianMean(vector<float> pre, \
        vector<float> count, float n, int id, bool strong_seeding) {
    vector<float> new_mean(_config -> dim(), 0);
    vector<float> gaussian_u0;
    float k0 = _config -> gaussian_k0();
    gaussian_u0 = _config -> gaussian_u0();
    for (int i = 0; i < _config -> dim(); ++i) {
        float un = (k0 * gaussian_u0[i] + count[i]) / (k0 + n);
        float kn = k0 + n;
        float std = sqrt(1 / (kn * pre[i]));
        vsRngGaussian(GAUSSIAN_METHOD, stream, 1, &new_mean[i], un, std); 
    }
    return new_mean;
}

void Sampler::SampleSegments(Datum* datum, \
                     vector<Cluster*>& cluster, \
                     map<vector<int>, L2S*>& l2s, \
                     vector<Cluster*>& cluster_counter, \
                     map<vector<int>, L2S*>& l2s_counter) {
    if (!((datum -> segments()).size()) && !((datum -> letters()).size())) {
        int i = 0;
        vector<Bound*> bounds = datum -> bounds();
        int max_num_bounds = (_config -> boundary_distribution()).size();
        while (i < (int) bounds.size()) {
            int n;
            if (bounds[i] -> is_boundary()) {
                n = 1;
            }
            else {
                // find the next boundary
                int j = i;
                while (!(bounds[j] -> is_boundary()) && j - i + 1 < max_num_bounds) {
                    ++j;
                }
                int num_bounds_considered = j - i + 1;
                vector<float> boundary_distribution((_config -> boundary_distribution()).begin(), \
                        (_config -> boundary_distribution()).begin() + num_bounds_considered);
                n = SampleIndexFromDistribution(boundary_distribution) + 1; 
            }
            Segment* segment = new Segment(_config, -1);
            for (int j = i; j < i + n && j < (int) bounds.size(); ++j) {
                segment -> push_back(bounds[j]);
            }
            int id;
            if (bounds[i] -> is_labeled()) {
                id = bounds[i] -> label();
            }
            else {
                vector<float> likelihood;
                for (int j = 0; j < (int) cluster.size(); ++j) {
                    // MKL
                    likelihood.push_back(cluster[j] -> ComputeSegmentProb(segment));
                }
                id = SampleIndexFromLogDistribution(likelihood);
            }
            segment -> set_id(id);
            SampleStateMixtureSeq(segment, cluster[id]);
            cluster_counter[id] -> Plus(segment);
            (datum -> segments()).push_back(segment);
            i += n;
        } 
    }
    else {
        if (SEG_DEBUG) {
            cout << "sampling from post" << endl;
        }
        int l = (datum -> letters()).size();
        int b = (datum -> bounds()).size();
        // Remove previous alignment results
        if (SEG_DEBUG) {
            cout << "Removing L2S" << endl;
        }
        RemoveL2SAlignment(datum, l2s_counter, l2s);
        if (SEG_DEBUG) {
            cout << "Removing Cluster Alignment" << endl;
        }
        RemoveClusterAssignment(datum, cluster_counter);
        // Compute P(b|c) 
        if (SEG_DEBUG) {
            cout << "Computing P(b|c)" << endl;
        }
        float*** segment_prob_given_cluster = new float** [_config -> weak_limit()];
        ComputeSegProbGivenCluster(datum, cluster, segment_prob_given_cluster);
        // Compute p(b|l)
        if (SEG_DEBUG) {
            cout << "Computing P(b|l)" << endl;
        }
        map<vector<int>, ProbList<int>****> segment_prob_individual_cluster; 
        map<vector<int>, ProbList<int>**> segment_prob;
        ComputeSegProbGivenLetter(datum, l2s, segment_prob_given_cluster, segment_prob, segment_prob_individual_cluster);
        // Compute B and Bstar
        if (SEG_DEBUG) {
            cout << "Allocating space for B and Bstar" << endl;
        }
        ProbList<int> **Bstar, **B; 
        B = new ProbList<int>* [l + 1];
        Bstar = new ProbList<int>* [l + 1];
        for (int i = 0 ; i <= l; ++i) {
            B[i] = new ProbList<int> [b + 1];
            Bstar[i] = new ProbList<int> [b + 1];
        }
        if (SEG_DEBUG) {
            cout << "Message Backward" << endl;
        }
        MessageBackward(datum, segment_prob, l2s, B, Bstar);
        // Sample forward
        if (SEG_DEBUG) {
            cout << "Sample Forward" << endl;
        }
        SampleForward(datum, segment_prob, segment_prob_individual_cluster, B, Bstar);
        // Sample State sequence and Mixture ID for segments
        if (SEG_DEBUG) {
            cout << "Sampling state and mixture seq" << endl;
        }
        vector<Segment*> segments = datum -> segments();
        for (int i = 0; i < (int) segments.size(); ++i) {
            if (SEG_DEBUG) {
                cout << "Sampling: " << i << endl;
            }
            SampleStateMixtureSeq(segments[i], cluster[segments[i] -> id()]);
        }
        // Add newly sampled alignment results
        if (SEG_DEBUG) {
            cout << "Adding cluster assignement" << endl;
        }
        AddClusterAssignment(datum, cluster_counter);
        // Delete allocated memory: B, Bstar, ProbList, segment_prob_given_cluster
        if (SEG_DEBUG) {
            cout << "deleting B and Bstar" << endl;
        }
        for (int i = 0; i <= l; ++i) {
            delete[] B[i];
            delete[] Bstar[i];
        }
        delete[] B;
        delete[] Bstar;
        if (SEG_DEBUG) {
            cout << "deleting segment_prob" << endl;
        }
        map<vector<int>, ProbList<int>**>::iterator p_iter = segment_prob.begin();
        for (; p_iter != segment_prob.end(); ++p_iter) {
            for (int j = 0; j < b; ++j) {
                delete[] (p_iter -> second)[j]; 
            }
            delete[] (p_iter -> second); 
        }
        if (SEG_DEBUG) {
            cout << "deleting segment_prob_individual" << endl;
        }
        map<vector<int>, ProbList<int>****>::iterator q_iter = \
            segment_prob_individual_cluster.begin();
        for (; q_iter != segment_prob_individual_cluster.end(); ++q_iter) {
            ProbList<int>**** ptr = q_iter -> second; 
            for (int u = 0; u < _config -> max_num_units(); ++u) {
                for (int v = 0; v <= u; ++v) {
                    for (int w = 0; w < b; ++w) {
                        delete[] ptr[u][v][w];
                    }
                    delete[] ptr[u][v];
                }
                delete[] ptr[u];
            }
            delete[] ptr;
        }
        if (SEG_DEBUG) {
            cout << "deleting segment_prob_given_cluster" << endl;
        }
        for (int i = 0; i < _config -> weak_limit(); ++i) {
            for (int j = 0; j < b; ++j) {
                delete[] segment_prob_given_cluster[i][j];
            }
            delete[] segment_prob_given_cluster[i]; 
        }
        delete[] segment_prob_given_cluster;
    }
    if (SEG_DEBUG) {
        cout << "Done sampling segments" << endl;
    }
    AddL2SAlignment(datum, l2s_counter, l2s);
}

void Sampler::SampleStateMixtureSeq(Segment* segment, Cluster* cluster) {
    if (cluster -> is_fixed()) {
        return;
    }
    // Message Backward
    ProbList<int> **B = cluster -> MessageBackwardForASegment(segment);
    // Sample Forward
    vector<int> state_seq;
    vector<int> mix_seq;
    int s;
    int m;
    for (int i = 0, j = 0; j < segment -> frame_num(); ++j) {
        s = B[i][j].index(SampleIndexFromLogDistribution(B[i][j].probs()));
        state_seq.push_back(s - 1);
        if (!(_config -> precompute())) {
            m = SampleIndexFromLogDistribution((cluster -> emission(s - 1)).\
                ComponentLikelihood(segment -> frame(j)));
        }
        else {
            m = SampleIndexFromLogDistribution((cluster -> emission(s - 1)).\
                ComponentLikelihood(segment -> frame_index(j)));
        }
        mix_seq.push_back(m);
        i = s;
    }
    segment -> set_state_seq(state_seq);
    segment -> set_mix_seq(mix_seq);
    // Delete B
    for (int i = 0; i <= cluster -> state_num(); ++i) {
        delete[] B[i];
    }
    delete[] B;
}

int Sampler::SampleIndexFromLogDistribution(vector<float> log_probs) {
    ToolKit::MaxRemovedLogDist(log_probs);
    return SampleIndexFromDistribution(log_probs);
}

int Sampler::SampleIndexFromDistribution(vector<float> probs) {
   float sum = ToolKit::NormalizeDist(probs);
   // sample a random number from a uniform dist [0,1]
   float random_unit_sample = rvg.GetUniformSample(); 
   while (random_unit_sample == 0 || random_unit_sample == 1) {
       random_unit_sample = rvg.GetUniformSample(); 
   }
   // figure out the index 
   size_t index = 0; 
   sum = probs[index];
   while (random_unit_sample > sum) {
       if (++index < probs.size()) {
           sum += probs[index];
       }
       else {
           break;
       }
   }
   if (index >= probs.size()) {
       index = probs.size() - 1;
   }
   return index;
}

void Sampler::ComputeSegProbGivenCluster(Datum* datum, \
                            vector<Cluster*>& clusters, \
                            float*** segment_prob_given_cluster) {
    for (int i = 0; i < (int) clusters.size(); ++i) {
        if (DEBUG) {
            cout << "cluster " << i << endl;
        }
        segment_prob_given_cluster[i] = \
            clusters[i] -> ConstructSegProbTable(datum -> bounds());
        if (SEG_DEBUG) {
            cout << "For Cluster: " << i << endl;
            int b = (datum -> bounds()).size();
            for (int j = 0; j < b; ++j) {
                for (int k = 0; k < b; ++k) {
                   cout << "b[" << j << "][" << k << "]: " << segment_prob_given_cluster[i][j][k] << " "; 
                }
                cout << endl;
            }
            cout << endl;
        }
    }
}

void Sampler::ComputeSegProbGivenLetter(Datum* datum, \
                     map<vector<int>, L2S*>& l2s, \
                     float*** segment_prob_given_cluster, \
                     map<vector<int>, ProbList<int>**>& segment_prob, \
                     map<vector<int>, ProbList<int>****>& segment_prob_individual_cluster) {
    for (int i = 0; i < (int) (datum -> letters()).size(); ++i) {
        vector<int> context = datum -> Context(i);
        if (segment_prob.find(context) == segment_prob.end()) {
            if (l2s.find(context) == l2s.end()) {
                EstablishRootInL2S(context, l2s);
            }
            segment_prob_individual_cluster[context] = \
                l2s[context] -> ConstructIndividualSegProbTable(datum -> bounds(), segment_prob_given_cluster); 
            segment_prob[context] = l2s[context] -> \
             ConstructSegProbTable(datum -> bounds(), segment_prob_individual_cluster[context]);
        }
    }
}

void Sampler::SampleForward(Datum* datum, \
            map<vector<int>, ProbList<int>**>& segment_prob, \
            map<vector<int>, ProbList<int>****>& segment_prob_individual_cluster, \
            ProbList<int>** B, ProbList<int>** Bstar) {
    vector<Bound*> bounds = datum -> bounds();
    int l = (datum -> letters()).size();
    int b = bounds.size();
    int i = 0, j = 0;
    while (i < l && j < b) {
        // Sample from B to decide which letter to go to
        if (SEG_DEBUG) {
            vector<float> next_i_dist = B[i][j].probs();
            for (int gg = 0; gg < (int) next_i_dist.size(); ++gg) {
                cout << "prob[" << gg + i << "]: " << next_i_dist[gg] << " ";
            }
            cout << endl;
        }
        int next_i = B[i][j].index(\
                SampleIndexFromLogDistribution(B[i][j].probs()));
        if (DEBUG) {
            vector<float> dist_next_i = B[i][j].probs();
            cout << "B[" << i << "][" << j << "]: " << endl;
            for (unsigned int z = 0; z < dist_next_i.size(); ++z) {
                cout << z << ": " << dist_next_i[z] << endl;
            }
        }
        // Sample from Bstar to decide which bound to go to
        if (DEBUG) {
            cout << "Sampling from Bstar to decide which bound to go to" << endl;
            vector<float> possible_next_j = Bstar[next_i][j].probs();
            for (int gg = 0; gg < (int) possible_next_j.size(); ++gg) {
                cout << "prob[" << gg + j << "]: " << possible_next_j[gg] << " ";
        }
        cout << endl;
        }
        int next_j = Bstar[next_i][j].index(\
                SampleIndexFromLogDistribution(Bstar[next_i][j].probs())); 
        if (DEBUG) {
            cout << "next_j: " << next_j << endl;
            cout << "Done sampling next bound" << endl;
            vector<float> next_dist = Bstar[next_i][j].probs();
            for (unsigned int z = 0; z < next_dist.size(); ++z) {
                cout << "z: " <<  z << " " << next_dist[z] << ',';
            }
            cout << endl;
        }
        vector<int> context = datum -> Context(next_i - 1);
        int sampled_index = SampleIndexFromLogDistribution(segment_prob[context][j][next_j - 1].probs());
        int boundary_id = segment_prob[context][j][next_j - 1].index(sampled_index);
        if (DEBUG) { 
            cout << "next_i: " << next_i << endl;
            cout << "next_j: " << next_j << endl;
            cout << "boundary_id: " << boundary_id << endl;
        }
        // Done sampling (except segment state_seq) add information
        if (next_i > i + 1) {
            for (int k = i + 1; k < next_i; ++k) {
                vector<int> cluster_id;
                datum -> letter(k - 1) -> set_phones(cluster_id);
            }
        }
        vector<int> cluster_id;
        Segment* segment;
        // Create proper segments
        if (boundary_id == next_j - 1) {
            int first_id = SampleIndexFromLogDistribution(segment_prob_individual_cluster[context][0][0][j][boundary_id].probs());
            cluster_id.push_back(first_id);
            segment = new Segment(_config, first_id);
            (datum -> segments()).push_back(segment);
            for (int p = j; p < next_j; ++p) {
                segment -> push_back(bounds[p]);
            }
        }
        else {
            int first_id = SampleIndexFromLogDistribution(segment_prob_individual_cluster[context][1][0][j][boundary_id].probs());
            cluster_id.push_back(first_id);
            segment = new Segment(_config, first_id);
            (datum -> segments()).push_back(segment);
            for (int p = j; p <= boundary_id; ++p) {
                segment -> push_back(bounds[p]);
            }
            int second_id = SampleIndexFromLogDistribution(segment_prob_individual_cluster[context][1][1][boundary_id + 1][next_j - 1].probs());
            cluster_id.push_back(second_id);
            segment = new Segment(_config, second_id);
            (datum -> segments()).push_back(segment);
            for (int p = boundary_id + 1; p < next_j; ++p) {
                segment -> push_back(bounds[p]);
            }
        }
        datum -> letter(next_i - 1) -> set_phones(cluster_id);
        i = next_i;
        j = next_j;
    }
    if (i >= l && j < b) {
        cout << "l_ptr: " << i << " " << ", b_ptr: " << j << " out of (" << l << ", " << b << ")\n";
        cout << "Align not complete in Sampling forward, need to check " << datum -> tag() << endl;
        datum -> set_corrupted(true);
        datum -> set_corrupted_type(5);
    }
    else if (i < l && j >= b) {
        for (int k = i + 1; k <= l; ++k) {
            vector<int> phones;
            datum -> letter(k - 1) -> set_phones(phones);
        }
    }
}

void Sampler::MessageBackward(Datum* datum, \
            map<vector<int>, ProbList<int>**>& segment_prob, \
            map<vector<int>, L2S*>& l2s, \
            ProbList<int>** B, ProbList<int>** Bstar) {
    int l = (datum -> letters()).size(); 
    int b = (datum -> bounds()).size();
    vector<Bound*> bounds = datum -> bounds();
    // Initialization
    B[l][b].push_back(0, -1);
    int i = l - 1;
    int num_epsilons_in_a_row = 0;
    while (num_epsilons_in_a_row < _config -> max_num_epsilons() && (i > 0 && datum -> letter(i - 1) -> label() >= 0)) {
        if (l2s.find(datum -> Context(i)) == l2s.end()) {
            EstablishRootInL2S(datum -> Context(i), l2s);
        }
        B[i][b].push_back(l2s[datum -> Context(i)] -> Prior(0) + B[i + 1][b].value(), -1);
        if (datum -> letter(i) -> label() >= 0) {
            ++num_epsilons_in_a_row;
        }
        else {
            num_epsilons_in_a_row = 0;
        }
        --i;
    }
    for (; i > 0; --i) {
        B[i][b].push_back(MIN_PROB_VALUE, -1);
    }
    int max_num_units = _config -> max_num_units();
    int max_duration = _config -> max_duration();
    int total_frame_num = 0;
    vector<int> accumulated_frame_num(b, 0);
    for (int i = 0; i < b; ++i) {
        total_frame_num += bounds[i] -> frame_num();
        accumulated_frame_num[i] = total_frame_num;
    }
    // Compute B and Bstar
    for (int i = b - 1; i >= 0; --i) {
        for (int j = l; j >= 0; --j) {
            // Compute Bstar
            if (j > 0) {
                vector<int> context = datum -> Context(j - 1);
                int start_frame = i == 0 ? 0 : accumulated_frame_num[i - 1];
                for (int k = i + 1; k <= b && ((accumulated_frame_num[k - 1] - start_frame) <= max_num_units * max_duration || k <= i + 2 ); ++k) { 
                    Bstar[j][i].push_back(B[j][k].value() + segment_prob[context][i][k - 1].value(), k);
                }
            }
            // Compute B
            if (j == l) {
                B[j][i].push_back(MIN_PROB_VALUE, -1);
            }
            else if ((j > 0 && i > 0) || (j == 0 && i == 0)){
                int k = j + 1;
                int num_epsilons_in_a_row = 0;
                float epsilon_prob = 0;
                bool seen_space = false;
                if (j > 0 && datum -> letter(j - 1) -> label() < 0) {
                    seen_space = true;
                } 
                while (k <= l && num_epsilons_in_a_row <= _config -> max_num_epsilons() && \
                        !(seen_space && datum -> letter(k - 1) -> label() < 0)) {
                    B[j][i].push_back(Bstar[k][i].value() + epsilon_prob, k);
                    // ++num_epsilons_in_a_row;
                    if (datum -> letter(k - 1) -> label() >= 0) {
                        ++num_epsilons_in_a_row;
                    }
                    else {
                        seen_space = true;
                        num_epsilons_in_a_row = 0;
                    }
                    if (l2s.find(datum -> Context(k - 1)) == l2s.end()) {
                        EstablishRootInL2S(datum -> Context(k - 1), l2s);
                    }
                    epsilon_prob += l2s[datum -> Context(k - 1)] -> Prior(0);
                    ++k;
                }
            }
        }
    }
}

vector<int> Sampler::FindRootInL2S(vector<int> label, map<vector<int>, L2S*>& l2s) {
    if (l2s.find(label) != l2s.end()) {
        return label;
    }
    else {
        vector<int> parent_label;
        if (label.size() == 1) {
            parent_label.push_back(-100);
        }
        else {
            for (int i = 1; i < (int) label.size() - 1; ++i) {
                parent_label.push_back(label[i]);
            }
        }
        return FindRootInL2S(parent_label, l2s);
    }
}

void Sampler::RemoveL2SAlignment(Datum* datum, \
          map<vector<int>, L2S*>& l2s_counter, map<vector<int>, L2S*>& l2s) {
    // The if function is for the initial case for datum with transcribed text
    // the text is not aligned to any segments yet, so no need to remove
    if ((datum -> segments()).size()) {
        for (int i = 0 ; i < (int) (datum -> letters()).size(); ++i) {
            vector<int> context = datum -> Context(i);
            if (l2s_counter.find(context) == l2s_counter.end()) {
                EstablishLeafInL2S(context, l2s_counter);
            }
            l2s_counter[context] -> Minus((datum -> letter(i)) -> phones());
            if (l2s[context] -> is_collapsed()) {
                if (l2s.find(context) == l2s.end()) {
                    EstablishRootInL2S(context, l2s);
                }
                l2s[context] -> Minus((datum -> letter(i)) -> phones());
            }
        }
    }
}

void Sampler::RemoveClusterAssignment(Datum* datum, \
                                vector<Cluster*>& cluster_counter) {
    vector<Segment*> segments = datum -> segments();
    vector<Segment*>::iterator s_iter = segments.begin();
    for (; s_iter != segments.end(); ++s_iter) {
        if (!(cluster_counter[(*s_iter) -> id()] -> is_fixed())) {
            cluster_counter[(*s_iter) -> id()] -> Minus(*s_iter); 
        }
    }
    datum -> ClearSegs();
}

void Sampler::AddL2SAlignment(Datum* datum, map<vector<int>, \
          L2S*>& l2s_counter, map<vector<int>, L2S*>& l2s) {
    int letter_seq_size = (datum -> letters()).size();
    for (int i = 0; i < letter_seq_size; ++i) {
        vector<int> context = datum -> Context(i);
        if (l2s_counter.find(context) == l2s_counter.end()) {
            EstablishLeafInL2S(context, l2s_counter);
        }
        l2s_counter[context] -> Plus(datum -> letter(i) -> phones());
        if (l2s[context] -> is_collapsed()) {
            if (l2s.find(context) == l2s.end()) {
                EstablishRootInL2S(context, l2s);
            }
            l2s[context] -> Plus(datum -> letter(i) -> phones());
        }
    } 
}

void Sampler::AddClusterAssignment(Datum* datum, \
                   vector<Cluster*>& cluster_counter) {
    vector<Segment*> segments = datum -> segments();
    vector<Segment*>::iterator s_iter = segments.begin();
    for (; s_iter != segments.end(); ++s_iter) {
        if (!(cluster_counter[(*s_iter) -> id()] -> is_fixed())) {
            cluster_counter[(*s_iter) -> id()] -> Plus(*s_iter);
        }
    }
}

void Sampler::EstablishRootInL2S(vector<int> context, \
        map<vector<int>, L2S*>& l2s) {
    L2S* new_l2s = new L2S(_config, context);
    if (!(new_l2s -> is_collapsed())) {
        SampleL2SParams(new_l2s);
    }
    l2s[context] = new_l2s;
    vector<int> label = context;
    vector<int> parent_label = new_l2s -> parent_label();
    while (l2s.find(parent_label) == l2s.end()) {
        L2S* parent_l2s = new L2S(_config, parent_label);
        if (!(parent_l2s -> is_collapsed())) {
            SampleL2SParams(parent_l2s);
        }
        l2s[parent_label] = parent_l2s;
        l2s[label] -> set_parent(parent_l2s);
        label = parent_label;
        parent_label = parent_l2s -> parent_label();
    }
    l2s[label] -> set_parent(l2s[parent_label]);
}

void Sampler::EstablishLeafInL2S(vector<int> label, map<vector<int>, L2S*>& l2s) {
    L2S* new_l2s = new L2S(_config, label);
    l2s[label] = new_l2s;
}

Sampler::~Sampler() {
    vslDeleteStream(&stream);
}

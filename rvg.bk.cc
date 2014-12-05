#include "rvg.h"

#include <mkl_vsl.h>
#include <climits>

#define BRNG VSL_BRNG_MT19937
#define UNIFORM_METHOD VSL_RNG_METHOD_UNIFORM_STD

using namespace std;
using namespace boost;
using namespace boost::random;

RandomVarGen::RandomVarGen() {
    _batch_size = 50000;
    _index = _batch_size;
    _uniform_samples = NULL;
    size_t SEED = time(0);
    generator.seed(SEED);
    vslNewStream(&_stream, BRNG, SEED);
}

float RandomVarGen::GetGammaSample(float a, float b) {
    gamma_distribution<> dist(a, b);
    return dist(generator);
}

float RandomVarGen::GetUniformSample() {
    if (_index >= _batch_size) {
        if (_uniform_samples == NULL) {
            _uniform_samples = new float [_batch_size];
        }
        vsRngUniform(UNIFORM_METHOD, _stream, _batch_size, _uniform_samples, 0, 1);
        _index = 0;
    }
    return _uniform_samples[_index++];
}

float RandomVarGen::GetNormalSample(float u, float v) {
    normal_distribution<> dist(u, v);
    return dist(generator);
}

RandomVarGen::~RandomVarGen() {
    if (_uniform_samples != NULL) {
        delete[] _uniform_samples;
    }
    vslDeleteStream(&_stream);
}

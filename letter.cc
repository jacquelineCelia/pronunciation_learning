#include "letter.h"

Letter::Letter(int label) {
    _label = label;
    _word_boundary = false;
    _sentence_boundary = false;
}

Letter::Letter(int label, vector<int> phones) {
    _label = label;
    _phones = phones;
    _sentence_boundary = false;
    _word_boundary = false;
}

Letter::Letter(const Letter& rhs) {
    _word_boundary = rhs.is_word_boundary();
    _sentence_boundary = rhs.is_sentence_boundary();
    _label = rhs.label();
    _phones = rhs.phones();
}

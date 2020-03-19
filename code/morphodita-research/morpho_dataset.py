import collections
import math
import pickle
import re

import numpy as np
import os
import transformers
import tensorflow as tf
from IPython.core.display import Math


class MorphoDataset:
    FORMS = 0
    LEMMAS = 1
    TAGS = 2
    FACTORS = 3
    FACTORS_MAP = {"Forms": FORMS, "Lemmas": LEMMAS, "Tags": TAGS}

    EMBEDDINGS = 3
    ELMOS = 4
    BERT = 5

    PAD = 0
    UNK = 1

    class _Factor:
        def __init__(self, characters, train=None):
            self.words_map = train.words_map if train else {'<pad>': MorphoDataset.PAD, '<unk>': MorphoDataset.UNK}
            self.words = train.words if train else ['<pad>', '<unk>']
            self.word_ids = []
            self.word_strings = []
            self.analyses_ids = []
            self.analyses_strings = []
            self.characters = characters
            if characters:
                self.alphabet_map = train.alphabet_map if train else {'<pad>': MorphoDataset.PAD,
                                                                      '<unk>': MorphoDataset.UNK}
                self.alphabet = train.alphabet if train else ['<pad>', '<unk>']
                self.charseqs_map = {'<pad>': MorphoDataset.PAD, '<unk>': MorphoDataset.UNK}
                self.charseqs = [[MorphoDataset.PAD], [MorphoDataset.UNK]]
                self.charseq_ids = []

    class FactorBatch:
        def __init__(self, word_ids, charseq_ids=None, charseqs=None, charseq_lens=None, analyses_ids=None):
            self.word_ids = word_ids
            self.charseq_ids = charseq_ids
            self.charseqs = charseqs
            self.charseq_lens = charseq_lens
            self.analyses_ids = analyses_ids

    def __init__(self, filename, embeddings=None, elmo=None, train=None, lemma_re_strip=None, lemma_rule_min=None,
                 shuffle_batches=True, max_sentences=None, bert=None, bert_words=None, compute_bert=False):
        # Create factors
        self._factors = []
        for f in range(self.FACTORS):
            self._factors.append(self._Factor(f == self.FORMS, train._factors[f] if train else None))

        # Prepare embeddings
        self._embeddings = {}
        # TODO doplnit pro BERTa
        if train:
            self._embeddings = train._embeddings
        elif embeddings is not None:
            for i, word in enumerate(embeddings):
                self._embeddings[word] = i + 1

        # Load contextualized embeddings
        self._elmo = []
        if elmo:
            for elmo_path in elmo.split(","):
                with np.load(elmo_path, allow_pickle=True) as elmo_file:
                    for i, (_, value) in enumerate(elmo_file.items()):
                        if i >= len(self._elmo):
                            self._elmo.append(value)
                        else:
                            self._elmo[i] = np.concatenate([self._elmo[i], value], axis=1)
                    assert i + 1 == len(self._elmo)
        self._elmo_size = train.elmo_size if train else self._elmo[0].shape[1] if self._elmo else 0

        # Initialize remma_re_strip
        self._lemma_re_strip = train._lemma_re_strip if train else re.compile(
            lemma_re_strip) if lemma_re_strip else None

        # Load the sentences
        lemma_rules = collections.defaultdict(lambda: 0)
        if filename is not None:
            with open(filename, "r", encoding="utf-8") as file:
                in_sentence = False
                for line in file:
                    line = line.rstrip("\r\n")

                    if line:
                        form, lemma, tag, *rest = line.split("\t")
                        assert len(rest) % 2 == 0

                        data = [form, lemma, tag]
                        analyses = [[], [], []]
                        if train:
                            analyses[self.LEMMAS] = [rest[i] for i in range(0, len(rest), 2)]
                            analyses[self.TAGS] = [rest[i] for i in range(1, len(rest), 2)]

                        for f in range(self.FACTORS):
                            factor = self._factors[f]
                            if not in_sentence:
                                if len(factor.word_ids): factor.word_ids[-1] = np.array(factor.word_ids[-1], np.int32)
                                factor.word_ids.append([])
                                factor.word_strings.append([])
                                factor.analyses_ids.append([])
                                factor.analyses_strings.append([])
                                if factor.characters: factor.charseq_ids.append([])

                            word = data[f]
                            analysis = analyses[f]

                            factor.word_strings[-1].append(word)
                            factor.analyses_strings[-1].append(analysis)
                            factor.analyses_ids[-1].append(np.array([1 for _ in analysis], np.int32))

                            # Character-level information

                            # factor.alphabet_map - kazde nove pismenko ma poradove cislo
                            # factor.charseqs_map - kazde nove slovo ma poradove cislo
                            # factor.charseqs - slovo reprezentovane cisly jednotlivych pismen
                            # factor.charseq_ids - text reprezentovany cisly slov (?)

                            # word_ids jsou id tagu (odpovedi) a nebo id slov (v pripade 0)
                            if factor.characters:
                                if word not in factor.charseqs_map:
                                    factor.charseqs_map[word] = len(factor.charseqs)
                                    factor.charseqs.append([])
                                    for c in word:
                                        if c not in factor.alphabet_map:
                                            if train:
                                                c = '<unk>'
                                            else:
                                                factor.alphabet_map[c] = len(factor.alphabet)
                                                factor.alphabet.append(c)
                                        factor.charseqs[-1].append(factor.alphabet_map[c])
                                factor.charseq_ids[-1].append(factor.charseqs_map[word])

                            # Word-level information
                            if f == self.LEMMAS:
                                if self._lemma_re_strip: word = self._lemma_re_strip.sub("", word)
                                word = self._gen_lemma_rule(data[self.FORMS], word)

                            if f == self.LEMMAS and not train:
                                factor.word_ids[-1].append(self.UNK)
                                lemma_rules[word] += 1
                            else:
                                if word not in factor.words_map:
                                    if train:
                                        word = '<unk>'
                                    else:
                                        factor.words_map[word] = len(factor.words)
                                        factor.words.append(word)
                                factor.word_ids[-1].append(factor.words_map[word])

                        in_sentence = True
                    else:
                        in_sentence = False
                        if max_sentences is not None and len(self._factors[self.FORMS].word_ids) >= max_sentences:
                            break

        # Compute sentence lengths
        self._sentence_lens = []
        sentences = len(self._factors[self.FORMS].word_ids)
        if sentences:
            self._sentence_lens = np.zeros([sentences], np.int32)
            for i in range(len(self._factors[self.FORMS].word_ids)):
                self._sentence_lens[i] = len(self._factors[self.FORMS].word_ids[i])

            # Map lemma rules to ids respecting lemma_rule_min for train data
            if not train:
                lemmas = self._factors[self.LEMMAS]
                for i in range(sentences):
                    for j in range(self._sentence_lens[i]):
                        word = lemmas.word_strings[i][j]
                        if self._lemma_re_strip: word = self._lemma_re_strip.sub("", word)
                        word = self._gen_lemma_rule(self._factors[self.FORMS].word_strings[i][j], word)
                        if lemma_rules[word] >= (lemma_rule_min or 1):
                            if word not in lemmas.words_map:
                                lemmas.words_map[word] = len(lemmas.words)
                                lemmas.words.append(word)
                            lemmas.word_ids[i][j] = lemmas.words_map[word]

            # Map analyses
            for f in [self.LEMMAS, self.TAGS]:
                factor = self._factors[f]
                for i in range(sentences):
                    for j in range(self._sentence_lens[i]):
                        for k in range(len(factor.analyses_strings[i][j])):
                            word = factor.analyses_strings[i][j][k]
                            if f == self.LEMMAS:
                                if self._lemma_re_strip: word = re.sub(self._lemma_re_strip, "", word)
                                word = self._gen_lemma_rule(self._factors[self.FORMS].word_strings[i][j], word)
                            if word in factor.words_map:
                                factor.analyses_ids[i][j][k] = factor.words_map[word]

            # Shuffling initialization
            self._shuffle_batches = shuffle_batches
            self._permutation = np.random.permutation(len(self._sentence_lens)) if self._shuffle_batches else np.arange(
                len(self._sentence_lens))

            # Asserts
            if self._elmo:
                assert sentences == len(self._elmo)
                for i in range(sentences):
                    assert self._sentence_lens[i] == len(self._elmo[i])

        if compute_bert:

            # TODO jaky model - cased nebo uncased
            config = transformers.BertConfig.from_pretrained("bert-base-multilingual-uncased")
            config.output_hidden_states = True
            # TODO nastavit tokeny stejne (mask, unk atd.. ?)
            tokenizer = transformers.BertTokenizer.from_pretrained("bert-base-multilingual-uncased"
                                                                   # ,
                                                                   # unk_token=self._factors[self.FORMS].words[self.UNK],
                                                                   # pad_token=self._factors[self.FORMS].words[self.PAD]
            )
            # TODO predstahnout
            model = transformers.TFBertModel.from_pretrained("bert-base-multilingual-uncased",
                                                             config=config)

            if not train:
                forms_nonunique = self._factors[self.FORMS].words
            else:
                forms_nonunique = [val for sublist in self.factors[0].word_strings for val in sublist]

            bert_words_new = np.unique(forms_nonunique)

            if bert_words:
                bert_words_new = list(set(bert_words_new) - set(bert_words))

            #bert_embeddings = [np.zeros(768) for i in range(len(bert_words_new))]

            batch_size_bert = 16
            bert_embeddings_tokens = None
            for i in range(0, math.ceil(len(bert_words_new) / batch_size_bert)):
                batch_words = bert_words_new[i: min(i + 16, len(bert_words_new))]
                w_subwords = [tokenizer.encode(w) for w in batch_words]
                #max_len = len(max(w_subwords, key=len))
                max_len = 15
                padded =  [i + [0]*(max_len-len(i)) for i in w_subwords]
                #word_tok = tf.convert_to_tensor([numb if numb is not None else 0 for numb in [x for x in w_subwords]])
                word_tok = tf.convert_to_tensor(padded)
                if bert_embeddings_tokens is None:
                    bert_embeddings_tokens = model(word_tok)[0]
                else:
                    bert_embeddings_tokens = tf.concat([bert_embeddings_tokens, model(word_tok)[0]], axis = 0)

            bert_embeddings = np.mean(bert_embeddings_tokens, axis = 1)


            # for i in range(0, len(bert_words_new)):
            #     # TODO batch size
            #     word = bert_words_new[i].lower()
            #     w_subwords = tokenizer.encode(word)
            #     word_tok = tf.convert_to_tensor(w_subwords)[None, :]
            #
            #     # embeddings for one word (1,768)
            #     bert_embeddings_tokens = model(word_tok)[0][0][1:6]
            #     bert_embeddings[i] = np.mean(bert_embeddings_tokens, axis=0)
            # self._bert_emb = bert_embeddings

            if bert_words:
                bert_words.append(bert_words_new)
            else:
                bert_words = bert_words_new

            if len(bert_embeddings):
                self.save_bert(bert_words_new, bert_embeddings, bert + "_" + "_".join(filename.split("-")[-2:]))

        if bert:
            self._berts = {}
            for i, word in enumerate(bert_words):
                self._berts[word] = i + 1

    @property
    def sentence_lens(self):
        return self._sentence_lens

    @property
    def factors(self):
        return self._factors

    @property
    def elmo_size(self):
        return self._elmo_size

    def save_mappings(self, path):
        with open(path, mode="wb") as mappings_file:
            pickle.dump(MorphoDataset(None, train=self), mappings_file)

    def save_bert(self, words, embeddings, file):
        for_save = [words, embeddings]
        with open(file + '.pickle', 'wb') as handle:
            pickle.dump(for_save, handle)

    @staticmethod
    def load_mappings(path):
        with open(path, mode="rb") as mappings_file:
            return pickle.load(mappings_file)

    def epoch_finished(self):
        if len(self._permutation) == 0:
            self._permutation = np.random.permutation(len(self._sentence_lens)) if self._shuffle_batches else np.arange(
                len(self._sentence_lens))
            return True
        return False

    def next_batch(self, batch_size):
        batch_size = min(batch_size, len(self._permutation))
        batch_perm = self._permutation[:batch_size]
        self._permutation = self._permutation[batch_size:]

        # General data
        batch_sentence_lens = self._sentence_lens[batch_perm]
        max_sentence_len = np.max(batch_sentence_lens)

        # Word-level data
        factors = []
        for factor in self._factors:
            factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32)))
            for i in range(batch_size):
                factors[-1].word_ids[i, 0:batch_sentence_lens[i]] = factor.word_ids[batch_perm[i]]

        # Embeddings
        forms = self._factors[self.FORMS]
        factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32)))
        if len(self._embeddings):
            for i in range(batch_size):
                for j, string in enumerate(forms.word_strings[batch_perm[i]]):
                    mapped = self._embeddings.get(string, 0)
                    if not mapped: mapped = self._embeddings.get(string.lower(), 0)
                    factors[-1].word_ids[i, j] = mapped

        # Contextualized embeddings
        if self._elmo:
            factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len, self.elmo_size], np.float32)))
            for i in range(batch_size):
                factors[-1].word_ids[i, :len(self._elmo[batch_perm[i]])] = self._elmo[batch_perm[i]]

        # # BERT
        # # TODO bert tady spocitat pro batch (uz vypoctene pro cely dataset)
        # # TODO vytvorit promennou self, bert
        # if self._bert:
        #     factors.append(self.FactorBatch(np.zeros([batch_size, max_sentence_len], np.int32)))
        #     for i in range(batch_size):
        #         for j, string in enumerate(forms.word_strings[batch_perm[i]]):
        #             mapped = self._embeddings.get(string, 0)
        #             if not mapped: mapped = self._bert.get(string.lower(), 0)
        #             factors[-1].word_ids[i, j] = mapped

        # Character-level data
        for f, factor in enumerate(self._factors):
            if not factor.characters: continue

            factors[f].charseq_ids = np.zeros([batch_size, max_sentence_len], np.int32)
            charseqs_map = {}
            charseqs = []
            charseq_lens = []
            for i in range(batch_size):
                for j, charseq_id in enumerate(factor.charseq_ids[batch_perm[i]]):
                    if charseq_id not in charseqs_map:
                        charseqs_map[charseq_id] = len(charseqs)
                        charseqs.append(factor.charseqs[charseq_id])
                    factors[f].charseq_ids[i, j] = charseqs_map[charseq_id]

            factors[f].charseq_lens = np.array([len(charseq) for charseq in charseqs], np.int32)
            factors[f].charseqs = np.zeros([len(charseqs), np.max(factors[f].charseq_lens)], np.int32)
            for i in range(len(charseqs)):
                factors[f].charseqs[i, 0:len(charseqs[i])] = charseqs[i]

        # Analyses data
        for f in [self.LEMMAS, self.TAGS]:
            factors[f].analyses_ids = []
            for index in batch_perm:
                factors[f].analyses_ids.append(self._factors[f].analyses_ids[index])

        return self._sentence_lens[batch_perm], factors

    def write_sentence(self, output, index, overrides):
        for i in range(self._sentence_lens[index]):
            fields = []
            for f in range(self.FACTORS):
                factor = self._factors[f]
                field = factor.word_strings[index][i]

                # Overrides
                if overrides is not None and f < len(overrides) and overrides[f] is not None:
                    if overrides[f][i] < 0:
                        field = factor.analyses_strings[index][i][-overrides[f][i] - 1]
                    else:
                        field = factor.words[overrides[f][i]]
                        if f == self.LEMMAS:
                            try:
                                field = self._apply_lemma_rule(fields[self.FORMS], field)
                            except:
                                field = fields[self.FORMS]
                fields.append(field)

            # Analyses
            assert len(self._factors[self.LEMMAS].analyses_strings[index][i]) == len(
                self._factors[self.TAGS].analyses_strings[index][i])
            for j in range(len(self._factors[self.LEMMAS].analyses_strings[index][i])):
                for f in [self.LEMMAS, self.TAGS]:
                    fields.append(self._factors[f].analyses_strings[index][i][j])

            print("\t".join(fields), file=output)
        print(file=output)

    @staticmethod
    def _min_edit_script(source, target):
        a = [[(len(source) + len(target) + 1, None)] * (len(target) + 1) for _ in range(len(source) + 1)]
        for i in range(0, len(source) + 1):
            for j in range(0, len(target) + 1):
                if i == 0 and j == 0:
                    a[i][j] = (0, "")
                else:
                    if i and a[i - 1][j][0] < a[i][j][0]:
                        a[i][j] = (a[i - 1][j][0] + 1, a[i - 1][j][1] + "-")
                    if j and a[i][j - 1][0] < a[i][j][0]:
                        a[i][j] = (a[i][j - 1][0] + 1, a[i][j - 1][1] + "+" + target[j - 1])
        return a[-1][-1][1]

    @staticmethod
    def _gen_lemma_rule(form, lemma):
        form = form.lower()

        previous_case = -1
        lemma_casing = ""
        for i, c in enumerate(lemma):
            case = "↑" if c.lower() != c else "↓"
            if case != previous_case:
                lemma_casing += "{}{}{}".format("¦" if lemma_casing else "", case,
                                                i if i <= len(lemma) // 2 else i - len(lemma))
            previous_case = case
        lemma = lemma.lower()

        best, best_form, best_lemma = 0, 0, 0
        for l in range(len(lemma)):
            for f in range(len(form)):
                cpl = 0
                while f + cpl < len(form) and l + cpl < len(lemma) and form[f + cpl] == lemma[l + cpl]: cpl += 1
                if cpl > best:
                    best = cpl
                    best_form = f
                    best_lemma = l

        rule = lemma_casing + ";"
        if not best:
            rule += "a" + lemma
        else:
            rule += "d{}¦{}".format(
                MorphoDataset._min_edit_script(form[:best_form], lemma[:best_lemma]),
                MorphoDataset._min_edit_script(form[best_form + best:], lemma[best_lemma + best:]),
            )
        return rule

    @staticmethod
    def _apply_lemma_rule(form, lemma_rule):
        casing, rule = lemma_rule.split(";", 1)
        if rule.startswith("a"):
            lemma = rule[1:]
        else:
            form = form.lower()
            rules, rule_sources = rule[1:].split("¦"), []
            assert len(rules) == 2
            for rule in rules:
                source, i = 0, 0
                while i < len(rule):
                    if rule[i] == "→" or rule[i] == "-":
                        source += 1
                    else:
                        assert rule[i] == "+"
                        i += 1
                    i += 1
                rule_sources.append(source)

            try:
                lemma, form_offset = "", 0
                for i in range(2):
                    j, offset = 0, (0 if i == 0 else len(form) - rule_sources[1])
                    while j < len(rules[i]):
                        if rules[i][j] == "→":
                            lemma += form[offset]
                            offset += 1
                        elif rules[i][j] == "-":
                            offset += 1
                        else:
                            assert (rules[i][j] == "+")
                            lemma += rules[i][j + 1]
                            j += 1
                        j += 1
                    if i == 0:
                        lemma += form[rule_sources[0]: len(form) - rule_sources[1]]
            except:
                lemma = form

        for rule in casing.split("¦"):
            if rule == "↓0": continue  # The lemma is lowercased initially
            case, offset = rule[0], int(rule[1:])
            lemma = lemma[:offset] + (lemma[offset:].upper() if case == "↑" else lemma[offset:].lower())

        return lemma

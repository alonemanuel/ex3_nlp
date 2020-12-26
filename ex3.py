import matplotlib.pyplot as plt
import nltk
import numpy as np

from scipy.linalg import svd

DATA = ['John likes NLP.',
        'He likes Mary.',
        'John likes machine learning.',
        'John wrote a post about NLP and got likes.']

STOP_WORDS = ['.', 'is', 'a', 'of', 'and']

PAIRS = [('John', 'He'), ('John', 'subfield'), ('Deep', 'machine')]


def build_co_occurence_mat(data=None, stop_words=None, win_size=1):
    if stop_words is None:
        stop_words = STOP_WORDS
    if data is None:
        data = DATA

    clean_vocab = get_clean_vocab_from_data(data, stop_words)
    vocab_size = len(clean_vocab)
    co_mat = np.zeros((vocab_size, vocab_size))

    for sentence in data:
        # print(f'\nOriginal sentence: {sentence}')
        tokenized = nltk.word_tokenize(sentence)
        # print(f'Tokenized sentence: {tokenized}')
        # remove all stop words from the tokenized sentence
        no_stopwords = [word for word in tokenized if word not in stop_words]
        # print(f'No-stopwords sentence: {no_stopwords}')

        for i, word_of_interest in enumerate(no_stopwords):
            window = no_stopwords[max(i - win_size, 0):i + win_size + 1]
            # print(f'Window of \'{word_of_interest}\' of size {win_size} is: {window}')
            for neighbor_word in window:
                if neighbor_word == word_of_interest:
                    continue
                neighbor_index = clean_vocab.index(neighbor_word)
                word_index = clean_vocab.index(word_of_interest)
                co_mat[neighbor_index, word_index] += 1

    # print(f'Vocab size is: {vocab_size}')
    # print(f'\nCo-mat shape is: {co_mat.shape}, and content is:\n{co_mat}')
    co_mat = co_mat.astype(int)
    heatmap_title = f'Word occurrences with window size of {win_size}'
    create_heatmap(heatmap_title, co_mat, clean_vocab)

    return co_mat


def create_heatmap(title, data, ordered_labels=None):
    fig, ax = plt.subplots()
    im = ax.imshow(data)
    vocab_size = len(data)
    if ordered_labels:
        ax.set_xticks(np.arange(vocab_size))
        ax.set_yticks(np.arange(vocab_size))
        ax.set_xticklabels(ordered_labels)
        ax.set_yticklabels(ordered_labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
    for i in range(vocab_size):
        for j in range(vocab_size):
            text = ax.text(j, i, data[i, j],
                           ha="center", va="center", color="w")
    ax.set_title(title)
    fig.tight_layout()
    plt.show()


def get_clean_vocab_from_data(data, stop_words):
    vocab = set()
    for sentence in data:
        vocab = vocab.union(nltk.word_tokenize(sentence))
    clean_vocab = list(vocab.difference(set(stop_words)))
    clean_vocab.sort()
    return clean_vocab


def create_svd(matrix):
    print(f'\nIn create_svd')
    U, S, VT = svd(matrix)
    S = np.diag(S)
    print(f'\nU:\n{np.round(U, 3)}')
    print(f'\nS:\n{np.round(S, 3)}')
    print(f'\nVT:\n{np.round(VT, 3)}')
    return U, S, VT


def reduce_svd(u, s, vt):
    reduced_mats = [u, s, vt]
    for i in range(3):
        mat = reduced_mats[i]
        orig_size = len(mat)
        print(f'Orig size: {orig_size}')
        reduced_size = np.floor(0.3 * orig_size).astype(int)
        print(f'Reduced_size: {reduced_size}')
        reduced_mats[i] = mat[:reduced_size, :reduced_size]

    u = reduced_mats[0]
    s = reduced_mats[1]
    vt = reduced_mats[2]
    x = u @ s @ vt

    print(f'\nU:\n{np.round(u, 3)}')
    print(f'\nS:\n{np.round(s, 3)}')
    print(f'\nVT:\n{np.round(vt, 3)}')
    print(f'\nX:\n{np.round(x, 3)}')

    return u, s, vt


def cosine_similarity(pairs=None):
    if pairs is None:
        pairs = PAIRS

    for pair in pairs:
        cos_sim =


def main():
    co_mat = build_co_occurence_mat()

    u, s, vt = create_svd(co_mat)
    u0, s0, vh0 = reduce_svd(u, s, vt)


if __name__ == '__main__':
    main()

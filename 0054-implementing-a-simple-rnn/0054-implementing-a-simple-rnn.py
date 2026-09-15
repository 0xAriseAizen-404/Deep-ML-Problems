import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
    # Your code here
    output = np.array(initial_hidden_state, dtype=float)
    for word_vector in input_sequence:
        wv = np.array(word_vector, dtype=float)
        output = np.tanh(
            np.dot(Wh, output) +
            np.dot(Wx, wv) +
            b
        )
    return np.round(output, 4).tolist()

# TC: O(T(HD + H²))
# SC: O(HD + H² + H)
import numpy as np

class LSTM:
	def __init__(self, input_size, hidden_size):
		self.input_size = input_size
		self.hidden_size = hidden_size

		# Initialize weights and biases
		self.Wf = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wi = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wc = np.random.randn(hidden_size, input_size + hidden_size)
		self.Wo = np.random.randn(hidden_size, input_size + hidden_size)

		self.bf = np.zeros((hidden_size, 1))
		self.bi = np.zeros((hidden_size, 1))
		self.bc = np.zeros((hidden_size, 1))
		self.bo = np.zeros((hidden_size, 1))

	def forward(self, X, initial_hidden_state, initial_cell_state):
		"""
		Processes a sequence of inputs and returns the hidden states, final hidden state, and final cell state.
		"""
		def sigmoid(x):
			return 1 / (1 + np.exp(-x))
		def tanh(x):
			return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
		
		H = initial_hidden_state
		C = initial_cell_state
		hidden_states = []
		for word in X:
			word = word.reshape(-1, 1)
			combined = np.vstack((H, word))
			# Forget Gate
			FG = sigmoid(np.dot(self.Wf, combined) + self.bf)
			# Input Gate
			IG = sigmoid(np.dot(self.Wi, combined) + self.bi)
			C_dash = tanh(np.dot(self.Wc, combined) + self.bc)
			# Cell State Updation
			C = C * FG + IG * C_dash
			# Output Gate
			OG = sigmoid(np.dot(self.Wo, combined) + self.bo)
			H = OG * tanh(C)
			hidden_states.append(H)
		return hidden_states, H, C
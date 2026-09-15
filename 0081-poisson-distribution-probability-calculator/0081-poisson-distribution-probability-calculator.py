import math

def poisson_probability(k, lam):
	"""
	Calculate the probability of observing exactly k events in a fixed interval,
	given the mean rate of events lam, using the Poisson distribution formula.
	:param k: Number of events (non-negative integer)
	:param lam: The average rate (mean) of occurrences in a fixed interval
	"""
	return ((lam**k) * (math.exp(-lam))) / math.factorial(k)

	# let's say that there is a shop, on based on the observation, they found out in a Specific Time Interval, like in One Hour, the average number of customers they are getting is 5, i.e, 'Lambda' here
	# and now, they want to know what is the probability of getting 3 customers in the next hour ?
	# or probability of getting 7 customers in the next hour ? like that, it is 'K' here
	# formula is Probability = ((lam**k) * (math.exp(-lam))) / math.factorial(k)

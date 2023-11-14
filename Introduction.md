
In Decision-Making, Machine Learning models need to not only make predictions but also quantify their predictions' uncertainty. A point prediction from the model might be dramatically different from the real value because of the high stochasticity of the real world. But, on the other hand, if the model could estimate the range which guarantees to cover the true value with high probability, model could compute the best and worst rewards and make more sensible decisions.

For example
* While buying a house, the predictions' upper bound can be useful for a buyer to be certain whether it will be able to buy a house or not.
* While Identifying an object, Applying threshold on softmax predictions can help us identify what that object could be.

Conformal prediction is a technique for quantifying such uncertainties for AI systems. In particular, given an input, conformal prediction estimates a prediction interval in regression problems and a set of classes in classification problems. Both the prediction interval and sets are guaranteed to cover the true value with high probability.

--- 

#### Theory


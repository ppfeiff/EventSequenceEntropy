# EventSequenceEntropy
Code for BPM Paper XXX

# Training LSTM
Unfortunately, I cannot publish the code for training the LSTM models as this is part of some bigger package which 
we cannot publish for now and which would make the authors identifiable. 

However, it's pretty straightforward. Given you have a LSTM model, you create ngrams using the function given in this repo.
Afterwards, you train the LSTM with the ngrams and predict the last activity in the ngram. Thats it.

# EventSequenceEntropy
Code for BPM Paper XXX

# Training LSTM
Unfortunately, we cannot publish the code for training the LSTM models as this is part of some bigger package which 
we cannot publish for now and which would make the authors identifiable. 

However, it's pretty straightforward. Given you have a LSTM model, you create ngrams using the functions given in this repo.
Either event ngrams, trace ordered ngrams or ngrams of events in traces.
Afterwards, you train the LSTM with the ngrams and predict the last activity in the ngram. Thats it.

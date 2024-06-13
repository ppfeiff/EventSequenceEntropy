# BPM 24 Event Sequence Entropy and Predictability
This repository contains code and higher resolution figures for the BPM 24 Forum Paper 
"Trace vs. Time: Entropy Analysis and Event Predictability of Traceless Event Sequencing", 
authored by Peter Pfeiffer and Peter Fettke (DFKI and Saarland University).

Abstract: Process mining offers powerful techniques to analyze realworld event data, aiming to improve processes. 
Typically, the data is stored and examined in event logs as traces, where each trace contains the sequence of events pertaining to a specific process case. 
A case can, e.g., represent the management of a customer request or the sequence of events from ordering to delivering a product to a customer in online
retail businesses. While this approach allows to analyze and gain insights from complex event data, it also isolates events that in reality are correlated, potentially concealing important process behavior. 
In this paper, we motivate and conceptualize the approach to describe the observations generated by the underlying system as a single event sequence that is ordered as being executed. 
We study and compare how much the event order and trace notion affect the entropy rates of different real-life processes. 
Further, we investigate how predictable next activities in event sequences are. 
Our study indicates that ordering the events as executed does not necessarily increase the entropy rates of the process. 
We discuss these findings and their implications for future research.

# Code
Tested with python 3.12

## Logs
We use the following logs
- BPIC 13 Incidents: https://doi.org/10.4121/uuid:500573e6-accc-4b0c-9576-aa5468b10cee
- BPIC 17: https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b
- BPIC 18: https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972
- RFM: https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5

## Computing Entropy Rates
In the python file _compute_entropies.py_ load the event log you want to compute the entropy rates for. 
By running the script, entropy rates are computed and the maximum reliable _n_ returned 

## Training PPM models
Unfortunately, we cannot publish the code for training the PPM models as it is part of a bigger package which 
we cannot publish for now. We can share the code upon request.

However, training is pretty straightforward. First, create ngrams using the functions given in this repo.
Either event ngrams, trace ordered ngrams or ngrams of events in traces.
Afterwards, you train the RF, LSTM or Transformer model with the ngrams and predict the last activity in the ngram. Thats it.


# Figures

The following figures are available in higher resolution in this repo:

- [Trace-ordered event sequence](BPIC_2013_trace_level_event_sequence.pdf)
- [Event-ordered event sequence](BPIC_2013_single_level_event_sequence.pdf)


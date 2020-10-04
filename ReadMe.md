# Question Duplicate Detection

The problem was to identify whether two questions are duplicates or not.
For example, let’s take these two questions- “How old are you ? ”&“What is your age?”. 
Though these questions have no word in common but have the same intent. In this repository
I have build a Deep learning system to detect whether or not a pair of questions are duplicates 
of each other.

## Siamese Network
Siamese networks are neural networks containing two or more identical subnetwork components. 
It is important that not only the architecture of the subnetworks is identical, 
but the weights have to be shared among them as well for the network to be 
called “siamese”. The main idea behind siamese networks is that they can learn 
useful data descriptors that can be further used to compare between the inputs 
of the respective subnetworks.


In this project, i have the Siamese network along with Triplet-Loss function
to train the duplicate question detector.

# References
- Siamese Network - Bromley, Jane, et al. “Signature verification using a” siamese” time delay neural network.” Advances in neural information processing systems. 1994.
- Triplet loss - Schroff, Florian, Dmitry Kalenichenko, and James Philbin. “Facenet: A unified embedding for face recognition and clustering.” 2015.
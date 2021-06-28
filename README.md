# Optical Character Recognition system
## Feature Extraction 

Gived the training pages and the bounding boxes for each character, I extracted 
the images inside each bounding box and converted them into feature 
vectors (on which dimensionality reduction is performed after that). Some random noise
is also added to data in order to make it more robust.

I experimented with two dimensionality reduction techniques: PCA and LDA.
While PCA returns the axes on which the data is most spread, LDA returns
the axes on which the data is best separated. In this case LDA seems to perform
slightly better. The scores for the clean pages are indeed a bit lower (< -1%) but they
are much higher on the noisy pages (> +3%).

## Classifier 

I started by implementing a classifier using nearest neighbour and although
it generally gave good results, the scores for the noisy pages (where there
was more ambiguity) were rather low. Thus, I implemented k-nearest neighbours.
By trial and error, I realised that it gave best results when 9 nearest neighbours
were used. k-nearest neighbours gave slightly worse results for the clean pages (-0.2%)
but it gave significantly better results for the noisy pages (+2-3%). 

## Error Correction 

The error correction stage is done by separating the characters into words
and those that are not valid words are replaced by their closest match from
a dictionary.

First, I compute the distance between bounding boxes and I get the positions
at which the space is bigger than 6 (computed by trial and error) pixels or
when a new line starts. These are the positions at which each word ends. Then 
I iterate over the ends and make a word using the characters from the previous
end to the current end. For each word, I remove punctuation and look for the word
in a list of English words (dictionary). If the word is found it must be correct.
If not, I compute the Hamming distance between that word and each word in the dictionary
with the same length and return the one with minimum distance. However, the match
mustn't be too different from the initial word: a maximum of 3 characters changed
is allowed.

The scores after error correction are higher for each test page( +1-3.5%).

## Performance
The percentage errors (to 1 decimal place) for the development data are
as follows:

‐ Page 1: 95.1%
‐ Page 2: 96.7%
‐ Page 3: 91.2%
‐ Page 4: 77.4%
‐ Page 5: 62.6%
‐ Page 6: 50.6%

## Other information 

Before classification, the noise is removed from the test data using a median filter.

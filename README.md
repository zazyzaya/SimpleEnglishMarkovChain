# SimpleEnglishMarkovChain
Just messing around with markov chains

The vocabulary is the 850 words defined by "Basic English" which is convieniently used by Wikipedia for several articles which were crawled and used to train the chain.
~~It's not really good, it only has one layer and can't learn new words, but I just wanted to see if I could even make it. Enjoy I guess?~~
The Markov Chain is now capible of learning new words, and very quickly training on new data. To generate your own markov chain, simply add this to `main()`:

 ```python
 saveChain(nodelist, 'FileNameHere')
 ``` 
 
 and change the `VOCAB_FILE` on line 6 to whatever your filename was.
 You can then load your Markov Chain anytime using the `loadChain()` function and generate sentences. You can even train the same chain on different data!
 
 ### TODO 
 Add another layer to the chain? Who knows maybe someday

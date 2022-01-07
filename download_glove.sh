if [[ ! -d glove ]]; then
	mkdir glove
fi

cd glove
# wget http://nlp.stanford.edu/data/wordvecs/glove.6B.500d.zip
# unzip glove.6B.768d.zip

wget http://nlp.stanford.edu/data/wordvecs/glove.42B.300d.zip
unzip glove.42B.300d.zip

all: inferno.txt purgatorio.txt paradiso.txt

pg8800.txt:
	curl -L http://www.gutenberg.org/ebooks/8800.txt.utf-8 >$@

inferno.txt: pg8800.txt
	sed -n 10339,15586p $^ | ./preprocess.pl > $@

purgatorio.txt: pg8800.txt
	sed -n 5135,10307p $^ | ./preprocess.pl > $@

paradiso.txt: pg8800.txt
	sed -n 107,5104p $^ | ./preprocess.pl > $@

from Transformer_Text_AutoEncoder.AutoEncoder import TTAE

def read(path='./data/FinancialNews.txt'):
	with open(path, "r", encoding='utf-8', errors='ignore') as f:
		data = [i.strip() for i in f.readlines()]
	return data

sentences = read()
ttae = TTAE(sentences)
ttae.train(10, batch_size=1)
print(ttae.predict("Hello world"))
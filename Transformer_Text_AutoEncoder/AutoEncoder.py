from transformers import EncoderDecoderModel, BertTokenizer
import torch
from tqdm.notebook import tqdm

class TempObj:
	def __init__(self, encoder_outputs):
		self.last_hidden_state = encoder_outputs[0].data
		self.pooler_output = None#encoder_outputs[1].data
		self.hidden_states=None
		self.past_key_values=None
		self.attentions=None
		self.cross_attentions=None
		self.array = [self.last_hidden_state, self.pooler_output, self.hidden_states, self.past_key_values, self.attentions, self.cross_attentions]

	def __getitem__(self, idx):
		if idx==0:
			return self.last_hidden_state
		else:
			return self.pooler_output
		return [self.last_hidden_state, self.pooler_output]

	def get_encodings(self):
		return self.last_hidden_state

class TTAE:
	def __init__(self, path, encoding_model='bert-base-uncased', decoding_model='bert-base-uncased', lr=0.001, r_errors=None, r_encoding=None, no_cuda=False):
		self.no_cuda = no_cuda
		self.encoding_model_name = encoding_model
		self.decoding_model_name = decoding_model
		self.tokenizer = BertTokenizer.from_pretrained(encoding_model)
		self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(self.encoding_model_name, self.decoding_model_name)
		self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
		self.model.config.pad_token_id = self.tokenizer.pad_token_id
		self.model.config.vocab_size = self.model.config.decoder.vocab_size
		self.errors = r_errors
		self.encoding = r_encoding
		self.path = path
		self.optimizer = torch.optim.SGD(self.model.parameters(), lr=lr)
		if not self.no_cuda:
			self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
		else:
			self.device = torch.device("cpu")
		self.data = self.read()
		self.model.to(self.device)

	def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
		shifted_input_ids = input_ids.new_zeros(input_ids.shape)
		shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
		if decoder_start_token_id is None:
			raise ValueError("Make sure to set the decoder_start_token_id attribute of the model's configuration.")
		shifted_input_ids[:, 0] = decoder_start_token_id
		if pad_token_id is None:
			raise ValueError("Make sure to set the pad_token_id attribute of the model's configuration.")
		shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
		return shifted_input_ids

	def read(self):
		with open(self.path, "r", encoding=self.encoding, errors=self.errors) as f:
			data = [i.strip() for i in f.readlines()]
		return data

	def train(self, epochs, batch_size=8):
		self.model.train()
		for epoch in range(epochs):
			bar = tqdm([[i for i in self.data[i:i+batch_size]] for i in range(0, len(self.data), batch_size)])
			running_loss = 0.0
			for batch_idx, batch_sentences in enumerate(bar):
				input_ids = self.tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True).input_ids
				labels = self.tokenizer(batch_sentences, return_tensors="pt", padding=True, truncation=True).input_ids
				input_ids = input_ids.to(self.device)
				labels = labels.to(self.device)
				self.optimizer.zero_grad()
				encoder_outputs = self.model.encoder(input_ids=input_ids)
				tmp_obj = TempObj(encoder_outputs)
				outputs = self.model(labels=labels, encoder_outputs=tmp_obj)
				loss, logits = outputs.loss, outputs.logits
				loss.backward()
				self.optimizer.step()
				running_loss += loss.item()
				bar.set_description(str({"epoch": epoch+1, "loss": round(running_loss/(batch_idx+1), 3)}))
			bar.close()
		return self.model

	def predict(self, sentences):
		self.model.eval()
		if type(sentences)==str:
			sentences = [sentences]
		input_ids = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).input_ids
		labels = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True).input_ids
		input_ids = input_ids.to(self.device)
		labels = labels.to(self.device)
		encoder_outputs = self.model.encoder(input_ids=input_ids)
		tmp_obj = TempObj(encoder_outputs)
		outputs = self.model(labels=labels, encoder_outputs=tmp_obj)
		loss, logits = outputs.loss, outputs.logits
		o = torch.argmax(logits, dim=2)
		outs = self.tokenizer.batch_decode(o)
		return outs, tmp_obj.last_hidden_state

if __name__ == '__main__':
	ttae = TTAE('./data/FinancialNews.txt', no_cuda=True)
	ttae.train(10, batch_size=1)
	print(ttae.predict("Hello world"))
import torch
import torch.nn as nn

class VisualDialog(nn.Module):

    def __init__(self, feature_extractor, vocab_size, device="cpu"):

        super(VisualDialog, self).__init__()

        # Parameters
        self.hidden_size = 128
        self.input_size  = 100
        self.hidden_layers = 2
        self.feature_extractor_dim = 512
        self.vocabulary_size = vocab_size
        self.device = device

        # Image feture extractor
        self.feature_extractor = feature_extractor

        # LM
        self.embeddings = nn.Embedding(self.vocabulary_size, self.input_size)
        self.question_encoder = self.__init_encoder()
        self.question_keys = nn.Linear(256, 256)
        self.question_values = nn.Linear(256, 256)
        
        self.answer_decoder = self.__init_decoder()
        self.answer_linear = nn.Linear(
            2 * self.hidden_size + 2 * self.hidden_size + self.feature_extractor_dim + self.hidden_size * self.hidden_layers * 2, self.vocabulary_size)
    
    def __init_encoder(self):
        return nn.LSTM(
                self.input_size,
                self.hidden_size,
                self.hidden_layers,
                bias=True,
                batch_first=True,
                bidirectional=True
            )
    
    def __init_decoder(self):
        return nn.LSTM(
                self.input_size,
                self.hidden_size,
                self.hidden_layers,
                bias=True,
                batch_first=True,
                bidirectional=True
            )
    
    def encode(self, x):
        """
        x: [Questions] 
            * Dim1) Number of questions
            * Dim2) Max Question Length
            * Padded in front  
        """
        x = self.embeddings(x)
        x, h = self.question_encoder(x)
        key, vals = self.question_keys(x), self.question_values(x)
        return key, vals, h
    
    def decode(self, x, xlens, enc_keys, enc_vals, enc_hiddens, img):
        """
        x: [Answers] 
            * Dim1) Number of answers
            * Dim2) Max answers Length
            * Padded in end

        """
        embs = self.embeddings(x) # NUM_ANS, MAX_ANS_LEN, INPUT_DIM
        img_feats = self.feature_extractor(img.unsqueeze(0)) # (1, IMG_HIDDEN_DIM)
        NUM_QUESTIONS, NUM_TIME_STEPS, HIDDEN_DIMS = embs.size()
        history = torch.zeros((1, self.hidden_size * self.hidden_layers * 2)).to(self.device)

        outputs = []
        for qno in range(NUM_QUESTIONS):
            hidden = [enc_hiddens[0][:, qno, :].unsqueeze(1), enc_hiddens[1][:, qno, :].unsqueeze(1)] # [(NUM_LAYERS * BIDIRECTIONAL, 1, HIDDEN_DIM), (NUM_LAYERS * BIDIRECTIONAL, 1, HIDDEN_DIM)]
            hidden = [hidden[0].contiguous(), hidden[1].contiguous()]
            k = enc_keys[qno]
            v = enc_vals[qno]

            answers = []
            for t in range(NUM_TIME_STEPS):
                if t >= xlens[qno]: break
 
                x = embs[qno, t, :].unsqueeze(0).unsqueeze(1) # (1, 1, INPUT_DIM)
                
                x, hidden = self.answer_decoder(x, hidden)
                x = x.squeeze(0) # (1, INPUT_DIM)

                ques_context = self.attend(x, k, v)
                x = torch.cat([x, ques_context, history, img_feats], dim=-1) # (1, HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM + IMG_FEAT_DIM )
                x = self.answer_linear(x) # (1, VOCAB_SIZE)

                answers.append(x)
            outputs.append(torch.cat(answers, dim=0))
            history = hidden[0].view((1, -1)).clone()
        return outputs
    
    def attend(self, q, k, v):
        """
        x: (1, HIDDEN_DIM)
        k: (TIME_STEPS, HIDDEN_DIM)
        v: (TIME_STEPS, HIDDEN_DIM)
        """
        energy = torch.mm(k, q.t()).squeeze(1) # (TIME_STEPS)
        attention = nn.functional.softmax(energy, dim=-1) # (TIME_STEPS)
        context = torch.mm(attention.unsqueeze(0), v) # (1, HIDDEN_DIM)
        return context
    
    def forward(self, questions, answers, answer_lens, img):
        k, v, enc_hiddens = self.encode(questions)
        ans = self.decode(answers, answer_lens, k, v, enc_hiddens, img)
        return ans
    
    def predict(self, question, img, mappings, reset=False, max_len=20):
        k, v, h = self.encode(question)
        ans = self.decode_test(k[0], k[0], h, img, mappings, reset, max_len)
        return ans
    
    def decode_test(self, k, v, hidden, img, mappings, reset, max_len):
        ip = mappings["w2i"]["<sos>"] # 1, INPUT_DIM
        img_feats = self.feature_extractor(img.unsqueeze(0)) # (1, IMG_HIDDEN_DIM)
        
        if reset is True:
            self.history = torch.zeros((1, self.hidden_size * self.hidden_layers * 2)).to(self.device)

        answer = []
        for t in range(max_len):
            embs = self.embeddings(ip * torch.ones(1).long().to(self.device))
            x = embs.unsqueeze(0) # (1, 1, INPUT_DIM)
            
            x, hidden = self.answer_decoder(x, hidden)
            x = x.squeeze(0) # (1, INPUT_DIM)

            ques_context = self.attend(x, k, v)
            x = torch.cat([x, ques_context, self.history, img_feats], dim=-1) # (1, HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM + IMG_FEAT_DIM )
            x = self.answer_linear(x) # (1, VOCAB_SIZE)
            x = x.argmax(dim=-1)
            ip = x.item()
            answer.append(x.item())
            if x.item() == mappings["w2i"]["<eos>"]: break
        
        self.history = hidden[0].view((1, -1)).clone()
        return answer
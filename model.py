import torch
import torch.nn as nn

class VisualDialog(nn.Module):

    def __init__(self, feature_extractor):

        super(VisualDialog, self).__init__()

        # Parameters
        self.hidden_size = 128
        self.input_size  = 100
        self.hidden_layers = 2
        self.feature_extractor_dim = 512
        self.vocabulary_size = 1000

        # Image feture extractor
        self.feature_extractor = feature_extractor

        # LM
        self.embeddings = nn.Embedding(self.vocabulary_size, self.input_size)
        self.question_encoder = self.__init_encoder()
        self.question_keys = nn.Linear(128, 128)
        self.question_values = nn.Linear(128, 128)
        
        self.answer_decoder = self.__init_decoder()
        self.answer_linear = nn.Linear(
            self.hidden_size + self.hidden_size + self.feature_extractor_dim + self.hidden_size,
            self.vocabulary_size)
    
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
        return key, vals, h[:, -1, :].unsqueeze(1)
    
    def decode(self, x, xlens, enc_keys, enc_vals, enc_hiddens, img):
        """
        x: [Answers] 
            * Dim1) Number of answers
            * Dim2) Max answers Length
            * Padded in end

        """
        embs = self.embeddings(x) # NUM_ANS, MAX_ANS_LEN, INPUT_DIM
        img_feats = self.feature_extractor(img) # IMG_HIDDEN_DIM
        NUM_QUESTIONS, TIME_STEPS, HIDDEN_DIMS = embs.size()
        history = torch.zeros((1, 1, self.hidden_size))

        outputs = []
        for qno in range(NUM_QUESTIONS):
            hidden = enc_hiddens[qno].unsqueeze(0) # (1, 1, HIDDEN_DIM)
            k = enc_keys[qno]
            v = enc_keys[qno]

            answers = []
            for t in range(NUM_TIMESTEPS):
                if t >= xlens[qno]: break
 
                x = embs[qno, t, :].unsqueeze(0).unsqueeze(1) # (1, 1, INPUT_DIM)

                x, hidden = self.answer_decoder(x, hidden)
                
                ques_context = self.attend(x, k, v)
                x = torch.cat([x, ques_context, history, img_feats], dim=-1) # (1, 1, HIDDEN_DIM + HIDDEN_DIM + HIDDEN_DIM + IMG_FEAT_DIM )
                x = self.answer_linear(x) # (1, 1, VOCAB_SIZE)
                x = x.squeeze(0)

                answers.append(x)
            outputs.append(torch.stack(answers))
            history = hidden
        return outputs
    
    def forward(self, questions, answers, answer_lens, img):
        k, v, enc_hiddens = self.encode(questions)
        ans = self.decode(answers, answer_lens, k, v, enc_hiddens, img)
        return ans
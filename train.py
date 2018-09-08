import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from models import SkipGram, CBOW, NegativeSampling

class Trainer:
    def __init__(self, config, data_loader):
        self.model_type = config.model_type

        self.embedding_dim = config.embedding_dim
        self.window_size = config.window_size
        self.context_size = config.window_size * 2
        self.n_epoch = config.num_epochs
        self.lr = config.lr

        self.data = data_loader.make_id_pair()
        self.voca_len = len(data_loader)
        self.id2word = data_loader.id2word
        self.wordcount = data_loader.counter

        self.writer = SummaryWriter(log_dir="runs_%s"%self.model_type)

    def train(self):
        nllLoss = nn.NLLLoss()
        if self.model_type == "Skip-Gram":
            model = SkipGram(self.voca_len, self.embedding_dim, self.context_size).cuda()

        elif self.model_type == "CBOW":
            model = CBOW(self.voca_len, self.embedding_dim, self.context_size).cuda()

        elif self.model_type == "NS":
            model = NegativeSampling(self.voca_len, self.wordcount, self.embedding_dim, self.window_size).cuda()

        else:
            raise("Wrong model type! CBOW or Skip-Gram.")

        print(model)
        optimizer = optim.SGD(model.parameters(), lr=self.lr)

        losses = []

        # model.train()

        global_step = 0
        for epoch in range(self.n_epoch):
            total_loss = 0
            for step, (target, context) in enumerate(self.data):

                # Step 1. Prepare the inputs to be passed to the model (i.e, turn the words
                # into integer indices and wrap them in tensors)
                target = torch.tensor([target], dtype=torch.long).cuda()
                context = torch.tensor(context, dtype=torch.long).cuda()

                model.zero_grad()

                if self.model_type == "Skip-Gram":
                    log_probs = model(target)
                    loss = nllLoss(log_probs, context)

                elif self.model_type == "CBOW":
                    log_probs = model(context)
                    loss = nllLoss(log_probs, target)

                elif self.model_type == "NS":
                    loss = model(target,context)

                # Step 5. Do the backward pass and update the gradient
                loss.backward()
                optimizer.step()

                # for p in model.parameters():
                #     print(p.data)

                # Get the Python number from a 1-element Tensor by calling tensor.item()
                if global_step % 1000 == 0:
                    print("[%s/%s][%s/%s] loss %s"%(epoch,self.n_epoch,step,len(self.data),round(loss.item(),3)))
                    self.vis_log(model.embeddings.weight.data, loss.item(), global_step)

                total_loss += loss.item()
                global_step += 1
            losses.append(total_loss)

        self.writer.close()


    def vis_log(self, feat_mat, loss, global_step):

        meta = list(self.id2word.values())

        self.writer.add_embedding(feat_mat,meta,global_step=global_step)
        self.writer.add_scalar("Loss(lr=%s,nepoch=%s)"%(self.lr,self.n_epoch),loss,global_step=global_step)

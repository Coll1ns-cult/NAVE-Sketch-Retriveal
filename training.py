from .Siamese import Siamese, Transformation, Teacher
import torch 
import torch.nn as nn
from torch.functional import F
from torch.optim import optimizer
import timm

teacher = timm.create_model('Teacher', pretrained=True) #importing seresnet from timm library, this is pretrained model.
teacher.eval()


model= Siamese()
transformation_s = Transformation()
transformation_p1 = Transformation()
transformation_p2 = Transformation()

log_vars = []
log_vars_s = torch.zeros((1,), requires_grad= True)
log_vars_p = torch.zeros((1,), requires_grad= True)
log_vars = log_vars.append(log_vars_s, log_vars_p)


def MMFM(x, y): #modality mean feature measure 
    batch_x = x.shape[0]
    batch_y = y.shape[0]
    return (x*x).mean()/batch_x - (y*y).mean()/batch_y

def merge_loss(loss1, loss2, log_vars): #merging losses with above.
    loss = 0
    precision_1 = torch.exp(-log_vars[0])
    precision_2 = torch.exp(-log_vars[1])
    loss = torch.sum(precision_1*loss1 + log_vars[0], -1)
    loss += torch.sum(precision_2*loss2 + log_vars[1], -1)
    return loss

optimizer_adam = optimizer.Adam([{'params': model.parameters(), 'lr': 1e-3},
                            {'params': transformation_s.parameters()},
                            {'params': transformation_p1.parameters()},
                            {'params': transformation_p2.parameters()},
                            {'params': log_vars, 'lr': 1e-4}],
                            lr=1e-3, betas = (0.99, 0.99))

def train_loop(dataloader, model, loss_fn, optimizer_adam, regularizer, P, tau, 
                transformation_s, transformation_p1, transformation_p2):
    size = len(dataloader)
    for batch, X_s, y_s, X_p, y_p in enumerate(dataloader):
        f_s = model(X_s) #output of sketch siamese 
        pred_s = transformation_s(f_s) #transformation for cross entropy loss
        f_p = model(X_p) #output of photo siamese 
        pred_p = transformation_p1(f_p) #transformation for cross entropy loss
        student = transformation_p2(f_p) #transformation for student-teacher part
        teacher = teacher(y_p) #teacher network ouput 
        loss_ts = loss_fn(student, teacher) #loss of student teacher
        loss_s = loss_fn(pred_s, y_s) # cross entropy loss of sketch 
        loss_p = loss_fn(pred_p, y_p) #cross entropy loss of photo 
        loss_p += loss_ts
        loss_mmfm = MMFM(pred_s, pred_p) #modality mean feature norm discrepancy 


        if loss_mmfm - P < -tau:
            index_s = y_s.argmax(dim = 1) #index of true label for y_s
            y_s_tilda_choices = torch.cat(torch.arange(index_s), torch.arange((index_s + 1), y_s.shape[1])) #choices of random classes
            y_s_tilda_index = y_s_tilda_choices[:, torch.randint(y_s.shape[1])]
            y_s_tilda = F.one_hot(y_s_tilda_index, num_classes = y_s.shape[1]) #one hot encoding above index
            loss_s_reg = loss_fn(pred_s, y_s_tilda) #noisy loss
            loss_s = loss_s + regularizer*loss_s_reg #addition 
        

        elif loss_mmfm - P > tau:
            index_p = y_p.argmax(dim = 1)
            y_p_tilda_choices  = torch.cat(torch.arange(index_p), torch.arange((index_p + 1), y_p.shape[1]))
            y_p_tilda_index = y_p_tilda_choices[:, torch.randint(y_p.shape[1])]
            y_p_tilda = F.one_hot(y_p_tilda_index, num_classes = y_p.shape[1])
            loss_p_reg = loss_fn(pred_p, y_p_tilda)
            loss_p = loss_p + regularizer*loss_p_reg

    
        #sketch part 
        loss = merge_loss(loss_s, loss_p, log_vars)
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()

        if batch % 100 == 0:
            loss_p, current_p = loss_p.item(), batch * len(X_p)
            loss_s, current_s = loss_s.item(), batch * len(X_s)
            print(f"loss_p: {loss_p:>7f}  [{current_p:>5d}/{size:>5d}] |||| loss_s: {loss_s:>7f}  [{current_s:>5d}/{size:>5d}]")
    







    
        
        
        

        




        


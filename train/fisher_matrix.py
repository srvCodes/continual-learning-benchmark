from torch.autograd import Variable
import train.losses as losses

embeddings = []
def get_embeddings(self, inputs, outputs):
    global embeddings
    embeddings = inputs[0]

def fisher_matrix_diag(model, loss_name, train_loader, device):
    criterion = losses.create(loss_name).to(device)

    model.fc.register_forward_hook(get_embeddings)    # Init
    fisher = {}
    for n, p in model.named_parameters():
        fisher[n] = 0*p.data

    model.train()
    count = 0
    for i, data in enumerate(train_loader, 0):
        count += 1
        inputs, labels = data
        # wrap them in Variable
        inputs = Variable(inputs).to(device)
        labels = Variable(labels).to(device)

        # Forward and backward
        model.zero_grad()
        outputs = model(inputs.float())
        if loss_name == 'msloss':
            loss = criterion(embeddings, labels, device)
        else:
            loss, _, _, _ = criterion(embeddings, labels, device)
        loss.backward()

        for n, p in model.named_parameters():
            if p.grad is not None:
                fisher[n] += p.grad.data.pow(2)

    for n, _ in model.named_parameters():
        fisher[n] = fisher[n]/float(count)
        fisher[n] = Variable(fisher[n], requires_grad=False)
    return fisher

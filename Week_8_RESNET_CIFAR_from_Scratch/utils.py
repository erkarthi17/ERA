import torch

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    _, pred = output.topk(maxk, 1, True, True)
    correct = pred.eq(target.view(-1, 1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:, :k].sum().item()
        res.append(100.0 * correct_k / target.size(0))
    return res

def save_checkpoint(model, path):
    torch.save(model.state_dict(), path)

def log_epoch(epoch, train_acc, val_acc, loss, log_file):
    with open(log_file, "a") as f:
        f.write(f"### Epoch {epoch}\n")
        f.write(f"Train Acc: {train_acc:.2f}%\n")
        f.write(f"Val Acc: {val_acc:.2f}%\n")
        f.write(f"Loss: {loss:.4f}\n\n")
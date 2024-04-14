from torch.utils.data import Dataset

class mistral_Dataset(Dataset):
    def __init__(self, input_ids, attn_masks, labels):
        self.input_ids = input_ids
        self.attn_masks = attn_masks
        self.labels = labels

    def __getitem__(self, index):
        a = self.input_ids[index]
        b = self.attn_masks[index]
        c = self.labels[index]
        return {'input_ids': a, 'attention_mask': b, 'labels':c}

    def __len__(self):
        return len(self.input_ids)
    
    
class mistral_test_Dataset(Dataset):
    def __init__(self, input_ids, attn_masks):
        self.input_ids = input_ids
        self.attn_masks = attn_masks

    def __getitem__(self, index):
        a = self.input_ids[index]
        b = self.attn_masks[index]
        return {'input_ids': a, "attn_mask": b}

    def __len__(self):
        return len(self.input_ids)
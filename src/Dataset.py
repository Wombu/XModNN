import copy

from torch.utils import data
import numpy as np
import torch
import random

class Dataset(data.Dataset):
    def __init__(self, x=None, y=None, id=None, f=None):
        # Initialization
        # Dict, Keys -> feature names
        if x is not None:
            self.xs_ = {}
            for key, item in x.items():
                self.xs_[key] = []
                for item2 in item:
                    self.xs_[key].append(torch.tensor([item2]))
        else:
            None

        # single label:
        if y is not None:
            y = y[list(y.keys())[0]]
            self.label_list = sorted(list(set(y)))

            # find Indize of label
            self.y_indices = {l: [i for i, x in enumerate(y) if x == l] for l in self.label_list}

            # transform into dict of laben with true: 1, false: 0
            # TODO: unnötig? lieber male/female o.ä. behalten, brauche das in Loss Funktion (one hot encodung) eh?
            #self.ys_ = {l: [torch.tensor([1]) if x == l else torch.tensor([0]) for i, x in enumerate(y)] for l in list(label_set)}  #TODO: i wird nicht benötigt, oder?
            self.ys_ = {l: [torch.tensor([1]) if x == l else torch.tensor([0]) for x in y] for l in self.label_list}
        else:
            self.label_list = None
            self.ys_ = None
            self.y_indices = {}

        # TODO: Abfrage unnötig?
        if y != None:
            #keys = list(y.keys())
            #keys = list(label_set)
            self.length_ = len(y)
        else:
            self.length_ = 0

        # TODO: unnötig?
        """if (id != None):
            self.id_ = {}
            for i, item in enumerate(id):
                self.id_[item] = i"""

        #TODO: unnötig?
        """if f is not None:
            self.f_ = {}
            for i, item in enumerate(f):
                self.f_[item] = i"""

    def __len__(self):
        # Denotes the total number of samples
        return self.length_

    def delete(self, index):
        """for key, item in self.xs_.items():
            del self.xs_[key][index]

        for key, item in self.ys_.items():
            del self.ys_[key][index]"""

        if not isinstance(index, list):
            index = [index]

        index.sort()
        index.reverse()

        for index_single in index:
            for key, item in self.xs_.items():
                del self.xs_[key][index_single]
            for key, item in self.ys_.items():
                del self.ys_[key][index_single]
            self.refresh_label_indizes(index=index_single)

        keys = list(self.ys_.keys())
        self.length_ = len(self.ys_[keys[0]])

    def refresh_label_indizes(self, index):
        # Umständlich, alle größeren Indizes von dict, self.y_indizes müssen aktualisiert werden, wenn ein Element gelöscht wird.
        for key in self.y_indices.keys():  # Finden, in welches Label das Indize ist und aktualiesiere alle dahinterliegenden, weil die Postition des Indizes gelöscht wurde.
            try:
                pos_tmp = self.y_indices[key].index(index)
                for i in range(pos_tmp + 1, len(self.y_indices[key])):
                    self.y_indices[key][i] -= 1
                del self.y_indices[key][pos_tmp]
            except ValueError:  # Aktualisiere auch die Indizes aus den anderen Labeln, die hinter dem gelöschten Element liegen
                # test = [i for i in reversed(self.y_indices[key]) if i <= index]  # sieht schöner aus, ist aber langsamer.
                for pos_tmp, index_tmp in enumerate(self.y_indices[key]):
                    #pos_tmp = len(self.y_indices[key]) - pos_tmp
                    if index_tmp > index:
                        for i in range(pos_tmp, len(self.y_indices[key])):
                            self.y_indices[key][i] -= 1
                        break


    def __getitem__(self, index):
        # Generates one sample of data
        # Select sample
        if not isinstance(index, list):
            index = [index]

        x = {i: [] for i in list(self.xs_.keys())}
        y = {i: [] for i in list(self.ys_.keys())}
        for index_single in index:
            for key, item in self.xs_.items():
                if(len(index) == 1):
                    x[key] = item[index_single]
                else:
                    x[key].append(item[index_single])
            for key, item in self.ys_.items():
                if (len(index) == 1):
                    y[key] = item[index_single]
                else:
                    y[key].append(item[index_single])

        return x, y


    def split_set(self, split=None, indices=None, keep_balence=True):
        if(split!=None):
            split_indices = []
            if(keep_balence):
                for key_label, indices_tmp in self.y_indices.items():
                    indices_loop = copy.copy(indices_tmp)
                    len_split = int(len(indices_loop) * split)
                    for i in range(len_split):
                        rnd = random.randint(a=0, b=len(indices_loop) - 1)
                        split_indices.append(indices_loop[rnd])
                        del indices_loop[rnd]
            else:
                len_split = int(self.length_ * split)
                indices_loop = list(range(self.length_))
                for i in range(len_split):
                    rnd = random.randint(a=0, b=len(indices_loop) - 1)
                    split_indices.append((indices_loop[rnd]))
                    del indices_loop[rnd]

        if(indices!=None):
            split_indices = indices

        x_new, y_new = self[split_indices] #! irgendwo ist noch ein Fehler
        self.delete(split_indices)

        new_dataset = Dataset(x=x_new)

        new_dataset.label_list = list(y_new.keys())
        new_dataset.ys_ = y_new
        new_dataset.length_ = len(y_new[new_dataset.label_list[0]])
        for key, item in y_new.items():
            new_dataset.y_indices[key] = [i for i, item in enumerate(y_new[key]) if item == torch.tensor([1])]

        return new_dataset, split_indices

    def del_idices(self, indices):
        if not isinstance(indices, list):
            indices = [indices]
        indices.sort(reverse=True)

        for i in indices:
            for key in list(self.xs_.keys()):
                del self.xs_[key][i]
            for key in list(self.ys_.keys()):
                del self.ys_[key][i]

            self.refresh_label_indizes(index=i)

        keys = list(self.ys_.keys())
        self.length_ = len(self.ys_[keys[0]])

    def tensor_output(self):
        x_tensor_dict = {}
        for key, item in self.xs_.items():
            x_tensor_dict[key] = torch.tensor(item).unsqueeze_(1)

        y_tensor_dict = {}
        for key, item in self.ys_.items():
            y_tensor_dict[key] = torch.tensor(item).unsqueeze_(1)

        return x_tensor_dict, y_tensor_dict

    def set_y_indices(self, y):
        # Finde die Positionen der verschiedenen Label.
        #label_set = set(y.keys())
        label_set = set(y)
        """y_indices = {}
        for l in list(label_set):
            y_indices[l] = [i for i, x in enumerate(y[l]) if x == l]"""
        #test = {l: [i for i, x in enumerate(y[l]) if x == float(1)] for l in list(label_set)}
        test = {l: [i for i, x in enumerate(y) if x == l] for l in list(label_set)}
        return test

    def balance(self):
        l_min = float("Inf")
        for key, item in self.y_indices.items():
            if l_min > len(item):
                l_min = len(item)

        indices_del = []
        for key, item in self.y_indices.items():
            #y_indices_del[key] = []
            indices_tmp = copy.copy(item)
            while len(indices_tmp) > l_min:
                rnd = random.randint(0, len(indices_tmp)-1)
                indices_del.append(indices_tmp[rnd])
                del indices_tmp[rnd]
                #del self.y_indices[key][rnd]

        #self.del_idices(indices=indices_del)
        self.delete(indices_del)

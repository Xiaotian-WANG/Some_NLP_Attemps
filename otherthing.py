for i in range(train_dataset.__len__()):
    num_zeros = list(train_dataset[i]['input_ids'].shape)[0] - list(train_dataset[i]['labels'].shape)[0]
    temp = np.hstack((np.array(train_dataset[i]['labels']), (np.zeros(num_zeros, dtype=int)+tag2id['O'])))
    train_dataset[i]['labels'] = torch.tensor(temp)
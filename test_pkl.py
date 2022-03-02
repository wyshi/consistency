import pickle as pkl


for i in range(3):
    records = []
    record = {i: i + 1}
    records.append(record)
    with open("test_pkl.pkl", "rb") as fh:
        d = pkl.load(fh)
        print(d)
    records = d + records
    with open("test_pkl.pkl", "wb") as fh:
        print("here")
        pkl.dump(records, fh)

    with open("test_pkl.pkl", "rb") as fh:
        print(pkl.load(fh))
    print(records)

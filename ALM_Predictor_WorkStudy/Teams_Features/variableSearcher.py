import heapq
import torch
from torch.utils.data import DataLoader
from Utility import CSVDataset, Net, getTrainTest, compare
import concurrent.futures
from neural_net import train_model, evaluate, evaluate_model

# expand function for the search
def expand(train_dl, test_dl, config):
    model = Net(config["input"], config["hl1"], config["hl2"])
    model.train()
    train_model(train_dl, model, config)
    model.eval()
    mean = evaluate_model(test_dl, model)[0]
    return mean

# generates a unique id for the variables
def hash(vars, cols):
    ha = 0
    for v in vars:
        ha += 2 ** cols.index(v)
    return ha

def main():
    inPath = "SS20_complete.csv"
    hl1 = 90
    hl2 = 45
    num = 3     # number of models to make per comparison

    # please do not edit the code below unless you know what you are doing
    base = ["Weight (kg)", "Arm Volume Right", "Thigh Circumference Right",
            "MidThigh Circumference Right",
            "Surface Area Leg Left", "Arm Length Left", "Outside Leg Length Right", "Surface Area Arm Right",
            "Ankle Circumference Right", "Abdomen Circumference", "Surface Area Leg Right",
            "Bicep Circumference Left",
            "Bicep Circumference Right", "Calf Circumference Left",
            "Inseam Left",
            "Upper Arm Circumference Right", "Forearm Circumference Left",
            "Leg Volume Left", "Upper Arm Circumference Left",
            "Arm Length Right", "Calf Circumference Right", "Surface Area Arm Left",
            "Leg Volume Right", "Arm Volume Left", "Thigh Circumference Left",
            "MidThigh Circumference Left",
            "Inseam Right", "Outside Leg Length Left", "Forearm Circumference Right",
            "Ankle Circumference Left"]
    add = ["Volume", "Surface Area Total", "Subject Height",
            "Torso Volume", "Surface Area Torso", "Horizontal Waist", "Seat Circumference",
            "Collar Circumference", "Height (cm)",
            "Narrow Waist", "Waist Circumference", "Chest",
            "Head Circumference", "Hip Circumference", ]
    cols = ["Weight (kg)", "Arm Volume Right", "Thigh Circumference Right",
            "MidThigh Circumference Right",
            "Surface Area Leg Left", "Arm Length Left", "Outside Leg Length Right", "Surface Area Arm Right",
            "Ankle Circumference Right", "Volume", "Abdomen Circumference", "Surface Area Leg Right",
            "Bicep Circumference Left",
            "Bicep Circumference Right", "Calf Circumference Left", "Surface Area Total", "Subject Height",
            "Torso Volume", "Surface Area Torso", "Horizontal Waist", "Inseam Left", "Seat Circumference",
            "Upper Arm Circumference Right", "Collar Circumference", "Forearm Circumference Left", "Height (cm)",
            "Leg Volume Left", "Narrow Waist", "Waist Circumference", "Chest", "Upper Arm Circumference Left",
            "Arm Length Right", "Calf Circumference Right", "Surface Area Arm Left",
            "Head Circumference", "Leg Volume Right", "Arm Volume Left", "Thigh Circumference Left",
            "MidThigh Circumference Left",
            "Inseam Right", "Hip Circumference", "Outside Leg Length Left", "Forearm Circumference Right",
            "Ankle Circumference Left"]

    device = torch.device("cpu")
    dftrain, dftest = getTrainTest(inPath, ["ALM (adjusted)"] + base, standardize=True)
    train_dl = DataLoader(CSVDataset(dftrain, device), batch_size=64, shuffle=False)
    test_dl = DataLoader(CSVDataset(dftest, device), batch_size=64, shuffle=False)
    config = {
        "input": len(base),
        "hl1": hl1,
        "hl2": hl2,
        "lr": 0.001,
        "epochs": 100
    }
    sums = 0
    for i in range(num):
        model = Net(config["input"], config["hl1"], config["hl2"])
        model.train()
        train_model(train_dl, model, config)
        model.eval()
        mean, median, std, Max = evaluate_model(test_dl, model)
        sums += mean

    states = [compare(base, sums / num)]
    print(states[0])
    unique = [hash(base, cols)]
    final = []

    # search for improvement variables
    # takes the average of 3 models to see if it is an improvement
    print("Begin Search")
    while len(states) > 0:
        print("States:\t\t\t", len(states))
        print("Final:\t\t\t", len(final))
        print("Top State:\n", states[0])
        tup = heapq.heappop(states)
        mean = tup.val
        state = tup.var
        no_new = True
        for a in add:
            if a in state:
                # skips iteration if variable is already in the state
                continue
            new = state + [a]
            config = {
                "input": len(new),
                "hl1": hl1,
                "hl2": hl2,
                "lr": 0.001,
                "epochs": 100
            }
            if hash(new, cols) in unique:
                continue
            dftrain, dftest = getTrainTest(inPath, ["ALM (adjusted)"] + new, standardize=True)
            train_dl = DataLoader(CSVDataset(dftrain, device), batch_size=64, shuffle=False)
            test_dl = DataLoader(CSVDataset(dftest, device), batch_size=64, shuffle=False)

            # multiple processing to speed up the process
            with concurrent.futures.ProcessPoolExecutor() as executor:
                results = []
                for i in range(num):
                    results.append(executor.submit(expand, train_dl, test_dl, config))
                sums = 0
                for f in concurrent.futures.as_completed(results):
                    ret = f.result()
                    sums += ret
                new_mean = sums / num
            if new_mean < mean:
                heapq.heappush(states, compare(new, new_mean))
                unique.append(hash(new, cols))
                no_new = False

        if no_new:
            final.append(compare(state, mean))

        print("END ITERATION\n")
    print("SORTING")
    final.sort()

    for f in final:
        print(f)
        print()
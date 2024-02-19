import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from Utility import compare, CSVDataset, Net, standardize
import gc
from neural_net import train_model, evaluate_model

# mix the 2 models together
# the child is a mix of the 2 parents

def breed(model1, model2):
    from random import Random
    rand = Random()
    bias = rand.random()
    model3 = []
    for m1, m2 in zip(model1, model2):
        layer3 = []
        for n1, n2 in zip(m1, m2):
            row3 = []
            for o1, o2 in zip(n1, n2):
                if rand.random() > bias:
                    row3.append(o2)
                else:
                    row3.append(o1)
            layer3.append(row3)
        model3.append(layer3)
    return model3

# turns model into a list of matrices
# I named it item
def toItem(model):
    item = []
    for p in model.parameters():
        item.append(p.tolist())
    item[1] = [item[1]]
    item[3] = [item[3]]
    item[5] = [item[5]]
    return item

# takes a list of matrices and turns it into a model
def toModel(item):
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
    config = {
        "input": len(cols),
        "hl1": 90,
        "hl2": 45,
        "lr": 0.001,
        "epochs": 100
    }
    # manual overwriting of the layers
    model = Net(config["input"], config["hl1"], config["hl2"])
    model.hidden1.weight = torch.nn.parameter.Parameter(torch.tensor(item[0]))
    model.hidden1.bias = torch.nn.parameter.Parameter(torch.tensor(item[1][0]))
    model.hidden2.weight = torch.nn.parameter.Parameter(torch.tensor(item[2]))
    model.hidden2.bias = torch.nn.parameter.Parameter(torch.tensor(item[3][0]))
    model.hidden3.weight = torch.nn.parameter.Parameter(torch.tensor(item[4]))
    model.hidden3.bias = torch.nn.parameter.Parameter(torch.tensor(item[5][0]))
    model.eval()
    return model

# returns the top 10, 89 randoms of the middle, and 1 of the bottom 10
def fittest(population):
    rand = random.Random()
    top = population[:10]
    middle = population[10:len(population) - 10]
    bottom = population[len(population) - 10:]
    rand.shuffle(middle)
    rand.shuffle(bottom)
    new = top + middle[:89] + [bottom[0]]
    del population
    del top
    del middle
    del bottom
    return new

# scorer I made
# mean + std + max / 10
def score(data):
   return data[0] + data[1] + data[2] / 10

def main():
    inPath = "LR-Mice_Filled_Train_SS20_Dataset427.csv"
    outModel = "Cell.pth"

    # NOTE: this process is extremely RAM intensive
    # I optimized this function to fit my PC to handle this
    # my PC had 8 GB of RAM
    # all of the gc.collect()'s and del statements are to clean up the RAM

    # please, do not edit the code below unless you know what you are doing
    cols = ["Height (cm)","Weight (kg)","Abdomen Circumference","Ankle Circumference Left","Arm Length Left",
        "Arm Volume Left","Bicep Circumference Left","Calf Circumference Left","Chest","Collar Circumference",
        "Forearm Circumference Left","Head Circumference","Hip Circumference","Horizontal Waist","Inseam Left",
        "Leg Volume Left","MidThigh Circumference Left","Narrow Waist","Outside Leg Length Left","Seat Circumference",
        "Surface Area Arm Left","Surface Area Leg Left","Surface Area Torso","Surface Area Total","Thigh Circumference Left",
        "Torso Volume","Volume","Waist Circumference","Subject Height","Ankle Circumference Right","Arm Length Right",
        "Arm Volume Right","Bicep Circumference Right","Calf Circumference Right","Forearm Circumference Right",
        "Inseam Right","Leg Volume Right","MidThigh Circumference Right","Outside Leg Length Right",
        "Surface Area Arm Right","Surface Area Leg Right","Thigh Circumference Right"]
    device = torch.device("cpu")
    config = {
        "input": len(cols),
        "hl1": 90,
        "hl2": 45,
        "lr": 0.001,
        "epochs": 100
    }
    dftrain = pd.read_csv(inPath, usecols=["ALM (adjusted)"] + cols)
    dftrain, means, stds = standardize(dftrain)
    dftest = pd.read_csv(inPath, usecols=["ALM (adjusted)"] + cols)
    dftest = standardize(dftest, means, stds)[0]

    train_dl = DataLoader(CSVDataset(dftrain, device), batch_size=64, shuffle=False)
    test_dl = DataLoader(CSVDataset(dftest, device), batch_size=64, shuffle=False)
    models = []

    print("Training Initial Generation")
    for i in range(100):
        model = Net(config["input"], config["hl1"], config["hl2"])
        model.train()
        train_model(train_dl, model, config)
        model.eval()
        models.append(model)
        del model
    gc.collect()

    # breed the first 100
    population = [toItem(m) for m in models]
    del models
    gc.collect()
    n = len(population)
    for i in range(n-1):
        for j in range(i+1, n):
            m1, m2 = breed(population[i], population[j])
            population.append(m1)
            population.append(m2)

    # begin making the new models
    # it saves the best model each generation
    for i in range(10000):
        print(f"Generation {i}")
        print(f"\tTesting")

        models = [toModel(p) for p in population]
        del population
        gc.collect()


        results = [compare(toItem(m), evaluate_model(test_dl, m)[0]) for m in models]
        del models
        gc.collect()

        results.sort()
        best = toModel(results[0].var)
        mean, std, Max, diff = evaluate_model(test_dl, best)
        print(f"\tBest Model: \n\t\tMean:\t{mean} \n\t\tSTD: \t{std} \n\t\tMax: \t{Max}")
        del best
        del mean
        del std
        del Max
        del diff
        from neural_net import save_model
        save_model(toModel(results[0].var), outModel)

        population = [r.var for r in results]
        del results
        gc.collect()

        print("\tSelecting Models")
        population = fittest(population)
        print("\tRepopulating")
        n = len(population)
        for i in range(n - 1):
            for j in range(i + 1, n):
                m1 = breed(population[i], population[j])
                population.append(m1)
                if j % 100 == 0:
                    gc.collect()
        del n
        gc.collect()
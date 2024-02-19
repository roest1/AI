import math
import pandas as pd

# uses TOTAL = L Arm + R Arm + L Leg + R Leg + Torso
def fillExact(data):
    cols = ["Total", "Arm Left", "Arm Right", "Leg Left", "Leg Right", "Torso"]
    # ^ for reference
    # in the case of more than one nan, it will just fill with nan
    #       because nan + FLOAT = nan
    for d in data:
        for i in range(len(d)):
            if math.isnan(d[i]):
                if i == 0:
                    d[i] = sum(d[1:])
                else:
                    corrected = d[0]
                    for j in range(1,len(d)):
                        if i != j:
                            corrected -= d[j]
                    d[i] = corrected
    return data

# fills just the Left and Right Volume and Surface Area measurements
def fillLR(data):
    cols = ["Total", "Arm Left", "Arm Right", "Leg Left", "Leg Right", "Torso"]
    # ^ for reference
    LR = {1: 2, 2: 1, 3: 4, 4: 3}
    for d in data:
        for i in range(1,5):
            if math.isnan(d[i]):
                d[i] = d[LR[i]]
    return data

# in the case that just both arms or both legs are missing
# it assumes that they are the same and divides them evenly
def fillDual(data):
    cols = ["Total", "Arm Left", "Arm Right", "Leg Left", "Leg Right", "Torso"]
    # ^ for reference
    for d in data:
        if math.isnan(d[1]) and math.isnan(d[2]):
            d[1] = d[2] = (d[0] - d[3] - d[4] - d[5]) / 2
        if math.isnan(d[3]) and math.isnan(d[4]):
            d[3] = d[4] = (d[0] - d[1] - d[2] - d[5]) / 2
    return data

def main():
    inPath = r"SS20_incomplete.csv"
    outPath = r"SS20 Exact LR Fill Incomplete.csv"

    # please do not change the code below unless you know what you are doing
    cols = ["Arm Length Left", "Arm Length Right", "Surface Area Arm Left", "Surface Area Arm Right",
            "Arm Volume Left", "Arm Volume Right", "Forearm Circumference Left", "Forearm Circumference Right",
            "Bicep Circumference Left", "Bicep Circumference Right", "Upper Arm Circumference Left",
            "Upper Arm Circumference Right", "Ankle Circumference Left", "Ankle Circumference Right",
            "Calf Circumference Left", "Calf Circumference Right", "MidThigh Circumference Left",
            "MidThigh Circumference Right", "Thigh Circumference Left", "Thigh Circumference Right", "Inseam Left",
            "Inseam Right", "Outside Leg Length Left", "Outside Leg Length Right", "Leg Volume Left",
            "Leg Volume Right",
            "Surface Area Leg Left", "Surface Area Leg Right", "Seat Circumference", "Hip Circumference",
            "Waist Circumference", "Horizontal Waist", "Narrow Waist", "Abdomen Circumference", "Collar Circumference",
            "Chest", "Surface Area Torso", "Torso Volume", "Head Circumference", "Subject Height", "Age", "Height (cm)",
            "Weight (kg)", "Surface Area Total", "Volume"]
    AC = ["Surface Area Total","Surface Area Arm Left", "Surface Area Arm Right",
          "Surface Area Leg Left", "Surface Area Leg Right","Surface Area Torso"]
    VC = ["Volume","Arm Volume Left", "Arm Volume Right",
          "Leg Volume Left", "Leg Volume Right","Torso Volume"]
    ALL = pd.read_csv(inPath, usecols=cols)
    Acol = ALL.columns.values.tolist()
    SA = ALL[AC]
    VOL = ALL[VC]
    # the order is: Formula, LR, Formula, Dual
    sa = fillDual(fillExact(fillLR(fillExact(SA.values.tolist()))))
    vol = fillDual(fillExact(fillLR(fillExact(VOL.values.tolist()))))
    # I know that the nested function calls look ugly
    # it just saves space

    # fill back in
    all = ALL.values.tolist()
    for i in range(len(all)):
        for j in range(len(AC)):
            all[i][Acol.index(AC[j])] = sa[i][j]
            all[i][Acol.index(VC[j])] = vol[i][j]
    ALLFIXED = pd.DataFrame(all)
    ALLFIXED.columns = Acol

    # fill the remaining LR values
    for column in ALLFIXED.columns:
        if 'Left' in column:
            other = column.replace('Left', 'Right')
            ALLFIXED[column] = ALLFIXED[column].fillna(ALLFIXED[other])
        elif 'Right' in column:
            other = column.replace('Right', 'Left')
            ALLFIXED[column] = ALLFIXED[column].fillna(ALLFIXED[other])
    ALLFIXED.to_csv(outPath)
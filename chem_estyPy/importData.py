import pandas as pd

Tinc = 1/6
NISTsubstances = pd.read_csv("NIST_data.csv").to_numpy()

fluidIDsList = NISTsubstances[:, 0]
fluidNamesList = NISTsubstances[:, 1]

fluidNames = {}
for fluidID, fluidName in zip(fluidIDsList, fluidNamesList):
    fluidNames[fluidID] = fluidName

fluidNames = {fluidID: fluidName for fluidID, fluidName in zip(fluidIDsList, fluidNamesList)}


def readFluidNames(fileName):
    NISTsubstances = pd.read_csv(fileName).to_numpy()

    fluidIDsList = NISTsubstances[:, 0]
    fluidNamesList = NISTsubstances[:, 1]

    fluidNames = {fluidID: fluidName for fluidID, fluidName in zip(fluidIDsList, fluidNamesList)}
    return fluidNames

# Define set
dataSet = pd.DataFrame()
for g in range(3):
    print(g+1)
    for j in range(15):
        print(j + 1)
        ## Po słowniku można se potem iterować
        for fluidId, fluidName in fluidNames.items():

            # Download data
            NistFluidUrl = \
                'https://webbook.nist.gov/cgi/fluid.cgi?Action=Load&ID=' + \
                fluidId \
                + '&Type=IsoBar&Digits=5&P=' \
                + str(j+1) \
                + '&THigh=' + str(g * 100+273) + '&' \
                + 'TLow=' + str((g + 1) * 100 + 273) + '&' \
                + 'TInc=' + str(Tinc) + '&' \
                + 'RefState=DEF&TUnit=K&PUnit=bar&DUnit=kg%2Fm3&HUnit=kJ%2Fmol&WUnit=m%2Fs&VisUnit=Pa*s&STUnit=N%2Fm'
            tableNist = pd.read_html(NistFluidUrl)
            fluidProp = tableNist[0]
            print(len(fluidProp))

            fluidProp['Fluid'] = fluidName

            # Remove unnecessary columns
            fluidProp.pop('Volume (m3/kg)')
            dataSet = pd.concat([dataSet, fluidProp], ignore_index=True)

print(dataSet)
dataSet.to_csv('sieciNeuronoweWInzynieriiProcesowej.csv', header=True)
print("DONE!!!")

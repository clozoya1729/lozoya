parameters, start, end = [], [], []

class textParser():

    def __init__(self):
        with open('instructions.txt', 'r') as iFile:        #The parameters required for extraction are read from the "instructions" file
            for line in iFile:
                parameters.append(line.rstrip())
        self.selectFile()

    def selectFile(self):

        path = parameters[0]                                #The parameters required for the extraction are read from the parameters array
        path = path [:(path.__len__()-4)]
        newPath = path + "Reduced.txt"
        path = path + ".txt"
        iterations = int(parameters[1])


        for i in range(2 * iterations):

            if not (i%2):                                   #Reading the starting and ending points of each extraction
                start.append(parameters[i + 2])
            else:
                end.append(parameters[i + 2])

        self.openFile(path, newPath, iterations)


    def openFile(self, path, newPath, iterations):

        #rFile = open(path, 'r')                         #Open the file in read mode
        with open(path, 'r') as rFile:
            self.readFile(newPath, rFile, iterations)     #Call the read file function with the opened file as the argument


    def readFile(self, newPath, rFile, iterations):

        i = 0
        reducedData = []
        parsing = False

        for line in rFile:

            tempS = start[i]            #Store the current iteration's starting point
            tempE = end[i]              #Store the current iteration's ending point
            smallS = 0
            smallE = 0
            word = ''
            if (parsing == False and tempS in line and tempE in line):  # Look for the starting token in each line of the document
                parsing = True
                print(line)
                for w in range(line.__len__()):
                    if (line[w] == tempE[0]):
                        w += 1
                        if tempE.__len__() == 1:
                            smallE = w
                        else:
                            for k in range(1, tempE.__len__()):
                                if (line[w] == tempE[k]):
                                    w += 1
                                    k += 1

                                    if (k == tempE.__len__()):
                                        smallE = w
                                        break
                                else:
                                    break
                        if (k == tempE.__len__()):
                            break

                word = line[:smallE]
                reducedData.append(word.strip())
                parsing = False

                if (i < (iterations - 1)):
                    i += 1
                else:
                    self.createFile(newPath, reducedData)
                    return

            elif (parsing == False and tempS in line and tempE not in line):           #Look for the starting token in each line of the document
                parsing = True
                print("Start")
                for w in range(line.__len__()):
                    if (line[w] == tempS[0]):
                        smallS = w
                        w += 1
                        if tempS.__len__() == 1:
                            pass
                        else:
                            for k in range(1, (tempS.__len__())):
                                if (line[w] == tempS[k]):
                                    w+=1
                                    k+=1

                                    if (k == tempS.__len__()): break
                                else: break
                        if (k == tempS.__len__()):
                            break

                word = line[smallS:]
                reducedData.append(word.strip())

            elif(parsing == True and tempE not in line):
                reducedData.append(line)

            elif(parsing == True and tempE in line):
                print("end")
                for w in range(line.__len__()):
                    if (line[w] == tempE[0]):
                        w += 1
                        if tempE.__len__() == 1:
                            smallE = w
                        else:
                            for k in range(1, tempE.__len__()):
                                if (line[w] == tempE[k]):
                                    w+=1
                                    k+=1

                                    if (k == tempE.__len__()):
                                        smallE = w
                                        break
                                else: break
                        if (k == tempE.__len__()):
                            break

                word = line[:smallE]
                reducedData.append(word.strip())
                parsing = False

                if (i < (iterations - 1)):
                    i += 1
                else:
                    self.createFile(newPath, reducedData)
                    return

    def start_end(self, tempList, tempList_length, tKey):
        for j in range(tempList_length):
            if tempList[j] == tKey:
                return j


    def createFile(self, newPath, reducedData):

        print('New file created.')
        wFile = open(newPath, "w")                           #Create a new file to store the reduced data
        self.writeFile(reducedData, wFile)                   #Call the write file function with the reduced data and nnew file as arguments


    def writeFile(self, reducedData, wFile):
        wFile.write("---------------------------------\n")
        for i in range(reducedData.__len__()):
            wFile.write(reducedData[i] + "\n")               #Write the reduced data into the new file
            i +=1
        wFile.write("---------------------------------\n")
        wFile.close()

#tP = textParser()
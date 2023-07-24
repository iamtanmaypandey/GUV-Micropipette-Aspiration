import easygui as eg
import time
import pandas as pd

#define app title
apptitle = 'SMBL-MPA Data App'

#temp file
allrows = []

index = 0
    
#ask for the frame rate
fps = int(eg.enterbox('Enter the frame rate: ',apptitle))

#ask if the video record is on
state = eg.ynbox('Click yes to start the timer',apptitle,('Yes','No'))
if state:
    initime = time.time_ns()
    initial = time.asctime()
    while True:
        pressure = eg.multenterbox('Enter the parameters \n or write end to stop it.',apptitle,['LSR','CSR'])
        if pressure[0] == 'end' or pressure[1]== 'end':
            df = pd.DataFrame(allrows,columns=['Sno','Initial Time (s)', 'Final Time (s)','Delta T (s)','Frame No (approx)','LSR','CSR'])
            path = eg.diropenbox('Select the directory where you want to save the csv file', apptitle)
            name = eg.enterbox('Enter the name for csv file',apptitle)
            df.to_csv(f'{path}/{name}.csv')
            print('File Saved ! \n Program Completed.')
            break
        else:
            currtime = time.time_ns()
            index += 1
            total = currtime-initime
            total_sec = total/(10**9)
            frame = int(fps*total_sec)
            final = time.asctime()
            row = []
            print(initial,final)
            row.append(index)
            row.append(initial)
            row.append(final)
            row.append(total_sec)
            row.append(frame)
            row.append(pressure[0])
            row.append(pressure[1])
            allrows.append(row)
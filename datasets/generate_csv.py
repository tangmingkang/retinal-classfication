import os
import pandas as pd
import random
file_dir = "./image"
files = os.listdir(file_dir)
image_name=[file[:-4] for file in files]
diagnosis=[]
fold=[]
for name in image_name:
    dir=int(name[3:5])
    fold.append(random.randint(0,9))
    if dir>=31 and dir<=33:
        diagnosis.append(0)
    else:
        diagnosis.append(1)
datas={"image_name" : image_name, "diagnosis" : diagnosis, "fold" : fold}#将列表a，b转换成字典
df=pd.DataFrame(datas)
df.to_csv('label.csv',sep=',',index=False,header=True)

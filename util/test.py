import pandas as pd

data = {'A':[1,2,3],'B':[3,2,1]}
df = pd.DataFrame(data)
df['amt'] = (df.A>=df.B)+0
print(df)
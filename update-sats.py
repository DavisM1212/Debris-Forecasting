import pandas as pd
DATA_PATH = './Data/'
satcat_url = 'https://celestrak.org/pub/satcat.csv'
satcat_out = 'satcat.csv'
df = pd.read_csv(satcat_url)
df.to_csv(DATA_PATH + satcat_out, index=False)
print(f"Updated CSV from '{satcat_url}' successfully saved to '{satcat_out}'")
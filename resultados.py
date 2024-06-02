import sys
import sqlite3 

import pandas as pd

dbconn = sqlite3.connect(sys.argv[1])

a = pd.read_sql('SELECT * FROM resultados',dbconn)

print(a)

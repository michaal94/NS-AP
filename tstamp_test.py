from datetime import datetime
import os

with open(os.path.join('.', 'timestamp.txt'), 'w') as f:
    now = datetime.now()
    f.write(
        f"{now.strftime('%Y/%m/%d, %H:%M:%S')}\n{datetime.timestamp(now)}"
    )
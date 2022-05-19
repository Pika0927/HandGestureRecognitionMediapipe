import time
import struct

from datetime import datetime

f = open(r'\\.\pipe\NPGesture', 'r+b', 0)
i = 1

while True:
    s = ('HiHi'+str(i)).encode('ascii')
    i += 1
        
    f.write(struct.pack('I', len(s)) + s)   # Write str length and str
    print(datetime.now().time())
    f.seek(0)                               # EDIT: This is also necessary
    print('Wrote:',s)

    time.sleep(2)
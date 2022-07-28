from bytewax import Dataflow
from ForcastCore import *
from Ftx import FtxClient

client = FtxClient('Your FTX Ket','Your FTX Secret')
startDate = datetime.datetime(2022,7,1,0,0,0)
endDate = datetime.datetime.utcnow()
unixStartDate = calendar.timegm(startDate.utctimetuple())
unixEndDate =  calendar.timegm(endDate.utctimetuple())

Bytewaxflow = Dataflow()
Bytewaxflow.map(getOpenValue)
Bytewaxflow.capture()

correctChoice=True
while correctChoice:
    choice=input("Choose which market focast you want  \n 1 - BTC/USD \n 2 - ETH/USD \n your pick :  ")
    if int(choice) == 1:
        cryptoForcast('BTC','BTC/USD', Bytewaxflow, client,  unixStartDate, unixEndDate)
        break;
    elif int(choice)==2:
        cryptoForcast('ETH','BTC/USD', Bytewaxflow, client,  unixStartDate, unixEndDate)
        break;
    print('Wrong choice, please input either 1 or 2')
    
